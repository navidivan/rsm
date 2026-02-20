from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class RSMCarry:
    current_data: Dict[str, torch.Tensor]


class RSMConfig(BaseModel):
    model_config = {"extra": "allow"}

    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int

    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16

    # Recurrence depth and inner compute
    H_cycles: int
    L_cycles: int
    L_layers: int

    # Block options
    mlp_t: bool = False
    hidden_size: int
    expansion: float
    num_heads: int

    pos_encodings: str = "none"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"

    # Stochastic Depth
    prob_detach_prev_H: float = 0.5

    freeze_emb_prc: float = 0.0
    freeze_lm_head_prc: float = 0.0


class RSMBlock(nn.Module):
    def __init__(self, cfg: RSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm_eps = cfg.rms_norm_eps

        if cfg.mlp_t:
            seq_width = cfg.seq_len + self._puzzle_len(cfg)
            self.mlp_t = SwiGLU(hidden_size=seq_width, expansion=cfg.expansion)
        else:
            self.self_attn = Attention(
                hidden_size=cfg.hidden_size,
                head_dim=cfg.hidden_size // cfg.num_heads,
                num_heads=cfg.num_heads,
                num_key_value_heads=cfg.num_heads,
                causal=False,
            )
        self.mlp = SwiGLU(hidden_size=cfg.hidden_size, expansion=cfg.expansion)

    @staticmethod
    def _puzzle_len(cfg: RSMConfig) -> int:
        if cfg.puzzle_emb_ndim <= 0: return 0
        return (cfg.puzzle_emb_ndim + cfg.hidden_size - 1) // cfg.hidden_size if cfg.puzzle_emb_len == 0 else cfg.puzzle_emb_len

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        if self.cfg.mlp_t:
            hs = hidden_states.transpose(1, 2)
            out = self.mlp_t(hs)
            hs = out.transpose(1, 2)
            hidden_states = rms_norm(hidden_states + hs, variance_epsilon=self.norm_eps)
        else:
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)

        mlp_out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)
        return hidden_states


class RSMModule(nn.Module):
    def __init__(self, layers: List[RSMBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, injection: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        hidden_states = hidden_states + injection
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states


class RSM(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.cfg = RSMConfig(**config_dict)
        self.forward_dtype = getattr(torch, self.cfg.forward_dtype)

        self.embed_scale = math.sqrt(self.cfg.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.cfg.vocab_size, self.cfg.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        self.puzzle_emb_len = 0
        if self.cfg.puzzle_emb_ndim > 0:
            self.puzzle_emb_len = (self.cfg.puzzle_emb_ndim + self.cfg.hidden_size - 1) // self.cfg.hidden_size if self.cfg.puzzle_emb_len == 0 else self.cfg.puzzle_emb_len
            self.puzzle_emb = CastedSparseEmbedding(
                self.cfg.num_puzzle_identifiers, self.cfg.puzzle_emb_ndim,
                batch_size=self.cfg.batch_size, init_std=0, cast_to=self.forward_dtype,
            )

        if self.cfg.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.cfg.hidden_size // self.cfg.num_heads,
                max_position_embeddings=self.cfg.seq_len + self.puzzle_emb_len,
                base=self.cfg.rope_theta,
            )
        elif self.cfg.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.cfg.seq_len + self.puzzle_emb_len, self.cfg.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        self.core = RSMModule(layers=[RSMBlock(self.cfg) for _ in range(self.cfg.L_layers)])
        self.lm_head = CastedLinear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)

        # WEIGHT TYING
        self.lm_head.weight = self.embed_tokens.embedding_weight

        h0 = trunc_normal_init_(torch.empty(self.cfg.hidden_size, dtype=self.forward_dtype), std=1.0)
        l0 = trunc_normal_init_(torch.empty(self.cfg.hidden_size, dtype=self.forward_dtype), std=1.0)
        self.register_buffer("H_init", h0, persistent=True)
        self.register_buffer("L_init", l0, persistent=True)

    @property
    def puzzle_embedding_module(self):
        return getattr(self, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> RSMCarry:
        return RSMCarry(current_data={k: torch.empty_like(v) for k, v in batch.items()})

    def _input_embeddings(self, tokens: torch.Tensor, puzzle_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed_tokens(tokens.to(torch.int32))
        if self.cfg.puzzle_emb_ndim > 0:
            pz = self.puzzle_emb(puzzle_ids)
            pad = self.puzzle_emb_len * self.cfg.hidden_size - pz.shape[-1]
            if pad > 0: pz = F.pad(pz, (0, pad))
            pz = pz.view(-1, self.puzzle_emb_len, self.cfg.hidden_size)
            emb = torch.cat((pz, emb), dim=-2)
        if self.cfg.pos_encodings == "learned":
            emb = 0.707106781 * (emb + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * emb

    def forward(
        self,
        carry: RSMCarry,
        batch: Dict[str, torch.Tensor],
        override_H_cycles: Optional[int] = None,
        debug_structure: bool = False,
        alpha: float = 1.0,
    ) -> Tuple[RSMCarry, Dict[str, torch.Tensor]]:

        B = batch["inputs"].shape[0]
        S = self.cfg.seq_len + self.puzzle_emb_len

        H_req = int(override_H_cycles) if override_H_cycles is not None else int(self.cfg.H_cycles)

        # [FIX] Enforce min-depth=2 ONLY during training. Allow H=1 during inference.
        H = max(2, H_req) if self.training else H_req

        idx_final = H - 1

        # Stochastic Logic
        include_prev_H = True
        if self.training and self.cfg.prob_detach_prev_H > 0.0:
            if torch.rand(1).item() < self.cfg.prob_detach_prev_H:
                include_prev_H = False

        if debug_structure:
            print(f"\n" + "="*80)
            print(f"[ARCH AUDIT] H={H} | L={self.cfg.L_cycles} | Train Transition? {include_prev_H}")
            print(f"  > Batch: {B} | Seq: {S} | Hidden: {self.cfg.hidden_size}")

        # 1. Inputs
        input_emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        z_H = self.H_init.view(1, 1, -1).expand(B, S, self.cfg.hidden_size)
        z_L = self.L_init.view(1, 1, -1).expand(B, S, self.cfg.hidden_size)

        # 2. Prepare RoPE (FIXED: Moved calculation here)
        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos_sin = self.rotary_emb()

        for h in range(H):
            # A. Determine Mode
            is_active = (h == idx_final)
            is_transition = (h == idx_final - 1 and include_prev_H and H >= 2)

            # B. Determine Detachment
            # We detach history if we are NOT chaining from a valid transition
            should_detach = True
            if is_active and include_prev_H and H >= 2:
                 should_detach = False

            if should_detach:
                z_H = z_H.detach()
                z_L = z_L.detach()

            # C. Determine Gradient Requirements
            train_inner = is_active
            train_outer = is_active or is_transition

            if debug_structure:
                mode_str = "ACTIVE" if is_active else ("TRANSITION" if is_transition else "WARMUP")
                print(f"  > Step {h} [{mode_str}]:")
                print(f"    - InputDetached: {not z_H.requires_grad}")
                print(f"    - Train Inner:   {train_inner} (Memory Heavy)")
                print(f"    - Train Outer:   {train_outer} (Compute Heavy)")

            with torch.autocast(device_type="cuda", dtype=self.forward_dtype, enabled=True):

                # Inner Loop
                with torch.set_grad_enabled(train_inner):
                    for l in range(self.cfg.L_cycles):
                        # Reinjection Audit
                        if debug_structure and l == 0:
                            combined = z_H + input_emb
                            print(f"    - [L_Cycle 0] Reinjection Check:")
                            print(f"      - z_H Grad: {z_H.requires_grad} | input_emb Grad: {input_emb.requires_grad}")
                            print(f"      - Combined GradFn: {type(combined.grad_fn).__name__ if combined.grad_fn else 'None'}")
                            if not train_inner and combined.grad_fn is not None:
                                print("      - [FAIL] Zombie Graph Detected! GradFn exists in NoGrad mode.")
                            elif not train_inner:
                                print("      - [PASS] Zombie Graph Dead. (GradFn is None)")

                        z_L = self.core(z_L, z_H + input_emb, cos_sin=cos_sin)

                # Outer Update
                with torch.set_grad_enabled(train_outer):
                    z_H = self.core(z_H, z_L, cos_sin=cos_sin)

        # Final prediction
        with torch.autocast(device_type="cuda", dtype=self.forward_dtype, enabled=True):
            logits_out = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        if debug_structure:
            print("="*80 + "\n")

        return RSMCarry(current_data=batch), {"logits": logits_out}

