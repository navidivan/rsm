from typing import Any, Tuple, Dict, Sequence, Optional
import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    # Stable transform used by stablemax
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)

    safe_labels = torch.where(valid_mask, labels, 0).to(torch.long)
    picked = torch.gather(logprobs, index=safe_labels.unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, picked, torch.zeros_like(picked))


def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # valid_mask is accepted for API parity; ignore_index handles masking.
    flat_logits = logits.to(torch.float32).view(-1, logits.shape[-1])
    flat_labels = labels.to(torch.long).view(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="none")
    return loss.view(labels.shape)


class LossHead(nn.Module):
    """
    Thin loss+metrics wrapper.

    Contract:
      - wraps a model that returns: (new_carry, outputs_dict)
      - outputs_dict must include "logits" of shape [B, SeqLen, V]
    """
    def __init__(self, model: nn.Module, loss_type: str = "softmax_cross_entropy"):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        logits = outputs["logits"]

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            outputs["preds"] = preds

            mask = (labels != IGNORE_LABEL_ID)
            token_counts = mask.sum(-1)                                # [B]
            divisor = token_counts.clamp_min(1).unsqueeze(-1)          # [B,1]

            is_correct_tok = mask & (preds == labels)
            seq_is_correct = (is_correct_tok.sum(-1) == token_counts) & (token_counts > 0)

            valid_seq = (token_counts > 0)
            count = valid_seq.sum().to(torch.float32).clamp_min(1.0)

            accuracy = torch.where(
                valid_seq,
                (is_correct_tok.to(torch.float32) / divisor).sum(-1),
                torch.zeros_like(token_counts, dtype=torch.float32),
            ).sum()

            exact_accuracy = seq_is_correct.sum().to(torch.float32)

            metrics = {
                "count": count,
                "accuracy": accuracy,
                "exact_accuracy": exact_accuracy,
            }

        lm_loss = (self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / divisor).sum()
        metrics["lm_loss"] = lm_loss.detach()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        total_loss = lm_loss
        finished = torch.tensor(True, device=lm_loss.device)

        return new_carry, total_loss, metrics, detached_outputs, finished
