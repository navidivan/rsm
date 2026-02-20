from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math
import yaml
import copy
import datetime
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig

    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []

    # DataLoader settings
    num_workers: int = 4
    prefetch_factor: int = 2

    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    grad_clip_norm: float = 1.0

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names & Paths
    project_name: Optional[str] = "DefaultProject"
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Resume
    resume_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False

    # Eval (steps)
    eval_interval: int = 2000
    min_eval_interval: int = 0
    eval_save_outputs: List[str] = []

    short_eval_interval: int = 0
    short_eval_fraction: float = 0.05

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False

    # Curriculum - H Cycles (Outer Recurrence)
    add_H_every_percent: float = 0.0
    curriculum_milestones: List[float] = []      # Thresholds for H
    curriculum_heads_to_add: List[int] = []      # Amount to add to H

    # Curriculum - L Cycles (Inner Compute)
    curriculum_L_milestones: List[float] = []    # Thresholds for L
    curriculum_L_to_add: List[int] = []          # Amount to add to L

    # Transition smoothing
    local_warmup_steps: int = 100              # alpha fade 0->1 after H changes
    transition_lr_warmup_steps: int = 0        # LR multiplier 0->1 after H changes (0 disables)

    # Optimizer state management on growth
    optimizer_reset_scale: float = 0.0

    # Checkpoints by percent
    save_checkpoints_at_percent: List[float] = []


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int

    active_H: int = 1
    prev_H: int = 1

    active_L: int = 1   # New: Track L cycles
    prev_L: int = 1     # New: Track L changes

    last_transition_step: int = -10**9

    # Throughput tracking
    last_log_time: float = 0.0


def get_target_H(cfg: PretrainConfig, step: int, total_steps: int) -> int:
    """Calculate target H_cycles based on curriculum."""
    base_h = int(getattr(cfg.arch, "H_cycles", 1))

    # Explicit Milestones
    if len(cfg.curriculum_milestones) > 0 and len(cfg.curriculum_heads_to_add) > 0:
        pct = (step / max(1, total_steps)) * 100.0
        current_h = base_h
        milestones = sorted(zip(cfg.curriculum_milestones, cfg.curriculum_heads_to_add), key=lambda x: x[0])
        for threshold, add in milestones:
            if pct >= threshold:
                current_h += int(add)
        return current_h

    # Linear Growth
    if cfg.add_H_every_percent > 0.0:
        pct_progress = step / max(1, total_steps)
        blocks = int((pct_progress * 100.0) / cfg.add_H_every_percent)
        return base_h + blocks

    return base_h


def get_target_L(cfg: PretrainConfig, step: int, total_steps: int) -> int:
    """Calculate target L_cycles based on curriculum."""
    # Default to config L_cycles if not specified
    base_l = int(getattr(cfg.arch, "L_cycles", 1))

    if len(cfg.curriculum_L_milestones) > 0 and len(cfg.curriculum_L_to_add) > 0:
        pct = (step / max(1, total_steps)) * 100.0
        current_l = base_l
        milestones = sorted(zip(cfg.curriculum_L_milestones, cfg.curriculum_L_to_add), key=lambda x: x[0])
        for threshold, add in milestones:
            if pct >= threshold:
                current_l += int(add)
        return current_l

    return base_l


def create_dataloader(cfg: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=cfg.seed,
            dataset_paths=cfg.data_paths_test if (len(cfg.data_paths_test) > 0 and split == "test") else cfg.data_paths,
            rank=rank,
            num_replicas=world_size,
            epochs_per_iter=1,
            **kwargs,
        ),
        split=split,
    )

    workers = getattr(cfg, "num_workers", 4)
    prefetch = getattr(cfg, "prefetch_factor", 2)

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        prefetch_factor=prefetch,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, dataset.metadata


def create_model(cfg: PretrainConfig, train_meta: PuzzleDatasetMetadata, rank: int, world_size: int):
    arch_dict = cfg.arch.model_dump() if hasattr(cfg.arch, "model_dump") else cfg.arch

    model_cfg = dict(
        **arch_dict,
        batch_size=cfg.global_batch_size // world_size,
        vocab_size=train_meta.vocab_size,
        seq_len=train_meta.seq_len,
        num_puzzle_identifiers=train_meta.num_puzzle_identifiers,
        causal=False,
    )

    if "loss" in model_cfg and not isinstance(model_cfg["loss"], dict):
        model_cfg["loss"] = model_cfg["loss"].model_dump()

    model_cls = load_model_class(cfg.arch.name)
    loss_head_cls = load_model_class(cfg.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(f" [MODEL] Created Architecture: L_layers={getattr(model.cfg, 'L_layers', '?')}, H_cycles={getattr(model.cfg, 'H_cycles', '?')}")

        loss_kwargs = cfg.arch.loss.model_dump()
        loss_kwargs.pop("name", None)

        model = loss_head_cls(model, **loss_kwargs)

        print("\n[INFO] torch.compile disabled.")
        if world_size > 1:
            with torch.no_grad():
                for t in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(t, src=0)

    # Optimizers
    if cfg.arch.__pydantic_extra__.get("puzzle_emb_ndim", 0) == 0:
        opts = [AdamATan2(model.parameters(), lr=0.0, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2))]
        lrs = [cfg.lr]
    elif cfg.freeze_weights:
        opts = [CastedSparseEmbeddingSignSGD_Distributed(model.model.puzzle_emb.buffers(), lr=0.0, weight_decay=cfg.puzzle_emb_weight_decay, world_size=world_size)]
        lrs = [cfg.puzzle_emb_lr]
    else:
        opts = [
            CastedSparseEmbeddingSignSGD_Distributed(model.model.puzzle_emb.buffers(), lr=0.0, weight_decay=cfg.puzzle_emb_weight_decay, world_size=world_size),
            AdamATan2(model.parameters(), lr=0.0, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2)),
        ]
        lrs = [cfg.puzzle_emb_lr, cfg.lr]

    return model, opts, lrs


def cosine_with_warmup(current_step: int, *, base_lr: float, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if current_step < warmup_steps:
        return base_lr * float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * 2.0 * 0.5 * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * max(0.0, cosine))


def lr_transition_mult(cfg: PretrainConfig, ts: TrainState) -> float:
    if cfg.transition_lr_warmup_steps <= 0:
        return 1.0
    dt = ts.step - ts.last_transition_step
    if dt <= 0:
        return 0.0
    if dt >= cfg.transition_lr_warmup_steps:
        return 1.0
    return float(dt) / float(cfg.transition_lr_warmup_steps)


def compute_lr(cfg: PretrainConfig, ts: TrainState, base_lr: float) -> float:
    scheduled = cosine_with_warmup(
        current_step=ts.step,
        base_lr=base_lr,
        warmup_steps=int(round(cfg.lr_warmup_steps)),
        total_steps=ts.total_steps,
        min_ratio=cfg.lr_min_ratio,
    )
    return scheduled * lr_transition_mult(cfg, ts)


def init_train_state(cfg: PretrainConfig, train_meta: PuzzleDatasetMetadata, rank: int, world_size: int) -> TrainState:
    total_steps = int(cfg.epochs * train_meta.total_groups * train_meta.mean_puzzle_examples / cfg.global_batch_size)
    model, opts, lrs = create_model(cfg, train_meta, rank=rank, world_size=world_size)

    # Initialize active H and L from start
    active_H = get_target_H(cfg, 0, total_steps)
    active_L = get_target_L(cfg, 0, total_steps)

    return TrainState(
        model=model, optimizers=opts, optimizer_lrs=lrs, carry=None,
        step=0, total_steps=total_steps,
        active_H=active_H, prev_H=active_H,
        active_L=active_L, prev_L=active_L
    )


def save_train_state(cfg: PretrainConfig, ts: TrainState, filename_override: Optional[str] = None):
    if cfg.checkpoint_path is None:
        return
    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    filename = filename_override if filename_override else f"ckpt_step_{ts.step}.pth"
    path = os.path.join(cfg.checkpoint_path, filename)

    torch.save(
        {
            "model_state": ts.model.state_dict(),
            "optimizer_states": [o.state_dict() for o in ts.optimizers],
            "step": ts.step,
            "active_H": ts.active_H,
            "prev_H": ts.prev_H,
            "active_L": ts.active_L,
            "prev_L": ts.prev_L,
            "last_transition_step": ts.last_transition_step,
            "config": cfg.model_dump(),
            "total_steps": ts.total_steps,
        },
        path,
    )
    print(f" [SAVE] Checkpoint saved: {path}")

    with open(os.path.join(cfg.checkpoint_path, "latest_config.yaml"), "w") as f:
        yaml.dump(cfg.model_dump(), f)


def create_evaluators(cfg: PretrainConfig, eval_meta: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = cfg.data_paths_test if len(cfg.data_paths_test) > 0 else cfg.data_paths
    evaluators = []
    for e in cfg.evaluators:
        for p in data_paths:
            cls = load_model_class(e.name, "evaluators.")
            evaluators.append(cls(data_path=p, eval_metadata=eval_meta, **e.__pydantic_extra__))
    return evaluators


def train_batch(cfg: PretrainConfig, ts: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    ts.step += 1
    batch = {k: v.cuda() for k, v in batch.items()}

    if ts.carry is None:
        with torch.device("cuda"):
            ts.carry = ts.model.initial_carry(batch)

    # Curriculum update
    ts.active_H = get_target_H(cfg, ts.step, ts.total_steps)
    ts.active_L = get_target_L(cfg, ts.step, ts.total_steps)

    debug_structure = False
    optimizer_reset_needed = False

    # Force debug on FIRST step
    if ts.step == 1:
        debug_structure = True

    # Detect Curriculum Change (H or L)
    changed_H = (ts.active_H != ts.prev_H)
    changed_L = (ts.active_L != ts.prev_L)

    if changed_H or changed_L:
        debug_structure = True
        optimizer_reset_needed = True
        ts.last_transition_step = ts.step

        if rank == 0:
            pct = (ts.step / max(1, ts.total_steps)) * 100.0
            print(f"\n[DEBUG] Growth step @ {pct:.2f}% | New H={ts.active_H} (Was {ts.prev_H}) | New L={ts.active_L} (Was {ts.prev_L})")

        ts.prev_H = ts.active_H
        ts.prev_L = ts.active_L

    # Log Speed & Memory
    if rank == 0 and ts.step % 20 == 0:
        now = time.time()
        if ts.last_log_time > 0:
            dt = now - ts.last_log_time
            samples_per_sec = (20 * global_batch_size) / dt
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f" [PERF] Step {ts.step} | Throughput: {int(samples_per_sec)} samples/s | Peak Mem: {mem_gb:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        ts.last_log_time = now

    # Freeze logic
    model_container = ts.model
    if hasattr(model_container, "model"):
        model_container = model_container.model

    pct = (ts.step / max(1, ts.total_steps)) * 100.0
    freeze_emb = getattr(model_container.cfg, "freeze_emb_prc", 0.0)
    freeze_lm = getattr(model_container.cfg, "freeze_lm_head_prc", 0.0)

    if freeze_emb > 0.0 and pct >= freeze_emb:
        if model_container.embed_tokens.embedding_weight.requires_grad:
            if rank == 0:
                print(f"\n[DEBUG] Freezing token/puzzle embeddings @ {pct:.2f}% (threshold {freeze_emb}%)")
            model_container.embed_tokens.embedding_weight.requires_grad = False
            if hasattr(model_container, "puzzle_emb"):
                for p in model_container.puzzle_emb.parameters():
                    p.requires_grad = False
            debug_structure = True

    if freeze_lm > 0.0 and pct >= freeze_lm:
        if model_container.lm_head.weight.requires_grad:
            if rank == 0:
                print(f"\n[DEBUG] Freezing LM head @ {pct:.2f}% (threshold {freeze_lm}%)")
            model_container.lm_head.weight.requires_grad = False
            if getattr(model_container.lm_head, "bias", None) is not None:
                model_container.lm_head.bias.requires_grad = False
            debug_structure = True

    # Optimizer state scaling
    if optimizer_reset_needed:
        scale = cfg.optimizer_reset_scale
        if rank == 0:
            print(f" ☢️  OPTIMIZER STATE: scale={scale}")

        if scale < 1.0 and cfg.transition_lr_warmup_steps <= 0:
             if rank == 0:
                print("\n[WARNING] ⚠️  DANGER: Optimizer state reset (scale < 1.0) without LR warmup!")
                print("          This will likely cause gradient explosion. Set transition_lr_warmup_steps > 0.\n")

        for opt in ts.optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    st = opt.state.get(p, None)
                    if st is None:
                        continue
                    if "exp_avg" in st:
                        st["exp_avg"].mul_(scale)
                    if "exp_avg_sq" in st:
                        st["exp_avg_sq"].mul_(scale)

    # Alpha fade
    alpha = 1.0
    if cfg.local_warmup_steps > 0:
        dt = ts.step - ts.last_transition_step
        if 0 <= dt <= cfg.local_warmup_steps:
            alpha = float(dt) / float(cfg.local_warmup_steps)

    if hasattr(ts.model, "model") and hasattr(ts.model.model, "cfg"):
        # Unwrap Loss Head -> Model -> Config
        ts.model.model.cfg.L_cycles = ts.active_L
    elif hasattr(ts.model, "cfg"):
        # Direct Model
        ts.model.cfg.L_cycles = ts.active_L

    carry, loss, metrics, outputs, _finished = ts.model(
        carry=ts.carry,
        batch=batch,
        return_keys=[],
        override_H_cycles=ts.active_H,
        debug_structure=debug_structure,
        alpha=alpha,
    )

    if rank == 0:
        metrics["alpha_fade"] = torch.tensor(alpha, device="cuda", dtype=torch.float32)
        metrics["lr_transition_mult"] = torch.tensor(lr_transition_mult(cfg, ts), device="cuda", dtype=torch.float32)

    ((1.0 / global_batch_size) * loss).backward()

    if world_size > 1:
        for p in ts.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)

    if cfg.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(ts.model.parameters(), cfg.grad_clip_norm)

    lr_main = compute_lr(cfg, ts, cfg.lr)

    for i, opt in enumerate(ts.optimizers):
        base_lr = ts.optimizer_lrs[i]
        lr_i = compute_lr(cfg, ts, base_lr)

        for pg in opt.param_groups:
            pg["lr"] = lr_i
            if i == 0 and len(ts.optimizers) > 1:
                pg["weight_decay"] = cfg.puzzle_emb_weight_decay
            else:
                pg["weight_decay"] = cfg.weight_decay

        opt.step()
        opt.zero_grad()

    # Reduce metrics
    if len(metrics) == 0:
        return None

    keys = sorted(metrics.keys())
    vals = torch.stack([metrics[k] for k in keys])
    if world_size > 1:
        dist.reduce(vals, dst=0)

    if rank != 0:
        return None

    vals = vals.cpu().numpy()
    m = {k: vals[i] for i, k in enumerate(keys)}

    count = max(float(m.get("count", 1.0)), 1.0)
    reduced = {}
    for k, v in m.items():
        if k.endswith("loss"):
            reduced[f"train/{k}"] = float(v) / float(global_batch_size)
        else:
            reduced[f"train/{k}"] = float(v) / count

    reduced["train/lr"] = float(lr_main)
    reduced["train/curriculum_H"] = float(ts.active_H)
    reduced["train/curriculum_L"] = float(ts.active_L)

    return reduced


def evaluate(cfg: PretrainConfig, ts: TrainState, loader, meta, evaluators, rank: int, world_size: int, cpu_group=None, max_batches: Optional[int] = None):
    if loader is None or meta is None:
        return None

    reduced_metrics = None
    with torch.inference_mode():
        return_keys = set(cfg.eval_save_outputs)
        for ev in evaluators:
            ev.begin_eval()
            return_keys.update(getattr(ev, "required_outputs", set()))

        set_ids = {k: i for i, k in enumerate(meta.sets)}
        metric_keys = None
        metric_values = None
        processed = 0

        # Update L_cycles for Eval
        if hasattr(ts.model, "model") and hasattr(ts.model.model, "cfg"):
            ts.model.model.cfg.L_cycles = ts.active_L
        elif hasattr(ts.model, "cfg"):
            ts.model.cfg.L_cycles = ts.active_L

        for set_name, batch, _gb in loader:
            processed += 1
            if max_batches is not None and processed > max_batches:
                break

            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = ts.model.initial_carry(batch)

            carry, loss, metrics, preds, _finished = ts.model(
                carry=carry,
                batch=batch,
                return_keys=return_keys,
                override_H_cycles=ts.active_H,
                debug_structure=False,
                alpha=1.0,
            )

            for ev in evaluators:
                ev.update_batch(batch, preds)

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = sorted(metrics.keys())
                metric_values = torch.zeros((len(set_ids), len(metric_keys)), dtype=torch.float32, device="cuda")

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                arr = metric_values.cpu().numpy()
                reduced_metrics = {}
                for set_name, sid in set_ids.items():
                    mm = {metric_keys[j]: float(arr[sid, j]) for j in range(len(metric_keys))}
                    count = max(mm.pop("count", 1.0), 1.0)
                    reduced_metrics[set_name] = {k: (v / count) for k, v in mm.items()}

        if rank == 0 and len(evaluators) > 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            for ev in evaluators:
                out = ev.result(None, rank=rank, world_size=world_size, group=cpu_group)
                if out is not None:
                    if reduced_metrics is None:
                        reduced_metrics = {}
                    reduced_metrics.update(out)

    return reduced_metrics


def load_synced_config(hydra_cfg: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objs = [None]
    if rank == 0:
        cfg = PretrainConfig(**hydra_cfg)

        if cfg.project_name is None:
            cfg.project_name = f"{os.path.basename(cfg.data_paths[0]).capitalize()}-RSM"

        if cfg.run_name is None:
            if cfg.resume_path:
                cfg.run_name = f"RESUME_{coolname.generate_slug(2)}"
            else:
                cfg.run_name = f"{cfg.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"

        DRIVE_ROOT = "/content/drive/MyDrive/TinyRecursiveRuns"
        if os.path.exists("/content/drive"):
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = f"{cfg.run_name}_{stamp}"
            cfg.checkpoint_path = os.path.join(DRIVE_ROOT, cfg.project_name, run_dir)
        else:
            cfg.checkpoint_path = os.path.join("checkpoints", cfg.project_name, cfg.run_name)

        objs = [cfg]

    if world_size > 1:
        dist.broadcast_object_list(objs, src=0)

    return objs[0]


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_cfg: DictConfig):
    rank = 0
    world_size = 1
    cpu_group = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        cpu_group = dist.new_group(backend="gloo")

    cfg = load_synced_config(hydra_cfg, rank=rank, world_size=world_size)
    torch.random.manual_seed(cfg.seed + rank)

    train_loader, train_meta = create_dataloader(cfg, "train", test_set_mode=False, global_batch_size=cfg.global_batch_size, rank=rank, world_size=world_size)

    try:
        eval_loader, eval_meta = create_dataloader(cfg, "test", test_set_mode=True, global_batch_size=cfg.global_batch_size, rank=rank, world_size=world_size)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader, eval_meta = None, None

    try:
        evaluators = create_evaluators(cfg, eval_meta) if eval_meta is not None else []
    except:
        evaluators = []

    ts = init_train_state(cfg, train_meta, rank=rank, world_size=world_size)

    # Resume
    if cfg.resume_path:
        loc = f"cuda:{rank}"
        ckpt = torch.load(cfg.resume_path, map_location=loc)
        ts.model.load_state_dict(ckpt["model_state"])
        if "optimizer_states" in ckpt:
            for i, opt in enumerate(ts.optimizers):
                if i < len(ckpt["optimizer_states"]):
                    opt.load_state_dict(ckpt["optimizer_states"][i])

        ts.step = int(ckpt.get("step", 0))
        ts.active_H = int(ckpt.get("active_H", 1))
        ts.prev_H = int(ckpt.get("prev_H", ts.active_H))

        # New L state
        ts.active_L = int(ckpt.get("active_L", 1))
        ts.prev_L = int(ckpt.get("prev_L", ts.active_L))

        ts.last_transition_step = int(ckpt.get("last_transition_step", -10**9))

        print(f" [RESUME] step={ts.step} | H={ts.active_H} | L={ts.active_L}")

    if rank == 0:
        bar = tqdm.tqdm(total=ts.total_steps, initial=ts.step)
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=cfg.model_dump(), settings=wandb.Settings(_disable_stats=True))
        print(f"STARTING TRAINING for {ts.total_steps} STEPS (Current: {ts.step}).")
        print(f"Checkpoints planned at: {sorted(cfg.save_checkpoints_at_percent)}%")
        print(f"Saving to: {cfg.checkpoint_path}")

    ema_helper = None
    if cfg.ema:
        if rank == 0:
            print("Setup EMA")
        ema_helper = EMAHelper(mu=cfg.ema_rate)
        ema_helper.register(ts.model)

    train_iter = iter(train_loader)
    milestones = sorted(cfg.save_checkpoints_at_percent)
    next_ckpt_idx = 0

    current_pct = (ts.step / max(1, ts.total_steps)) * 100.0
    while next_ckpt_idx < len(milestones) and current_pct >= milestones[next_ckpt_idx]:
        next_ckpt_idx += 1

    while ts.step < ts.total_steps:
        try:
            set_name, batch, gb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            set_name, batch, gb = next(train_iter)

        ts.model.train()
        metrics = train_batch(cfg, ts, batch, gb, rank=rank, world_size=world_size)

        # Checkpoints by percent
        current_pct = (ts.step / max(1, ts.total_steps)) * 100.0
        while next_ckpt_idx < len(milestones) and current_pct >= milestones[next_ckpt_idx]:
            thr = milestones[next_ckpt_idx]
            if rank == 0:
                print(f"\n[INFO] Crossed {thr}% (Current: {current_pct:.2f}%). Saving checkpoint.")
                save_train_state(cfg, ts, filename_override=f"ckpt_pct_{int(thr)}.pth")
            next_ckpt_idx += 1

        if rank == 0 and metrics is not None:
            wandb.log(metrics, step=ts.step)
            bar.update(1)

        if ema_helper is not None:
            ema_helper.update(ts.model)

        # Short eval
        if cfg.short_eval_interval > 0 and (ts.step % cfg.short_eval_interval == 0):
            if eval_loader is not None and eval_meta is not None:
                if rank == 0:
                    print(f"SHORT EVAL (Step {ts.step})")

                ts_eval = ts
                if ema_helper is not None:
                    ts_eval = copy.deepcopy(ts)
                    ts_eval.model = ema_helper.ema_copy(ts_eval.model)

                ts_eval.model.eval()

                total_examples = eval_meta.total_groups
                total_batches = math.ceil(total_examples / cfg.global_batch_size)
                limit = max(1, int(total_batches * cfg.short_eval_fraction))

                em = evaluate(cfg, ts_eval, eval_loader, eval_meta, evaluators, rank=rank, world_size=world_size, cpu_group=cpu_group, max_batches=limit)
                if rank == 0 and em is not None:
                    wandb.log(em, step=ts.step)

        # Full eval
        if cfg.eval_interval > 0 and (ts.step % cfg.eval_interval == 0):
            if eval_loader is not None and eval_meta is not None:
                if rank == 0:
                    print(f"FULL EVAL (Step {ts.step})")

                ts_eval = ts
                if ema_helper is not None:
                    ts_eval = copy.deepcopy(ts)
                    ts_eval.model = ema_helper.ema_copy(ts_eval.model)

                ts_eval.model.eval()
                if rank == 0:
                    save_train_state(cfg, ts_eval)

                total_examples = eval_meta.total_groups
                total_batches = math.ceil(total_examples / cfg.global_batch_size)
                limit = max(1, int(total_batches * 0.10))

                em = evaluate(cfg, ts_eval, eval_loader, eval_meta, evaluators, rank=rank, world_size=world_size, cpu_group=cpu_group, max_batches=limit)
                if rank == 0 and em is not None:
                    wandb.log(em, step=ts.step)

    if dist.is_initialized():
        dist.destroy_process_group()

    wandb.finish()


if __name__ == "__main__":
    launch()

