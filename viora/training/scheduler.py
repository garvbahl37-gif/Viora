"""Learning-rate schedules with linear warmup (cosine / linear / constant).

Implemented as a ``LambdaLR`` multiplier so it composes with any optimizer and
checkpoints via the standard scheduler ``state_dict``.
"""

from __future__ import annotations

import math

import torch

from viora.utils.config import TrainingConfig


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainingConfig, total_steps: int | None = None
) -> torch.optim.lr_scheduler.LambdaLR:
    total = total_steps or cfg.max_steps
    warmup = max(0, cfg.warmup_steps)
    min_ratio = cfg.min_lr / cfg.lr if cfg.lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        if cfg.scheduler == "cosine":
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if cfg.scheduler == "linear":
            return min_ratio + (1 - min_ratio) * (1 - progress)
        if cfg.scheduler == "constant":
            return 1.0
        raise ValueError(f"unknown scheduler '{cfg.scheduler}'")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
