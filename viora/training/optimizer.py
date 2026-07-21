"""Optimizer construction with sane weight-decay grouping.

Norm/bias/embedding parameters are excluded from weight decay (standard for
transformers); everything else decays. Only trainable parameters are included, so
a frozen vision encoder or LLM contributes nothing.
"""

from __future__ import annotations

import torch
from torch import nn

from viora.utils.config import TrainingConfig


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower() or "embed" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas))
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9)
    raise ValueError(f"unknown optimizer '{cfg.optimizer}' (adamw|sgd)")
