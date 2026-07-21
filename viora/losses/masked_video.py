"""Masked video modeling loss (feature-space regression).

A genuine masked-prediction objective: given predictions and targets for the
masked token positions, minimise smooth-L1 over those positions only. It is *not*
a stub — but note it needs a masking + prediction pipeline (the caller masks
tubelet tokens and provides a lightweight predictor); that Stage-1 wiring lives in
the trainer, not here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def masked_feature_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``pred``/``target``: ``[B, N, D]``; ``mask``: ``[B, N]`` bool (True = predict here)."""
    per_token = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)  # [B, N]
    denom = mask.sum().clamp_min(1)
    return (per_token * mask).sum() / denom


class MaskedVideoModelingLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_feature_loss(pred, target, mask)
