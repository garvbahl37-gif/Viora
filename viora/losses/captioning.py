"""Captioning / language-modeling loss (next-token cross-entropy).

The LLM adapter already returns a loss when given ``labels``; this module computes
the same quantity from raw ``logits``/``labels`` for paths that need it explicitly
(e.g. auxiliary caption heads), with the standard causal shift and ``-100`` mask.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def captioning_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """``logits``: ``[B, L, V]``; ``labels``: ``[B, L]``. Shifted next-token CE."""
    shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    if not (shift_labels != ignore_index).any():
        return logits.sum() * 0.0
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)


class CaptioningLoss(nn.Module):
    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return captioning_loss(logits, labels, self.ignore_index)
