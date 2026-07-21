"""Temporal order prediction loss.

Given two event representations, predict whether event A precedes event B — a
self-supervisable signal (order is known from timestamps) that teaches the model
temporal directionality. The small ordering classifier is owned by the loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TemporalOrderLoss(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def logits(self, event_a: torch.Tensor, event_b: torch.Tensor) -> torch.Tensor:
        """``[N, D]``, ``[N, D]`` -> ``[N]`` logit that A precedes B."""
        return self.classifier(torch.cat([event_a, event_b], dim=-1)).squeeze(-1)

    def forward(
        self, event_a: torch.Tensor, event_b: torch.Tensor, a_before_b: torch.Tensor
    ) -> torch.Tensor:
        """``a_before_b``: ``[N]`` float in {0,1}."""
        return F.binary_cross_entropy_with_logits(
            self.logits(event_a, event_b), a_before_b.float()
        )
