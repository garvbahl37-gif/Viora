"""Answer-confidence head.

Produces a scalar score in ``(0, 1)`` from a fused answer/evidence representation.
It is explicitly an **uncalibrated model score**: without a calibration objective
(temperature scaling, etc.) it must not be reported as a probability. The
inference layer surfaces ``score_type="uncalibrated_model_score"`` accordingly.
"""

from __future__ import annotations

import torch
from torch import nn


class ConfidenceHead(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or dim // 2
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.calibrated = False  # flipped only once a calibration step has been fit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, D] -> [B]`` uncalibrated score in (0, 1)."""
        return torch.sigmoid(self.net(x)).squeeze(-1)
