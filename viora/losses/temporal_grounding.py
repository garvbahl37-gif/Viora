"""Temporal grounding loss over discretised start/end bins.

Cross-entropy on the start bin and the end bin (the discretised formulation of the
grounding head). :meth:`targets_from_seconds` converts ground-truth ``(start,
end)`` seconds to bin indices given the clip duration and bin count.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TemporalGroundingLoss(nn.Module):
    def forward(
        self,
        start_logits: torch.Tensor,  # [B, num_bins]
        end_logits: torch.Tensor,    # [B, num_bins]
        start_bin: torch.Tensor,     # [B] long
        end_bin: torch.Tensor,       # [B] long
    ) -> torch.Tensor:
        return 0.5 * (F.cross_entropy(start_logits, start_bin) + F.cross_entropy(end_logits, end_bin))

    @staticmethod
    def targets_from_seconds(
        start_s: torch.Tensor, end_s: torch.Tensor, duration: torch.Tensor, num_bins: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``(start, end)`` seconds to bin indices in ``[0, num_bins)``."""
        frac_s = (start_s / duration.clamp_min(1e-6)).clamp(0, 1)
        frac_e = (end_s / duration.clamp_min(1e-6)).clamp(0, 1)
        sb = (frac_s * (num_bins - 1)).round().long()
        eb = (frac_e * (num_bins - 1)).round().long()
        return sb, eb
