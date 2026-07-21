"""Temporal grounding head — Viora returns *evidence*, not only answers.

A discretised temporal-bin formulation (more stable than direct regression):
temporal features are resampled to ``num_bins`` positions, conditioned on a query
representation, and scored for start/end. Predictions are returned as normalised
positions in ``[0, 1]`` (the caller maps to seconds using the clip duration) plus
an **uncalibrated** confidence — never presented as a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from viora.models.common import MLP
from viora.utils.config import GroundingConfig


@dataclass
class GroundingOutput:
    start_logits: torch.Tensor    # [B, num_bins]
    end_logits: torch.Tensor      # [B, num_bins]
    start_norm: torch.Tensor      # [B] predicted start in [0,1]
    end_norm: torch.Tensor        # [B] predicted end in [0,1]
    confidence: torch.Tensor      # [B] uncalibrated model score

    def to_seconds(self, duration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map normalised predictions to seconds given clip ``duration`` ``[B]``."""
        return self.start_norm * duration, self.end_norm * duration


class TemporalGroundingHead(nn.Module):
    def __init__(self, cfg: GroundingConfig, dim: int, query_dim: int | None = None) -> None:
        super().__init__()
        self.num_bins = cfg.num_bins
        self.q_proj = nn.Linear(query_dim or dim, dim)
        self.fuse = MLP(dim, cfg.hidden_dim)
        self.start = nn.Linear(dim, 1)
        self.end = nn.Linear(dim, 1)

    def forward(
        self, temporal_features: torch.Tensor, query: torch.Tensor
    ) -> GroundingOutput:
        """``temporal_features``: ``[B, T', D]``; ``query``: ``[B, query_dim]``."""
        # resample the time axis to a fixed number of bins
        f = temporal_features.transpose(1, 2)  # [B, D, T']
        f = F.interpolate(f, size=self.num_bins, mode="linear", align_corners=False)
        f = f.transpose(1, 2)  # [B, num_bins, D]

        h = self.fuse(f + self.q_proj(query).unsqueeze(1))  # query-conditioned
        start_logits = self.start(h).squeeze(-1)  # [B, num_bins]
        end_logits = self.end(h).squeeze(-1)

        denom = max(self.num_bins - 1, 1)
        start_norm = start_logits.argmax(-1).float() / denom
        end_norm = end_logits.argmax(-1).float() / denom
        conf = (
            start_logits.softmax(-1).amax(-1) * end_logits.softmax(-1).amax(-1)
        ).sqrt()
        return GroundingOutput(start_logits, end_logits, start_norm, end_norm, conf)
