"""Video-text contrastive loss (symmetric InfoNCE, CLIP-style).

A learnable temperature scales cosine similarities between L2-normalised video and
text embeddings; the loss pulls matched pairs together and pushes mismatched pairs
apart in both directions. Optional in-batch hard-negative weighting is left for a
later pass behind the same interface.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class VideoTextContrastiveLoss(nn.Module):
    def __init__(self, init_temperature: float = 0.07, learnable: bool = True, max_scale: float = 100.0) -> None:
        super().__init__()
        logit_scale = torch.tensor(math.log(1.0 / init_temperature))
        self.logit_scale = nn.Parameter(logit_scale) if learnable else logit_scale
        self.max_scale = max_scale

    def forward(self, video_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """``video_emb``/``text_emb``: ``[B, P]`` (assumed L2-normalised)."""
        scale = self.logit_scale.exp().clamp(max=self.max_scale)
        logits = scale * video_emb @ text_emb.t()  # [B, B]
        target = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
