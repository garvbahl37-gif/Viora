"""Video-text retrieval head: project both modalities to a shared, L2-normalised
space for cosine-similarity retrieval and contrastive learning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RetrievalHead(nn.Module):
    def __init__(self, video_dim: int, text_dim: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.video_proj = nn.Linear(video_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

    def forward(
        self, video_feat: torch.Tensor, text_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, video_dim]``, ``[B, text_dim]`` -> two ``[B, proj_dim]`` unit vectors."""
        v = F.normalize(self.video_proj(video_feat), dim=-1)
        t = F.normalize(self.text_proj(text_feat), dim=-1)
        return v, t

    def encode_video(self, video_feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.video_proj(video_feat), dim=-1)

    def encode_text(self, text_feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_proj(text_feat), dim=-1)
