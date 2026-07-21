"""Channel normalization for video tensors ``[C, T, H, W]``."""

from __future__ import annotations

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)


class VideoNormalize(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1, 1))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``[C, T, H, W]`` (or ``[B, C, T, H, W]``) in [0,1] -> standardized."""
        mean, std = self.mean, self.std
        if video.dim() == 5:
            mean, std = mean.unsqueeze(0), std.unsqueeze(0)
        return (video - mean) / std
