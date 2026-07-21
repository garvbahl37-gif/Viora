"""Spatial transforms for video tensors ``[C, T, H, W]`` (frame-consistent)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from viora.data.preprocessing.normalize import IMAGENET_MEAN, IMAGENET_STD, VideoNormalize


def resize_video(video: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear-resize the spatial dims of ``[C, T, H, W]`` to ``size x size``."""
    c, t, h, w = video.shape
    x = video.permute(1, 0, 2, 3)  # [T, C, H, W]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.permute(1, 0, 2, 3).contiguous()


def center_crop_video(video: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop ``[C, T, H, W]`` spatially to ``size x size`` (resizes up if smaller)."""
    _, _, h, w = video.shape
    if h < size or w < size:
        video = resize_video(video, size)
        return video
    top = (h - size) // 2
    left = (w - size) // 2
    return video[:, :, top : top + size, left : left + size]


class VideoTransform:
    """Compose resize (short side) -> center crop -> normalize into one call."""

    def __init__(self, size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> None:
        self.size = size
        self.normalize = VideoNormalize(mean, std)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        video = resize_video(video, self.size)
        video = center_crop_video(video, self.size)
        return self.normalize(video)
