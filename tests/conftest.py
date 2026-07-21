"""Shared test fixtures. Everything here is tiny and synthetic — no downloads."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from viora.utils.config import VisionConfig
from viora.utils.seed import set_seed


@pytest.fixture(autouse=True)
def _determinism():
    set_seed(0)


@pytest.fixture
def tiny_vision_config() -> VisionConfig:
    """A ViT small enough to run in milliseconds on CPU."""
    return VisionConfig(
        image_size=32,
        num_frames=8,
        in_channels=3,
        tubelet_size=2,   # -> T'=4
        patch_size=16,    # -> H'=W'=2
        dim=24,
        depth=2,
        num_heads=3,      # head_dim=8
        mlp_ratio=2.0,
        attention_mode="factorized",
        drop_path_rate=0.0,
    )


@pytest.fixture
def video_batch() -> torch.Tensor:
    """[B=2, C=3, T=8, H=32, W=32] synthetic clip."""
    return torch.randn(2, 3, 8, 32, 32)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
