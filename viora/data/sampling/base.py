"""Frame-sampling interface.

Samplers choose *which* frames to keep from a (decoded) clip. They operate on
frame counts + timestamps so they are unit-testable without video I/O, and every
selection preserves source timestamps. The API is uniform so strategies are
interchangeable via :func:`build_sampler`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameSelection:
    indices: np.ndarray            # int [K] indices into the source frames
    timestamps: np.ndarray | None  # float [K] seconds, if available


class FrameSampler(ABC):
    """Base class. Subclasses implement :meth:`select`."""

    def __init__(self, num_frames: int) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.num_frames = num_frames

    @abstractmethod
    def select(
        self,
        total: int,
        timestamps: np.ndarray | None = None,
        frames: np.ndarray | None = None,
        generator: np.random.Generator | None = None,
    ) -> FrameSelection:
        """Return the chosen indices (+timestamps) from ``total`` source frames."""

    def __call__(self, *args, **kwargs) -> FrameSelection:
        return self.select(*args, **kwargs)

    @staticmethod
    def _gather_ts(timestamps: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
        return None if timestamps is None else np.asarray(timestamps)[idx]


def build_sampler(name: str, num_frames: int, **kwargs) -> FrameSampler:
    """Factory: ``"uniform"`` | ``"random_clip"`` | ``"adaptive"``."""
    from viora.data.sampling.adaptive import AdaptiveFrameSampler
    from viora.data.sampling.random_clip import RandomClipSampler
    from viora.data.sampling.uniform import UniformFrameSampler

    table = {
        "uniform": UniformFrameSampler,
        "random_clip": RandomClipSampler,
        "adaptive": AdaptiveFrameSampler,
    }
    if name not in table:
        raise ValueError(f"unknown sampler '{name}' (have: {sorted(table)})")
    return table[name](num_frames=num_frames, **kwargs)
