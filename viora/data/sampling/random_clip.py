"""Random contiguous-clip sampling — for temporal augmentation during training."""

from __future__ import annotations

import numpy as np

from viora.data.sampling.base import FrameSampler, FrameSelection


class RandomClipSampler(FrameSampler):
    """Sample a random contiguous window of ``num_frames`` at a given ``stride``.

    Reproducible when a NumPy ``generator`` is supplied. If the strided window
    cannot fit, the stride shrinks; if the clip is shorter than ``num_frames`` all
    frames are returned.
    """

    def __init__(self, num_frames: int, stride: int = 1) -> None:
        super().__init__(num_frames)
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = stride

    def select(self, total, timestamps=None, frames=None, generator=None) -> FrameSelection:
        if total <= 0:
            raise ValueError("total must be >= 1")
        if total <= self.num_frames:
            idx = np.arange(total)
            return FrameSelection(indices=idx, timestamps=self._gather_ts(timestamps, idx))

        gen = generator or np.random.default_rng()
        stride = self.stride
        # largest stride whose window fits
        while stride > 1 and (self.num_frames - 1) * stride >= total:
            stride -= 1
        span = (self.num_frames - 1) * stride
        start = int(gen.integers(0, total - span)) if total - span > 0 else 0
        idx = start + np.arange(self.num_frames) * stride
        idx = idx.clip(0, total - 1)
        return FrameSelection(indices=idx, timestamps=self._gather_ts(timestamps, idx))
