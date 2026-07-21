"""Uniform frame sampling — deterministic, evenly spaced across the clip."""

from __future__ import annotations

import numpy as np

from viora.data.sampling.base import FrameSampler, FrameSelection


class UniformFrameSampler(FrameSampler):
    """Pick ``num_frames`` indices evenly spaced over ``[0, total)``.

    When ``total <= num_frames`` all frames are returned (no padding here; the
    collator pads and builds the temporal mask).
    """

    def select(self, total, timestamps=None, frames=None, generator=None) -> FrameSelection:
        if total <= 0:
            raise ValueError("total must be >= 1")
        if total <= self.num_frames:
            idx = np.arange(total)
        else:
            idx = np.linspace(0, total - 1, self.num_frames).round().astype(int)
        return FrameSelection(indices=idx, timestamps=self._gather_ts(timestamps, idx))
