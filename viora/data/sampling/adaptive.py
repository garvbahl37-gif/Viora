"""Adaptive frame sampling — motion / temporal-diversity aware.

A robust deterministic baseline: split the clip into ``num_frames`` contiguous
bins (guaranteeing temporal coverage / a bounded token budget) and, when raw
frames are available, keep the **highest-motion** frame in each bin (motion =
mean absolute inter-frame difference, a cheap scene-change proxy). Without frames
it degrades to the bin centre (i.e. uniform). This leaves room to later swap in
learned or feature-based saliency behind the same interface.
"""

from __future__ import annotations

import numpy as np

from viora.data.sampling.base import FrameSampler, FrameSelection


def motion_scores(frames: np.ndarray) -> np.ndarray:
    """Per-frame motion = mean |frame_i - frame_{i-1}|; ``frames`` uint8 ``[T,H,W,C]``."""
    if frames.shape[0] < 2:
        return np.zeros((frames.shape[0],), dtype=np.float64)
    f = frames.astype(np.float32)
    diff = np.abs(f[1:] - f[:-1]).reshape(f.shape[0] - 1, -1).mean(axis=1)
    return np.concatenate([[0.0], diff])  # frame 0 has no predecessor


class AdaptiveFrameSampler(FrameSampler):
    def __init__(self, num_frames: int, min_motion_bins: bool = True) -> None:
        super().__init__(num_frames)
        self.min_motion_bins = min_motion_bins

    def select(self, total, timestamps=None, frames=None, generator=None) -> FrameSelection:
        if total <= 0:
            raise ValueError("total must be >= 1")
        if total <= self.num_frames:
            idx = np.arange(total)
            return FrameSelection(indices=idx, timestamps=self._gather_ts(timestamps, idx))

        # contiguous bins guarantee coverage + a bounded number of tokens
        edges = np.linspace(0, total, self.num_frames + 1).astype(int)
        scores = motion_scores(frames) if frames is not None else None

        idx = []
        for b in range(self.num_frames):
            lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
            if scores is not None:
                local = scores[lo:hi]
                idx.append(lo + int(np.argmax(local)))
            else:
                idx.append((lo + hi - 1) // 2)  # bin centre -> uniform fallback
        idx = np.asarray(sorted(set(idx)))
        return FrameSelection(indices=idx, timestamps=self._gather_ts(timestamps, idx))
