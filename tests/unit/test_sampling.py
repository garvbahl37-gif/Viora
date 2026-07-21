"""Frame samplers: counts, determinism, ranges, adaptive motion selection."""

from __future__ import annotations

import numpy as np

from viora.data.sampling import (
    AdaptiveFrameSampler,
    RandomClipSampler,
    UniformFrameSampler,
    build_sampler,
    motion_scores,
)


def test_uniform_count_and_endpoints():
    s = UniformFrameSampler(num_frames=8)
    sel = s.select(total=100)
    assert len(sel.indices) == 8
    assert sel.indices[0] == 0 and sel.indices[-1] == 99
    assert list(sel.indices) == sorted(sel.indices)


def test_uniform_deterministic():
    s = UniformFrameSampler(num_frames=5)
    a = s.select(total=57).indices
    b = s.select(total=57).indices
    assert np.array_equal(a, b)


def test_returns_all_when_short():
    s = UniformFrameSampler(num_frames=16)
    sel = s.select(total=10)
    assert np.array_equal(sel.indices, np.arange(10))


def test_timestamps_preserved():
    ts = np.linspace(0, 10, 100)
    sel = UniformFrameSampler(8).select(total=100, timestamps=ts)
    assert sel.timestamps is not None
    assert np.allclose(sel.timestamps, ts[sel.indices])


def test_random_clip_reproducible_and_in_range():
    s = RandomClipSampler(num_frames=8, stride=2)
    g1 = np.random.default_rng(123)
    g2 = np.random.default_rng(123)
    a = s.select(total=100, generator=g1).indices
    b = s.select(total=100, generator=g2).indices
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < 100
    # contiguous strided window
    assert np.all(np.diff(a) == 2)


def test_adaptive_covers_bins_and_prefers_motion():
    total = 40
    frames = np.zeros((total, 8, 8, 3), dtype=np.uint8)
    # inject a high-motion frame at index 21 (second half)
    frames[21] = 255
    sel = AdaptiveFrameSampler(num_frames=8).select(total=total, frames=frames)
    assert len(sel.indices) <= 8
    assert list(sel.indices) == sorted(sel.indices)
    # the motion spike should be selected (it maximizes its bin)
    assert 21 in set(sel.indices.tolist())


def test_adaptive_without_frames_is_coverage():
    sel = AdaptiveFrameSampler(num_frames=8).select(total=40)
    assert len(sel.indices) == 8
    assert sel.indices[0] >= 0 and sel.indices[-1] < 40


def test_motion_scores_shape():
    frames = np.random.randint(0, 255, size=(6, 4, 4, 3), dtype=np.uint8)
    m = motion_scores(frames)
    assert m.shape == (6,)
    assert m[0] == 0.0


def test_build_sampler_factory():
    assert isinstance(build_sampler("uniform", 8), UniformFrameSampler)
    assert isinstance(build_sampler("adaptive", 8), AdaptiveFrameSampler)
