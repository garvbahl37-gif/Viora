"""Sharded WebDataset pipeline: write -> read -> decode round-trip."""

from __future__ import annotations

import shutil

import pytest
import torch

wds = pytest.importorskip("webdataset")
_HAS_FFMPEG = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg for mp4 encode")

from viora.data.webdataset_pipeline import (  # noqa: E402
    build_video_text_webdataset,
    write_video_text_shards,
)


def test_shard_round_trip(tmp_path):
    samples = [
        {"video": torch.rand(3, 8, 32, 32), "meta": {"question": "what?", "answer": f"a{i}"}}
        for i in range(5)
    ]
    pattern = str(tmp_path / "train-%06d.tar")
    n = write_video_text_shards(samples, pattern, maxcount=3)
    assert n == 5
    assert (tmp_path / "train-000000.tar").exists()
    assert (tmp_path / "train-000001.tar").exists()  # 5 samples / maxcount 3 -> 2 shards

    shards = [str(tmp_path / "train-000000.tar"), str(tmp_path / "train-000001.tar")]
    ds = build_video_text_webdataset(shards, num_frames=8, resampled=False, shuffle=0)
    items = list(iter(ds))
    assert len(items) == 5
    it = items[0]
    assert it["video"].shape[0] == 3 and it["video"].shape[1] == 8  # [C=3, T=8, H, W]
    assert "answer" in it and "question" in it


def test_decode_bytes_shape(tmp_path):
    from viora.data.webdataset_pipeline import decode_video_bytes, tensor_to_mp4_bytes

    data = tensor_to_mp4_bytes(torch.rand(3, 10, 48, 48), fps=5.0)
    video, ts = decode_video_bytes(data, num_frames=6)
    assert video.shape[0] == 3 and video.shape[1] == 6
    assert ts.shape[0] == 6


def test_decode_bytes_long_clip_sparse_path(tmp_path):
    # long clip -> seek-sample path; must still return exactly num_frames spanning the clip
    from viora.data.webdataset_pipeline import decode_video_bytes, tensor_to_mp4_bytes

    frames = torch.rand(3, 64, 32, 32)
    data = tensor_to_mp4_bytes(frames, fps=8.0)
    video, ts = decode_video_bytes(data, num_frames=8)
    assert video.shape == (3, 8, 32, 32)
    assert ts.shape[0] == 8


def test_decode_bytes_short_clip_fallback(tmp_path):
    # fewer frames than requested -> full-decode fallback, capped at available frames
    from viora.data.webdataset_pipeline import decode_video_bytes, tensor_to_mp4_bytes

    data = tensor_to_mp4_bytes(torch.rand(3, 4, 32, 32), fps=4.0)
    video, ts = decode_video_bytes(data, num_frames=8)
    assert video.shape[0] == 3 and video.shape[1] == 4   # only 4 frames exist
    assert ts.shape[0] == 4


def test_decoder_aborts_on_fully_broken_stream():
    """A stream where every sample fails to decode must fail fast, not skip forever."""
    import json

    from viora.data.webdataset_pipeline import _StreamBroken, _VideoTextDecoder

    dec = _VideoTextDecoder(num_frames=8, transform=None, max_consecutive_failures=5)
    bad = (b"not-a-video", json.dumps({"captions": ["x"]}).encode())
    for _ in range(4):                       # first failures are skippable (not the abort)
        with pytest.raises(Exception) as ei:  # noqa: PT011
            dec(bad)
        assert not isinstance(ei.value, _StreamBroken)
    with pytest.raises(_StreamBroken):        # 5th consecutive failure trips the abort
        dec(bad)


def test_decoder_failure_counter_resets_on_success():
    import json

    from viora.data.webdataset_pipeline import (
        _StreamBroken,
        _VideoTextDecoder,
        tensor_to_mp4_bytes,
    )

    dec = _VideoTextDecoder(num_frames=8, transform=None, max_consecutive_failures=3)
    bad = (b"xxx", json.dumps({}).encode())
    good = (tensor_to_mp4_bytes(torch.rand(3, 8, 32, 32)),
            json.dumps({"captions": ["a"]}).encode())
    for _ in range(2):
        with pytest.raises(Exception):  # noqa: PT011, B017
            dec(bad)
    dec(good)                                  # a success resets the consecutive counter
    with pytest.raises(Exception) as ei:       # noqa: PT011
        dec(bad)
    assert not isinstance(ei.value, _StreamBroken)  # would have aborted at 3 without the reset


def test_shard_spec_splits_on_double_colon(tmp_path):
    """One CLI string must be able to MIX several shard sources (e.g. a local dataset
    plus one streamed from a remote host) via '::'. Without this, training can only
    ever draw from a single source."""
    import torch as _torch

    from viora.data.webdataset_pipeline import write_video_text_shards

    for s in range(2):
        write_video_text_shards(
            [{"video": _torch.rand(3, 8, 32, 32), "meta": {"captions": [f"s{s}c{i}"]}}
             for i in range(2)],
            str(tmp_path / f"src{s}-%06d.tar"), maxcount=100)

    spec = f"{tmp_path / 'src0-000000.tar'}::{tmp_path / 'src1-000000.tar'}"
    ds = build_video_text_webdataset(spec, num_frames=8, resampled=False, shuffle=0)
    caps = {c for it in ds for c in it["captions"]}
    assert {"s0c0", "s0c1", "s1c0", "s1c1"} <= caps   # BOTH sources present
