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
