"""Video decoder: real decode of an ffmpeg-synthesized clip (skips if unavailable)."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess

import numpy as np
import pytest

from viora.data.decoding.video_decoder import (
    VideoDecodeError,
    VideoDecoder,
    validate_video_path,
)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_AV = importlib.util.find_spec("av") is not None
requires_video = pytest.mark.skipif(
    not (_HAS_FFMPEG and _HAS_AV), reason="needs ffmpeg + PyAV"
)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("vid") / "testsrc.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=10",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return str(path)


@requires_video
def test_probe(sample_video):
    meta = VideoDecoder().probe(sample_video)
    assert meta.width == 64 and meta.height == 48
    assert meta.duration is not None and 1.5 < meta.duration < 2.5


@requires_video
def test_decode_full_clip(sample_video):
    clip = VideoDecoder().decode_clip(sample_video)
    assert clip.frames.dtype.is_floating_point is False  # uint8
    assert clip.frames.shape[1:] == (48, 64, 3)  # [T, H, W, C]
    assert clip.num_frames >= 15  # ~20 frames for 2s @ 10fps
    # timestamps strictly increasing and start near 0
    assert np.all(np.diff(clip.timestamps) > 0)
    assert clip.timestamps[0] < 0.2


@requires_video
def test_decode_window(sample_video):
    clip = VideoDecoder().decode_clip(sample_video, start_time=0.5, end_time=1.5)
    assert clip.timestamps.min() >= 0.5 - 1e-6
    assert clip.timestamps.max() <= 1.5 + 1e-6


@requires_video
def test_max_frames_cap(sample_video):
    clip = VideoDecoder().decode_clip(sample_video, max_frames=5)
    assert clip.num_frames == 5


@requires_video
def test_as_model_input(sample_video):
    clip = VideoDecoder().decode_clip(sample_video, max_frames=4)
    x = clip.as_model_input()
    assert x.shape == (3, 4, 48, 64)  # [C, T, H, W]
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_validation_errors(tmp_path):
    with pytest.raises(VideoDecodeError):
        validate_video_path(tmp_path / "missing.mp4")
    bad = tmp_path / "file.txt"
    bad.write_text("not a video")
    with pytest.raises(VideoDecodeError):
        validate_video_path(bad)


def test_available_backends_reports_pyav():
    # PyAV is installed in this environment
    assert "pyav" in VideoDecoder.available_backends()
