"""Video decoding with a swappable backend.

Only PyAV is implemented here (it is the one backend installed and is pure-Python
to install); torchvision / decord are recognised but raise a clear error until
wired, so nothing fails silently. The decoder is **lazy**: it seeks to the
requested window and never decodes an entire long video to grab a short clip.

Every returned frame keeps its **source timestamp** (seconds) — grounding and
memory depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from viora.utils.logging import get_logger

logger = get_logger(__name__)

VALID_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}
DEFAULT_MAX_BYTES = 4 * 1024**3  # 4 GB safety cap


class VideoDecodeError(RuntimeError):
    """Raised on corrupt files, unreadable streams, or empty decode windows."""


@dataclass
class VideoMetadata:
    path: str
    duration: float | None  # seconds
    fps: float | None
    width: int | None
    height: int | None
    num_frames: int | None


@dataclass
class DecodedClip:
    """A decoded clip. ``frames`` is uint8 ``[T, H, W, C]``; ``timestamps`` seconds."""

    frames: torch.Tensor            # uint8 [T, H, W, C]
    timestamps: np.ndarray          # float64 [T]
    metadata: VideoMetadata
    extra: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    def as_model_input(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return ``[C, T, H, W]`` float in ``[0, 1]`` (normalization is applied later)."""
        x = self.frames.to(dtype) / 255.0
        return x.permute(3, 0, 1, 2).contiguous()


def validate_video_path(path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Path:
    """Validate a video path: existence, extension allow-list, and size cap."""
    p = Path(path)
    if not p.is_file():
        raise VideoDecodeError(f"not a file: {p}")
    if p.suffix.lower() not in VALID_EXTENSIONS:
        raise VideoDecodeError(
            f"unsupported extension '{p.suffix}' (allowed: {sorted(VALID_EXTENSIONS)})"
        )
    size = p.stat().st_size
    if size == 0:
        raise VideoDecodeError(f"empty file: {p}")
    if size > max_bytes:
        raise VideoDecodeError(f"file too large: {size} bytes > cap {max_bytes}")
    return p


def _select_indices(n: int, timestamps: np.ndarray, target_fps: float | None, max_frames: int | None) -> np.ndarray:
    """Choose a subset of frame indices honoring ``target_fps`` then ``max_frames``."""
    idx = np.arange(n)
    if target_fps and n > 1 and timestamps[-1] > timestamps[0]:
        span = timestamps[-1] - timestamps[0]
        want = max(1, int(round(span * target_fps)) + 1)
        grid = np.linspace(timestamps[0], timestamps[-1], want)
        idx = np.unique(np.searchsorted(timestamps, grid).clip(0, n - 1))
    if max_frames and len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]
    return idx


class VideoDecoder:
    """Backend-abstracted clip decoder.

    Args:
        backend: ``"auto"`` | ``"pyav"`` | ``"torchvision"`` | ``"decord"``.
        max_bytes: file-size safety cap.
        max_decoded_frames: hard cap on how many frames are held in memory at once.
            Decoding materialised EVERY frame at full resolution before subsampling,
            so a 1-minute 1080p clip allocated ~11 GB and the OS SIGKILLed the process
            ("zsh: killed") -- for a model that only needs 8 frames. Frames are now
            decimated on the fly (keep every Nth, doubling N whenever the cap is hit)
            so memory is bounded while temporal coverage stays roughly uniform.
    """

    def __init__(
        self, backend: str = "auto", *, max_bytes: int = DEFAULT_MAX_BYTES,
        max_decoded_frames: int = 128,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_decoded_frames = max(1, max_decoded_frames)
        self.backend = self._resolve_backend(backend)

    @staticmethod
    def available_backends() -> list[str]:
        import importlib.util

        out = []
        for name, mod in (("pyav", "av"), ("torchvision", "torchvision"), ("decord", "decord")):
            if importlib.util.find_spec(mod) is not None:
                out.append(name)
        return out

    def _resolve_backend(self, backend: str) -> str:
        avail = self.available_backends()
        if backend == "auto":
            if not avail:
                raise VideoDecodeError(
                    "no video backend installed; install one, e.g. `uv pip install av`"
                )
            # prefer pyav for correctness of timestamps
            return "pyav" if "pyav" in avail else avail[0]
        if backend not in avail:
            raise VideoDecodeError(f"backend '{backend}' not installed (have: {avail})")
        return backend

    def probe(self, path: str | Path) -> VideoMetadata:
        p = validate_video_path(path, max_bytes=self.max_bytes)
        if self.backend == "pyav":
            return self._probe_pyav(p)
        raise VideoDecodeError(f"backend '{self.backend}' not implemented for probe")

    def decode_clip(
        self,
        path: str | Path,
        *,
        start_time: float = 0.0,
        end_time: float | None = None,
        target_fps: float | None = None,
        max_frames: int | None = None,
    ) -> DecodedClip:
        """Decode a ``[start_time, end_time]`` window, subsampled to ``target_fps`` / ``max_frames``."""
        p = validate_video_path(path, max_bytes=self.max_bytes)
        if start_time < 0:
            raise ValueError("start_time must be >= 0")
        if end_time is not None and end_time <= start_time:
            raise ValueError("end_time must be > start_time")
        if self.backend == "pyav":
            return self._decode_pyav(p, start_time, end_time, target_fps, max_frames)
        raise VideoDecodeError(f"backend '{self.backend}' not implemented for decode")

    # ----------------------------------------------------------------- PyAV
    def _probe_pyav(self, p: Path) -> VideoMetadata:
        import av

        try:
            with av.open(str(p)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else None
                duration = (
                    float(stream.duration * stream.time_base)
                    if stream.duration is not None
                    else (float(container.duration / av.time_base) if container.duration else None)
                )
                return VideoMetadata(
                    path=str(p), duration=duration, fps=fps,
                    width=stream.codec_context.width, height=stream.codec_context.height,
                    num_frames=stream.frames or None,
                )
        except Exception as e:  # noqa: BLE001 - normalize backend errors
            raise VideoDecodeError(f"probe failed for {p}: {e}") from e

    def _decode_pyav(
        self, p: Path, start_time: float, end_time: float | None,
        target_fps: float | None, max_frames: int | None,
    ) -> DecodedClip:
        import av

        meta = self._probe_pyav(p)
        frames: list[np.ndarray] = []
        times: list[float] = []
        try:
            with av.open(str(p)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                if start_time > 0:
                    # seek to the keyframe at/just before start (offset in time_base units)
                    offset = int(start_time / stream.time_base)
                    container.seek(offset, backward=True, stream=stream)
                cap = self.max_decoded_frames
                stride = 1        # keep every `stride`-th in-window frame
                seen = 0          # in-window frames encountered so far
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    t = float(frame.pts * stream.time_base)
                    if t < start_time - 1e-6:
                        continue  # frames before the window (post-keyframe-seek)
                    if end_time is not None and t > end_time + 1e-6:
                        break
                    # Only materialise every `stride`-th frame: to_ndarray() is what
                    # actually allocates (H*W*3 bytes), so skipping it here is what
                    # bounds memory on long/high-resolution clips.
                    if seen % stride == 0:
                        frames.append(frame.to_ndarray(format="rgb24"))
                        times.append(t)
                        if len(frames) > cap:
                            # too many: halve what we hold and sample twice as sparsely
                            # from here on. Keeps ~uniform coverage without knowing the
                            # clip's total frame count up front.
                            del frames[1::2]
                            del times[1::2]
                            stride *= 2
                    seen += 1
        except Exception as e:  # noqa: BLE001
            raise VideoDecodeError(f"decode failed for {p}: {e}") from e

        if not frames:
            raise VideoDecodeError(
                f"no frames decoded in window [{start_time}, {end_time}] of {p}"
            )

        ts = np.asarray(times, dtype=np.float64)
        idx = _select_indices(len(frames), ts, target_fps, max_frames)
        arr = np.stack([frames[i] for i in idx], axis=0)  # [T, H, W, C] uint8
        return DecodedClip(
            frames=torch.from_numpy(arr),
            timestamps=ts[idx],
            metadata=meta,
            extra={"decoded": len(frames), "kept": int(len(idx))},
        )
