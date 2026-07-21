"""Sharded video-text data pipeline (WebDataset).

For cloud/multi-GPU training you don't want per-file random reads over millions of
clips — you want a handful of large **tar shards** streamed sequentially, split
across nodes/workers. This module writes and reads that format:

* :func:`write_video_text_shards` — pack ``(video, metadata)`` into ``.tar`` shards.
* :func:`build_video_text_webdataset` — a streaming ``IterableDataset`` that
  decodes each clip and yields model-ready dicts, node/worker-split for DDP/FSDP.

Videos are stored as encoded mp4 bytes (compact) and decoded on the fly.
"""

from __future__ import annotations

import io
import json
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch

from viora.utils.logging import get_logger

logger = get_logger(__name__)


def tensor_to_mp4_bytes(video: torch.Tensor, fps: float = 4.0) -> bytes:
    """Encode ``[C, T, H, W]`` float [0,1] to mp4 bytes (x264, mpeg4 fallback)."""
    import av

    c, t, h, w = video.shape
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        container = av.open(tmp.name, mode="w")
        try:
            stream = container.add_stream("libx264", rate=int(round(fps)))
            stream.options = {"crf": "18", "preset": "veryfast"}
        except Exception:  # noqa: BLE001
            stream = container.add_stream("mpeg4", rate=int(round(fps)))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(t):
            arr = (video[:, i].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        return Path(tmp.name).read_bytes()


def write_video_text_shards(
    samples: Iterable[dict], pattern: str, *, maxcount: int = 1000, fps: float = 4.0
) -> int:
    """Write ``{video: [C,T,H,W] | video_bytes, meta: dict}`` items to tar shards.

    ``pattern`` is a printf-style path like ``data/train-%06d.tar``. Returns count.
    """
    import webdataset as wds

    Path(pattern).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for i, s in enumerate(samples):
            data = s.get("video_bytes") or tensor_to_mp4_bytes(s["video"], s.get("fps", fps))
            sink.write({"__key__": f"{i:09d}", "mp4": data, "json": json.dumps(s["meta"]).encode()})
            n += 1
    logger.info("wrote %d samples to shards %s", n, pattern)
    return n


def _decode_all(container, stream, tb, fps_hint: float):
    """Decode every frame of the stream (correct but O(total_frames))."""
    frames, times = [], []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        times.append(float(frame.pts * tb) if frame.pts is not None else len(times) / fps_hint)
    return frames, times


def _seek_sample(container, stream, num_frames: int, tb, fps_hint: float):
    """Seek to ~``num_frames`` positions across the clip, decoding ~num_frames frames.

    Returns ``([], [])`` if seeking is unsupported/incomplete so the caller can fall
    back to a full decode. Bounds work to a few frames per sample instead of the whole
    clip — the win for long full-resolution video where decode starves the GPU.
    """
    start = stream.start_time or 0
    duration = stream.duration
    frames, times = [], []
    for i in range(num_frames):
        target = int(start + duration * i / num_frames)
        try:
            container.seek(target, stream=stream, backward=True, any_frame=False)
            picked = None
            for frame in container.decode(stream):
                picked = frame
                if frame.pts is not None and frame.pts >= target:
                    break
        except Exception:  # noqa: BLE001 - any seek/decode issue -> fall back to full decode
            return [], []
        if picked is None:
            return [], []
        frames.append(picked.to_ndarray(format="rgb24"))
        times.append(float(picked.pts * tb) if picked.pts is not None else i / max(fps_hint, 1e-6))
    return frames, times


def decode_video_bytes(
    data: bytes, num_frames: int, transform: Callable | None = None, fps_hint: float = 4.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode mp4 bytes -> ``([C,T,H,W] float, timestamps[T])`` uniformly sampled.

    Long clips are sparse-sampled by seeking (~num_frames decodes); short/awkward
    clips (or any seek failure) fall back to decoding all frames, then uniform-subsample.
    """
    import av

    container = av.open(io.BytesIO(data))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # multithreaded decode
        tb = stream.time_base
        total = int(stream.frames or 0)
        frames, times = [], []
        if total > num_frames * 3 and stream.duration:  # long clip -> seek, don't decode all
            frames, times = _seek_sample(container, stream, num_frames, tb, fps_hint)
        if len(frames) < num_frames:                     # short clip or seek failed -> full decode
            container.seek(0)
            frames, times = _decode_all(container, stream, tb, fps_hint)
    finally:
        container.close()
    if not frames:
        raise ValueError("no frames decoded from shard sample")

    total = len(frames)
    idx = np.linspace(0, total - 1, min(num_frames, total)).round().astype(int)
    arr = np.stack([frames[i] for i in idx])  # [T,H,W,C]
    video = torch.from_numpy(arr).float().div_(255.0).permute(3, 0, 1, 2)  # [C,T,H,W]
    if transform is not None:
        video = transform(video)
    ts = torch.tensor([times[i] for i in idx], dtype=torch.float32)
    return video, ts


class _StreamBroken(RuntimeError):
    """Raised when too many *consecutive* samples fail to decode — i.e. the whole
    shard stream is broken (corrupt videos / wrong format), not just an odd bad clip.

    With ``resampled=True`` (infinite) and warn-and-continue, a universally-broken
    stream would otherwise skip every sample forever and hang training at 0 steps
    with no error. This surfaces it instead.
    """


class _VideoTextDecoder:
    """Picklable decode step for WebDataset.

    Must be a module-level callable (not a local closure) so ``DataLoader`` with
    ``num_workers > 0`` can pickle it to send to worker processes. Tracks consecutive
    decode failures (per worker) and aborts if the stream looks entirely broken.
    """

    def __init__(
        self, num_frames: int, transform: Callable | None, *, max_consecutive_failures: int = 100
    ) -> None:
        self.num_frames = num_frames
        self.transform = transform
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive = 0

    def __call__(self, sample: tuple) -> dict:
        try:
            data, meta_bytes = sample
            meta = json.loads(meta_bytes)
            video, ts = decode_video_bytes(data, self.num_frames, self.transform)
        except Exception as e:  # noqa: BLE001 - classify: odd bad clip vs broken stream
            self._consecutive += 1
            if self._consecutive >= self.max_consecutive_failures:
                raise _StreamBroken(
                    f"{self._consecutive} consecutive shard samples failed to decode — the stream "
                    "looks broken (corrupt videos or wrong format), not just an occasional bad clip."
                ) from e
            raise  # occasional bad clip: let the handler skip it and continue
        self._consecutive = 0
        return {"video": video, "timestamps": ts, **meta}


def _decode_stream_handler(exn: Exception) -> bool:
    """Skip an occasional undecodable clip, but let a broken-stream abort propagate."""
    if isinstance(exn, _StreamBroken):
        raise exn
    logger.warning("skipping undecodable shard sample: %s", exn)
    return True


def build_video_text_webdataset(
    shards: str | list[str],
    *,
    num_frames: int,
    transform: Callable | None = None,
    shuffle: int = 1000,
    resampled: bool = True,
):
    """Streaming ``IterableDataset`` yielding model-ready dicts, split across nodes/workers.

    ``shards`` is a brace pattern (``"data/train-{000000..000099}.tar"``) or a list.
    Set ``resampled=True`` for infinite sampling with replacement (standard for
    large-scale multi-epoch training).
    """
    import webdataset as wds

    ds = wds.WebDataset(
        shards,
        shardshuffle=True,
        resampled=resampled,
        nodesplitter=wds.split_by_node,
        handler=wds.warn_and_continue,  # skip corrupt samples, don't crash the run
    )
    if shuffle:
        ds = ds.shuffle(shuffle)
    ds = ds.to_tuple("mp4", "json")
    return ds.map(_VideoTextDecoder(num_frames, transform), handler=_decode_stream_handler)
