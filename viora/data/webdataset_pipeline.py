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


def decode_video_bytes(
    data: bytes, num_frames: int, transform: Callable | None = None, fps_hint: float = 4.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode mp4 bytes -> ``([C,T,H,W] float, timestamps[T])`` uniformly sampled."""
    import av

    container = av.open(io.BytesIO(data))
    stream = container.streams.video[0]
    tb = stream.time_base
    frames, times = [], []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        times.append(float(frame.pts * tb) if frame.pts is not None else len(times) / fps_hint)
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

    def _decode(sample):
        data, meta_bytes = sample
        meta = json.loads(meta_bytes)
        video, ts = decode_video_bytes(data, num_frames, transform)
        return {"video": video, "timestamps": ts, **meta}

    return ds.map(_decode, handler=wds.warn_and_continue)
