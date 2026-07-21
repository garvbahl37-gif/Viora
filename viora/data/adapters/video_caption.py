"""Convert real video-caption datasets into Viora WebDataset shards.

The training pipeline streams ``(mp4 bytes, json meta)`` tar shards
(:mod:`viora.data.webdataset_pipeline`). This adapter builds those shards from a
downloaded dataset without transcoding: each clip's **raw** mp4 bytes are stored
once, with *all* its reference captions in metadata, so N captions per clip cost
no extra disk (the collator samples one caption per training view).

Two input layouts are supported:

* **MSR-VTT** (:func:`parse_msrvtt`) — a ``videodatainfo.json`` carrying
  ``{"videos": [{video_id, split}], "sentences": [{video_id, caption}]}``.
* **Folder sidecar** (:func:`parse_folder_sidecar`) — a JSON mapping
  ``{video_id_or_filename: caption | [captions]}``. Works for MSVD, custom sets,
  or anything you can express as that map.

Both resolve clip ids to files via :func:`index_video_files` (recursive), so odd
directory nesting or extensions do not matter.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path

from viora.utils.logging import get_logger

logger = get_logger(__name__)

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".gif")


def index_video_files(videos_dir: str | Path) -> dict[str, Path]:
    """Map every video file under ``videos_dir`` by both its name and stem.

    ``{"video0.mp4": path, "video0": path}`` so annotation ids that carry or omit
    the extension both resolve. Searched recursively.
    """
    root = Path(videos_dir)
    idx: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            idx.setdefault(p.name, p)
            idx.setdefault(p.stem, p)
    return idx


def parse_msrvtt(annotations: str | Path, split: str | None = None) -> dict[str, list[str]]:
    """Parse an MSR-VTT ``videodatainfo.json`` into ``video_id -> [captions]``.

    If ``split`` is given (``"train"``/``"validate"``/``"test"``), only clips whose
    ``videos[].split`` matches are kept.
    """
    data = json.loads(Path(annotations).read_text())
    keep: set[str] | None = None
    if split is not None:
        keep = {v["video_id"] for v in data.get("videos", []) if v.get("split") == split}
    caps: dict[str, list[str]] = defaultdict(list)
    for s in data.get("sentences", []):
        vid = s.get("video_id")
        cap = s.get("caption")
        if vid is None or cap is None:
            continue
        if keep is None or vid in keep:
            caps[vid].append(cap)
    return dict(caps)


def parse_folder_sidecar(annotations: str | Path) -> dict[str, list[str]]:
    """Parse a ``{video_id_or_filename: caption | [captions]}`` JSON sidecar."""
    raw = json.loads(Path(annotations).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError("folder sidecar must be a JSON object mapping id -> caption(s)")
    caps: dict[str, list[str]] = {}
    for k, v in raw.items():
        items = [v] if isinstance(v, str) else list(v)
        cleaned = [str(c) for c in items if c]
        if cleaned:
            caps[str(k)] = cleaned
    return caps


def caption_shard_samples(
    captions: dict[str, list[str]],
    video_index: dict[str, Path],
    *,
    limit: int | None = None,
) -> Iterator[dict]:
    """Yield ``{"video_bytes": bytes, "meta": {video_id, captions}}`` for shard writing.

    Clips whose id has no matching video file (or an unreadable/empty file) are
    skipped with an aggregate warning — one bad clip never aborts the build.
    """
    missing = 0
    emitted = 0
    for vid, caps in captions.items():
        if limit is not None and emitted >= limit:
            break
        path = video_index.get(vid) or video_index.get(f"{vid}.mp4")
        if path is None:
            missing += 1
            continue
        try:
            data = path.read_bytes()
        except OSError:
            data = b""
        if not data:
            missing += 1
            continue
        yield {"video_bytes": data, "meta": {"video_id": vid, "captions": caps}}
        emitted += 1
    if missing:
        logger.warning("%d annotated clips had no readable video file (skipped)", missing)


def build_caption_shards(
    videos_dir: str | Path,
    annotations: str | Path,
    out_pattern: str,
    *,
    fmt: str = "msrvtt",
    split: str | None = None,
    limit: int | None = None,
    maxcount: int = 500,
) -> int:
    """End-to-end: parse annotations, resolve videos, write shards. Returns clip count."""
    from viora.data.webdataset_pipeline import write_video_text_shards

    videos_dir = Path(videos_dir)
    if not videos_dir.is_dir():
        raise NotADirectoryError(f"--videos '{videos_dir}' is not a directory")
    if not Path(annotations).is_file():
        raise FileNotFoundError(f"--annotations '{annotations}' not found")

    video_index = index_video_files(videos_dir)
    if not video_index:
        raise FileNotFoundError(
            f"no video files ({', '.join(VIDEO_EXTS)}) found under {videos_dir}"
        )

    if fmt == "msrvtt":
        captions = parse_msrvtt(annotations, split)
    elif fmt == "folder":
        captions = parse_folder_sidecar(annotations)
    else:
        raise ValueError(f"unknown fmt '{fmt}' (expected 'msrvtt' or 'folder')")
    if not captions:
        raise ValueError("no captions parsed from annotations (check --format / --split)")

    n_files = len(set(video_index.values()))
    logger.info("%d annotated clips | %d video files -> writing shards", len(captions), n_files)
    samples = caption_shard_samples(captions, video_index, limit=limit)
    return write_video_text_shards(samples, out_pattern, maxcount=maxcount)
