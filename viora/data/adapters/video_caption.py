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


def _strip_video_ext(vid: str) -> str:
    """Strip a trailing known video extension from a parsed id, if present.

    Different annotation sources key the SAME video differently: MSR-VTT captions
    use a bare id (``"video0"``), while the official MSRVTT-QA json
    (Salesforce LAVIS) uses a ``"video"`` key holding the filename
    (``"video0.mp4"``). Without normalizing both to the same bare form,
    :func:`merged_shard_samples` would treat them as two different videos and
    write each one twice, defeating the "store each clip once" design.
    """
    low = vid.lower()
    for ext in VIDEO_EXTS:
        if low.endswith(ext):
            return vid[: -len(ext)]
    return vid


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
    if not isinstance(data, Mapping):
        raise ValueError(
            f"MSR-VTT --annotations must be a JSON object with 'videos'/'sentences' "
            f"(videodatainfo.json); got a top-level JSON {type(data).__name__}. "
            "MSRVTT-QA *_qa.json and bare caption lists are not captioning input here — "
            "point --annotations at videodatainfo.json, or convert to a {id: caption(s)} "
            "sidecar and pass --format folder."
        )
    keep: set[str] | None = None
    if split is not None:
        videos = data.get("videos", [])
        keep = {v["video_id"] for v in videos if v.get("split") == split}
        if not keep and videos:
            present = sorted({v.get("split") for v in videos})
            raise ValueError(
                f"--split {split!r} matched 0 clips. Splits present in this file: {present}. "
                "MSR-VTT test files only contain 'test'; use train_val_videodatainfo.json for train."
            )
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


_ID_KEYS = ("video_id", "clip_id", "image_id", "vid", "id", "video")
_CAP_KEYS = ("caption", "sentence", "text", "captions", "sentences")


def parse_caption_records(annotations: str | Path) -> dict[str, list[str]]:
    """Parse a top-level LIST of caption records into ``video_id -> [captions]``.

    Handles the common 'expanded' caption format (one row per caption), e.g.
    friedrichor/MSR-VTT's ``msrvtt_train_7k.json``:
    ``[{"video_id": "video0", "caption": "..."}, ...]``. The id and caption keys
    are auto-detected across a few common spellings (``_ID_KEYS`` / ``_CAP_KEYS``).
    """
    data = json.loads(Path(annotations).read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"--format records expects a top-level JSON list of caption records; got "
            f"{type(data).__name__}. Use --format msrvtt (videodatainfo.json) or "
            "--format folder ({id: caption} object)."
        )
    caps: dict[str, list[str]] = defaultdict(list)
    for rec in data:
        if not isinstance(rec, Mapping):
            continue
        vid = next((str(rec[k]) for k in _ID_KEYS if rec.get(k) is not None), None)
        if vid is None:
            continue
        vid = _strip_video_ext(vid)
        for ck in _CAP_KEYS:
            val = rec.get(ck)
            if not val:
                continue
            if isinstance(val, str):
                caps[vid].append(val)
            else:
                caps[vid].extend(str(c) for c in val if c)
            break
    return dict(caps)


_QA_KEY_PAIRS: tuple[tuple[str, str], ...] = (("question", "answer"), ("q", "a"), ("Q", "A"))


def parse_qa_records(annotations: str | Path) -> dict[str, list[tuple[str, str]]]:
    """Parse a top-level LIST of QA records into ``video_id -> [(question, answer), ...]``.

    Handles the standard MSRVTT-QA shape (one row per QA pair) as published by
    Salesforce LAVIS (``qa_train.json`` / ``qa_val.json`` / ``qa_test.json``), e.g.
    ``[{"video": "video0.mp4", "question": "...", "answer": "..."}, ...]``, as well
    as mirrors that instead use a bare ``video_id``. The id key is auto-detected via
    ``_ID_KEYS`` and normalized (a trailing ``.mp4``-style extension is stripped, so
    ``"video0.mp4"`` and ``"video0"`` key to the SAME video); the question/answer
    key pair is auto-detected via ``_QA_KEY_PAIRS`` (some mirrors use ``q``/``a``).
    Records missing an id, a question, or an answer are skipped.
    """
    data = json.loads(Path(annotations).read_text())
    if not isinstance(data, list):
        raise ValueError(
            f"MSRVTT-QA --qa-annotations expects a top-level JSON list of "
            f"{{video_id, question, answer}} records; got {type(data).__name__}."
        )
    qa: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rec in data:
        if not isinstance(rec, Mapping):
            continue
        vid = next((str(rec[k]) for k in _ID_KEYS if rec.get(k) is not None), None)
        if vid is None:
            continue
        vid = _strip_video_ext(vid)
        for qk, ak in _QA_KEY_PAIRS:
            q, a = rec.get(qk), rec.get(ak)
            if q and a:
                qa[vid].append((str(q), str(a)))
                break
    return dict(qa)


def load_captions_auto(annotations: str | Path, split: str | None = None) -> dict[str, list[str]]:
    """Detect the annotation shape and dispatch to the right parser.

    * top-level list -> :func:`parse_caption_records`
    * object with ``videos``/``sentences`` -> :func:`parse_msrvtt`
    * any other object -> :func:`parse_folder_sidecar`
    """
    data = json.loads(Path(annotations).read_text())
    if isinstance(data, list):
        return parse_caption_records(annotations)
    if isinstance(data, Mapping):
        if "sentences" in data or "videos" in data:
            return parse_msrvtt(annotations, split)
        return parse_folder_sidecar(annotations)
    raise ValueError(f"unsupported annotations JSON (top-level {type(data).__name__})")


def _resolve_video(vid: str, video_index: dict[str, Path]) -> Path | None:
    """Map a caption's clip id to a video file, tolerating id spelling differences.

    Tries the id as-is and with ``.mp4``; if the id is bare digits (``"0"``), also
    tries the ``videoN`` convention MSR-VTT files use.
    """
    for key in (vid, f"{vid}.mp4"):
        if key in video_index:
            return video_index[key]
    if vid.isdigit():
        for key in (f"video{vid}", f"video{vid}.mp4"):
            if key in video_index:
                return video_index[key]
    return None


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
        path = _resolve_video(vid, video_index)
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


def merged_shard_samples(
    captions: dict[str, list[str]],
    qa: dict[str, list[tuple[str, str]]],
    video_index: dict[str, Path],
    *,
    limit: int | None = None,
) -> Iterator[dict]:
    """Yield one shard sample per video carrying whichever of captions/QA it has.

    A video with neither captions nor QA pairs is skipped (nothing to supervise).
    Storing each video once — regardless of how many captions/QA pairs it has —
    is what keeps a combined caption+QA shard set the same size as a captions-only
    one; the collator samples one caption or one QA pair per training view.
    """
    missing = 0
    emitted = 0
    for vid in {**captions, **qa}:
        if limit is not None and emitted >= limit:
            break
        caps, pairs = captions.get(vid), qa.get(vid)
        if not caps and not pairs:
            continue
        path = _resolve_video(vid, video_index)
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
        meta: dict = {"video_id": vid}
        if caps:
            meta["captions"] = caps
        if pairs:
            meta["qa"] = [list(p) for p in pairs]
        yield {"video_bytes": data, "meta": meta}
        emitted += 1
    if missing:
        logger.warning("%d annotated clips had no readable video file (skipped)", missing)


def build_caption_qa_shards(
    videos_dir: str | Path,
    out_pattern: str,
    *,
    captions_path: str | Path | None = None,
    captions_fmt: str = "auto",
    qa_path: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    maxcount: int = 500,
) -> int:
    """Build ONE shard set combining captions and MSRVTT-QA (each video stored once).

    At least one of ``captions_path`` / ``qa_path`` must be given. ``captions_fmt``
    matches :func:`build_caption_shards`'s ``fmt`` (``auto``/``msrvtt``/``records``/
    ``folder``); QA is always the MSRVTT-QA list format (:func:`parse_qa_records`).
    Returns the number of videos written.
    """
    if captions_path is None and qa_path is None:
        raise ValueError("build_caption_qa_shards needs captions_path and/or qa_path")

    videos_dir = Path(videos_dir)
    if not videos_dir.is_dir():
        raise NotADirectoryError(f"--videos '{videos_dir}' is not a directory")
    video_index = index_video_files(videos_dir)
    if not video_index:
        raise FileNotFoundError(f"no video files ({', '.join(VIDEO_EXTS)}) found under {videos_dir}")

    captions: dict[str, list[str]] = {}
    if captions_path is not None:
        if captions_fmt == "auto":
            captions = load_captions_auto(captions_path, split)
        elif captions_fmt == "msrvtt":
            captions = parse_msrvtt(captions_path, split)
        elif captions_fmt == "records":
            captions = parse_caption_records(captions_path)
        elif captions_fmt == "folder":
            captions = parse_folder_sidecar(captions_path)
        else:
            raise ValueError(f"unknown captions_fmt '{captions_fmt}' (expected auto|msrvtt|records|folder)")

    qa: dict[str, list[tuple[str, str]]] = parse_qa_records(qa_path) if qa_path is not None else {}
    if not captions and not qa:
        raise ValueError("no captions and no QA pairs parsed (check paths/format/split)")

    from viora.data.webdataset_pipeline import write_video_text_shards

    logger.info(
        "%d captioned + %d QA-annotated videos -> writing combined shards",
        len(captions), len(qa),
    )
    samples = merged_shard_samples(captions, qa, video_index, limit=limit)
    return write_video_text_shards(samples, out_pattern, maxcount=maxcount)


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

    if fmt == "auto":
        captions = load_captions_auto(annotations, split)
    elif fmt == "msrvtt":
        captions = parse_msrvtt(annotations, split)
    elif fmt == "records":
        captions = parse_caption_records(annotations)
    elif fmt == "folder":
        captions = parse_folder_sidecar(annotations)
    else:
        raise ValueError(f"unknown fmt '{fmt}' (expected auto|msrvtt|records|folder)")
    if not captions:
        raise ValueError("no captions parsed from annotations (check --format / --split)")

    n_files = len(set(video_index.values()))
    logger.info("%d annotated clips | %d video files -> writing shards", len(captions), n_files)
    samples = caption_shard_samples(captions, video_index, limit=limit)
    return write_video_text_shards(samples, out_pattern, maxcount=maxcount)
