"""Deduplication and train/val leakage checks.

Metadata / URL / external-id dedup on annotations, plus exact file-hash dedup
where pixels exist. Perceptual (near-duplicate) video dedup is left as a future
optional module — this covers the exact/metadata cases that prevent the most
common leakage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from viora.data.schema import VideoSample


def _video_key(s: VideoSample) -> str:
    v = s.video
    return v.path or v.url or v.external_id or s.sample_id


def dedup_samples(
    samples: list[VideoSample], key_fn: Callable[[VideoSample], str] = _video_key
) -> tuple[list[VideoSample], int]:
    """Drop later samples whose key was already seen. Returns (kept, num_removed)."""
    seen: set[str] = set()
    kept: list[VideoSample] = []
    for s in samples:
        k = key_fn(s)
        if k in seen:
            continue
        seen.add(k)
        kept.append(s)
    return kept, len(samples) - len(kept)


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (for exact-duplicate detection)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_leakage(
    train: list[VideoSample], val: list[VideoSample], key_fn: Callable[[VideoSample], str] = _video_key
) -> set[str]:
    """Return keys present in both splits (train/val leakage to remove)."""
    return {key_fn(s) for s in train} & {key_fn(s) for s in val}
