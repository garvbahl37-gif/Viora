#!/usr/bin/env python3
"""Build Viora training shards from a real video-caption dataset.

MSR-VTT (videos dir + videodatainfo.json)::

    python scripts/prepare_video_dataset.py \
        --videos /kaggle/input/msrvtt/TrainValVideo \
        --annotations /kaggle/input/msrvtt/train_val_videodatainfo.json \
        --format msrvtt --split train \
        --out data/shards/msrvtt-train-%06d.tar

Any dataset (folder of videos + a {id: caption(s)} JSON sidecar; e.g. MSVD)::

    python scripts/prepare_video_dataset.py \
        --videos /path/to/videos --annotations captions.json \
        --format folder --out data/shards/train-%06d.tar

Each clip is stored ONCE (raw mp4 bytes, decoded/sampled at train time) with all
its reference captions in metadata. Use --limit for a quick subset first.
"""
from __future__ import annotations

import argparse

from viora.data.adapters.video_caption import build_caption_shards


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare video-caption shards for Viora training.")
    ap.add_argument("--videos", required=True, help="directory of video files (searched recursively)")
    ap.add_argument("--annotations", required=True,
                    help="MSR-VTT videodatainfo.json, or a {id: caption(s)} sidecar for --format folder")
    ap.add_argument("--format", choices=["msrvtt", "folder"], default="msrvtt")
    ap.add_argument("--split", default=None, help="MSR-VTT split filter: train | validate | test")
    ap.add_argument("--out", required=True, help="printf pattern, e.g. data/shards/msrvtt-train-%%06d.tar")
    ap.add_argument("--maxcount", type=int, default=500, help="clips per shard")
    ap.add_argument("--limit", type=int, default=None, help="cap clips (quick trial run before the full build)")
    args = ap.parse_args()

    n = build_caption_shards(
        args.videos, args.annotations, args.out,
        fmt=args.format, split=args.split, limit=args.limit, maxcount=args.maxcount,
    )
    print(f"wrote {n} clips to shards matching {args.out}")
    if n == 0:
        raise SystemExit(
            "0 clips written — video filenames likely don't match annotation ids. "
            "Check that --videos points at the folder containing the .mp4 files."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
