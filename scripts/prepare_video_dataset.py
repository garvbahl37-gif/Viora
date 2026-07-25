#!/usr/bin/env python3
"""Build Viora training shards from a real video-caption (+ optional QA) dataset.

MSR-VTT captions only (videos dir + videodatainfo.json)::

    python scripts/prepare_video_dataset.py \
        --videos /kaggle/input/msrvtt/TrainValVideo \
        --annotations /kaggle/input/msrvtt/train_val_videodatainfo.json \
        --format msrvtt --split train \
        --out data/shards/msrvtt-train-%06d.tar

Any dataset (folder of videos + a {id: caption(s)} JSON sidecar; e.g. MSVD)::

    python scripts/prepare_video_dataset.py \
        --videos /path/to/videos --annotations captions.json \
        --format folder --out data/shards/train-%06d.tar

Captions + MSRVTT-QA COMBINED (one shard set, both tasks; add --qa-annotations)::

    python scripts/prepare_video_dataset.py \
        --videos /kaggle/input/msrvtt/TrainValVideo \
        --annotations /kaggle/input/msrvtt/train_val_videodatainfo.json \
        --qa-annotations /path/to/train_qa.json \
        --format msrvtt --split train \
        --out data/shards/msrvtt-mixed-%06d.tar

QA only (no --annotations; the collator falls back to QA-only views)::

    python scripts/prepare_video_dataset.py \
        --videos /kaggle/input/msrvtt/TrainValVideo \
        --qa-annotations /path/to/train_qa.json \
        --out data/shards/msrvtt-qa-%06d.tar

Each clip is stored ONCE (raw mp4 bytes, decoded/sampled at train time) with all
its reference captions and/or QA pairs in metadata. Use --limit for a quick
subset first.
"""
from __future__ import annotations

import argparse

from viora.data.adapters.video_caption import build_caption_qa_shards, build_caption_shards


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare video-caption(+QA) shards for Viora training.")
    ap.add_argument("--videos", required=True, help="directory of video files (searched recursively)")
    ap.add_argument("--annotations", default=None,
                    help="MSR-VTT videodatainfo.json, or a {id: caption(s)} sidecar for --format folder. "
                         "Omit if using --qa-annotations alone (QA-only shards).")
    ap.add_argument("--qa-annotations", default=None,
                    help="MSRVTT-QA json: a top-level list of {video_id, question, answer} records. "
                         "Combine with --annotations for a single mixed caption+QA shard set.")
    ap.add_argument("--format", choices=["auto", "msrvtt", "records", "folder"], default="auto",
                    help="caption annotation format (ignored if --annotations is omitted): "
                         "auto-detect (default); msrvtt=videodatainfo.json; "
                         "records=list of {video_id, caption}; folder={id: caption(s)} object")
    ap.add_argument("--split", default=None, help="MSR-VTT split filter: train | validate | test")
    ap.add_argument("--out", required=True, help="printf pattern, e.g. data/shards/msrvtt-train-%%06d.tar")
    ap.add_argument("--maxcount", type=int, default=500, help="clips per shard")
    ap.add_argument("--limit", type=int, default=None, help="cap clips (quick trial run before the full build)")
    args = ap.parse_args()

    if args.qa_annotations is not None:
        n = build_caption_qa_shards(
            args.videos, args.out,
            captions_path=args.annotations, captions_fmt=args.format, qa_path=args.qa_annotations,
            split=args.split, limit=args.limit, maxcount=args.maxcount,
        )
    else:
        if args.annotations is None:
            raise SystemExit("provide --annotations and/or --qa-annotations")
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
