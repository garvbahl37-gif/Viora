#!/usr/bin/env python3
"""Pack a dataset into WebDataset tar shards for fast cloud/multi-GPU training.

    # synthetic shards (validate the pipeline, no data needed):
    python scripts/build_shards.py --synthetic 256 --out data/shards/train-%06d.tar

    # real datasets: implement the per-dataset -> VideoSample -> (video, meta) mapping
    # in the adapter, then feed those samples here (see docs/PRODUCTION.md).
"""
from __future__ import annotations

import argparse

from viora.data.webdataset_pipeline import write_video_text_shards


def synthetic_samples(n: int):
    import torch

    from viora.data.datasets.synthetic_tasks import DIRECTIONS, PALETTE, make_video

    g = torch.Generator().manual_seed(0)
    for _ in range(n):
        direction = DIRECTIONS[int(torch.randint(0, 2, (1,), generator=g))]
        color = list(PALETTE)[int(torch.randint(0, 4, (1,), generator=g))]
        t_appear = int(torch.randint(0, 6, (1,), generator=g))
        video = make_video(direction, color, t_appear, size=64, num_frames=8, obj=16)
        ask_color = bool(torch.randint(0, 2, (1,), generator=g))
        if ask_color:
            q, a = "which color is the object", color
        else:
            q, a = "which side does it enter from", direction
        yield {"video": video, "fps": 4.0, "meta": {"question": q, "answer": a}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, help="number of synthetic samples to shard")
    ap.add_argument("--out", required=True, help="printf pattern, e.g. data/shards/train-%06d.tar")
    ap.add_argument("--maxcount", type=int, default=1000, help="samples per shard")
    args = ap.parse_args()

    if args.synthetic:
        n = write_video_text_shards(synthetic_samples(args.synthetic), args.out, maxcount=args.maxcount)
        print(f"wrote {n} synthetic samples to shards matching {args.out}")
        return 0
    raise SystemExit("provide --synthetic N (real dataset sharding: see docs/PRODUCTION.md)")


if __name__ == "__main__":
    raise SystemExit(main())
