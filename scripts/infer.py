#!/usr/bin/env python3
"""Offline video inference CLI: index a video, then ask it questions.

    python scripts/infer.py index --video sample.mp4 --out sample.viora
    python scripts/infer.py ask   --index sample.viora --question "When did the package first appear?"

With no --checkpoint, an UNTRAINED tiny model is used — answers are evidence-only
and explicitly labeled as such (no fabricated results).
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from viora.inference.pipeline import VideoIndex, VioraInferencePipeline  # noqa: E402
from viora.models.viora import VioraForVideoUnderstanding  # noqa: E402
from viora.training.checkpointing import load_checkpoint  # noqa: E402
from viora.utils.config import load_config  # noqa: E402
from viora.utils.logging import configure_logging  # noqa: E402


def _build_model(model_cfg_path: str, checkpoint: str | None) -> VioraForVideoUnderstanding:
    cfg = load_config(model_cfg_path)
    model = VioraForVideoUnderstanding(cfg)
    if checkpoint:
        load_checkpoint(checkpoint, model, map_location="cpu")
    return model


def main() -> int:
    configure_logging("WARNING")
    ap = argparse.ArgumentParser(description="Viora offline inference.")
    sub = ap.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="index a video into a .viora file")
    pi.add_argument("--video", required=True)
    pi.add_argument("--out", required=True)
    pi.add_argument("--model", default="configs/model/viora_tiny.yaml")
    pi.add_argument("--checkpoint")

    pa = sub.add_parser("ask", help="ask a question against a .viora index")
    pa.add_argument("--index", required=True)
    pa.add_argument("--question", required=True)
    pa.add_argument("--model", default="configs/model/viora_tiny.yaml")
    pa.add_argument("--checkpoint")

    args = ap.parse_args()
    model = _build_model(args.model, args.checkpoint)
    pipe = VioraInferencePipeline(model, device="cpu")

    if args.command == "index":
        idx = pipe.index(args.video)
        idx.save(args.out)
        print(f"indexed {args.video} -> {args.out}  "
              f"(duration {idx.duration:.1f}s, {idx.temporal.shape[0]} temporal tokens)")
        return 0

    idx = VideoIndex.load(args.index)
    answer = pipe.ask(idx, args.question)
    print(json.dumps(answer.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
