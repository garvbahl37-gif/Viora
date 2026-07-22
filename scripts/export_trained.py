#!/usr/bin/env python3
"""Export only the TRAINED weights from a full checkpoint (small + portable).

A full checkpoint bundles the frozen SigLIP + LLM base (~GBs) + optimizer state.
For inference/sharing you only need the trained params (LoRA + bridge + heads);
the frozen base reloads from its pretrained source. Load the result with
``load_checkpoint(out, model, strict=False)``.

    python scripts/export_trained.py \
        --model configs/model/viora_pragmatic.yaml \
        --checkpoint runs/pragmatic/final.pt --out viora-trained.pt \
        llm.name_or_path=Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import os

from viora.models.viora import VioraForVideoUnderstanding
from viora.training.checkpointing import export_trained_weights
from viora.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Export only the trained weights (small checkpoint).")
    ap.add_argument("--model", default="configs/model/viora_pragmatic.yaml")
    ap.add_argument("--checkpoint", required=True, help="full checkpoint, e.g. runs/pragmatic/final.pt")
    ap.add_argument("--out", required=True, help="output path, e.g. viora-trained.pt")
    ap.add_argument("overrides", nargs="*", help="model config overrides (e.g. llm.name_or_path=...)")
    args = ap.parse_args()

    cfg = load_config(args.model, overrides=args.overrides)
    model = VioraForVideoUnderstanding(cfg)
    kept, total = export_trained_weights(model, args.checkpoint, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"exported {kept}/{total} tensors -> {args.out} ({size_mb:.0f} MB). "
          "Reload with load_checkpoint(out, model, strict=False).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
