#!/usr/bin/env python3
"""Assemble and validate the PRODUCTION pragmatic model on this machine.

Builds Viora with a frozen pretrained SigLIP vision encoder + a real Qwen2.5 LLM
with LoRA adapters, then runs a forward on a video + question and a real
generation. Proves the production architecture assembles and runs with real
pretrained components (untrained connective tissue — quality comes from training).

    python scripts/check_pragmatic.py
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from viora.models.multimodal.token_injection import build_multimodal_inputs  # noqa: E402
from viora.models.viora import VioraForVideoUnderstanding  # noqa: E402
from viora.utils.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config("configs/model/viora_pragmatic.yaml")
    print("building pragmatic model (downloads SigLIP + uses cached Qwen)...")
    model = VioraForVideoUnderstanding(cfg).eval()

    counts = model.count_parameters()
    print("\nparameters:")
    for k in ("total", "vision", "temporal", "multimodal", "llm", "heads"):
        print(f"  {k:11s} {counts[k]:>14,}")
    print(f"  {'trainable':11s} {counts['trainable']:>14,}  "
          f"({100*counts['trainable']/counts['total']:.1f}% — the connective tissue + LoRA)")

    tok = model.llm.tokenizer
    video = torch.randn(1, 3, cfg.vision.num_frames, 224, 224)  # a (random) clip
    question = "<video>\nQuestion: What is happening in this video?\nAnswer:"
    ids = torch.tensor([tok(question).input_ids])

    with torch.no_grad():
        # encode video -> resampled -> project -> inject -> LLM forward
        enc = model.encode_video(video)
        print(f"\nencode OK: frame feats {tuple(enc['temporal'].shape)}, "
              f"resampled {tuple(enc['resampled'].shape)}")
        projected = model.projector(enc["resampled"])
        mm = build_multimodal_inputs(ids, projected, model.llm.get_input_embeddings(),
                                     video_token_id=model.llm.video_token_id,
                                     attention_mask=torch.ones_like(ids))
        out = model.llm(mm.inputs_embeds, mm.attention_mask)
        nxt = out.logits[:, -1].argmax(-1)
        print(f"forward OK: injected seq {tuple(mm.inputs_embeds.shape)}, "
              f"next-token -> {tok.decode(nxt)!r}")

    print("\n✅ production architecture assembles and runs with REAL pretrained vision + LLM.")
    print("   Untrained connective tissue -> answers are not meaningful yet; that needs the")
    print("   LoRA training run (scripts/train.py on real data; cloud for full quality).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
