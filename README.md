<div align="center">

# Viora‑1

**A modular spatiotemporal video‑language model that answers questions and points to *when* the answer happened.**

`architecture preview · CPU/MPS/CUDA · 120 passing tests`

</div>

Viora is built from scratch around a custom 3D vision encoder, hierarchical
temporal reasoning, event tokenization, bounded temporal memory, a multimodal
resampler, and a pluggable language model. It is **not** an API wrapper and does
not send frames to a hosted VLM — the spatiotemporal stack is implemented in this
repository and trains end‑to‑end.

```
Video
  ↓  Adaptive sampling            (timestamps preserved)
  ↓  3D ViT (tubelet + factorized spatiotemporal attention)
  ↓  Hierarchical temporal attention (local windows → global)
  ↓  Event tokens                 (learnable queries, inspectable attention)
  ↓  Temporal memory              (short / long, bounded, timestamped)
  ↓  Multimodal resampler → projector
  ↓  Language model               (frozen / LoRA / full)
  ↓  Answer + Temporal Evidence + (uncalibrated) Confidence
```

## Status — honest by design

This repository is **architecture‑complete and test‑verified**, but the model is
**not yet trained**. Every number the system emits is labeled accordingly:
scores are `uncalibrated_model_score`, never probabilities; the demo UI and the
inference CLI clearly mark untrained, evidence‑only responses. There are **no
fabricated benchmarks** anywhere.

| Area | State |
|------|-------|
| Vision (tubelet, 3D ViT, factorized/full attention) | **Implemented + tested** |
| Temporal (hierarchical encoder, event tokens, bounded memory) | **Implemented + tested** |
| Multimodal (resampler, Q‑Former, projector, visual‑token injection) | **Implemented + tested** |
| Unified model + grounding/retrieval/heads + multi‑task losses | **Implemented + tested** |
| Data platform (schema, registry, mixture, curriculum, collator) | **Implemented + tested** |
| Training (AMP, grad‑accum/clip, checkpoint resume, distributed‑ready) | **Implemented + tested** |
| Evaluation (VideoQA / grounding IoU / retrieval R@K / BLEU / ROUGE‑L) | **Implemented + tested** |
| Inference (offline indexing, streaming memory, QA with evidence) | **Implemented + tested** |
| Serving (FastAPI) + Viora Studio web UI | **Implemented + tested** |
| Trained weights, real benchmark results | **Not done — experimental / planned** |

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"          # core + video + llm + serving + dev
python scripts/validate_environment.py
```

Runs on **CPU, Apple MPS, and CUDA** — no CUDA assumptions. (Note: Conv3d is
unsupported on MPS in current torch; use CPU or `PYTORCH_ENABLE_MPS_FALLBACK=1`.)

## Quick synthetic smoke test

```bash
pytest -q                           # 120 tests, seconds on CPU
python scripts/train.py --smoke     # a few real optimization steps on synthetic data
```

The smoke run exercises the whole pipeline (video → encoder → temporal → events →
resampler → projector → LLM → language loss) and shows the loss decreasing, the LR
schedule, and a saved checkpoint — architecture validation only.

## Try it on a real video

```bash
# index once, then ask — answers come back with temporal evidence
python scripts/infer.py index --video sample.mp4 --out sample.viora
python scripts/infer.py ask   --index sample.viora \
       --question "When did the package first appear?"
```

```jsonc
{
  "answer": "The most relevant moment is 0.3s–1.2s. (Model is untrained — evidence-only, retrieval-based.)",
  "evidence": [{ "start": 0.3, "end": 1.2, "score": 0.41 }],
  "score": 0.41,
  "score_type": "uncalibrated_model_score",
  "events_used": [0.12, 0.5, 0.88]
}
```

## Serving + the Studio UI

```bash
uvicorn viora.serving.api:app --reload      # http://127.0.0.1:8000
```

`/` serves **Viora Studio** — a temporal‑evidence workspace where asking a
question ignites the exact evidence segment on a timeline (the self‑contained page
also lives at `viora/serving/web/studio.html`). The JSON API exposes
`/video/index`, `/video/ask`, `/video/events`, `/video/summarize`, `/health`.

## Datasets

Nothing is auto‑downloaded. Each of the 12 target datasets reports its access
rules honestly (annotations‑only, manual download, auth‑gated, …) and HF repo ids
live in config, never hardcoded:

```bash
python scripts/inspect_dataset.py --list
python scripts/inspect_dataset.py --dataset nextqa --root data/nextqa
python scripts/prepare_dataset.py --dataset activitynet --root data/activitynet
```

See [docs/DATASETS.md](docs/DATASETS.md).

## Training

Curriculum stages (visual pretraining → alignment → long‑temporal → reasoning →
instruction tuning) are configured in `configs/training/`. See
[docs/TRAINING.md](docs/TRAINING.md).

## Correct answers require training

Viora is untrained, so it returns temporal *evidence*, not semantic answers — and
there is no shortcut: a network's understanding comes from training its own
weights, not from wrapping an external model. To make Viora answer, train it (see
[docs/PRODUCTION.md](docs/PRODUCTION.md)). No external answering API is used
anywhere in this codebase.

## Documentation

- [docs/PRODUCTION.md](docs/PRODUCTION.md) — train Viora for real (pragmatic recipe, cloud)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — tensor flow, every component, shapes
- [docs/DATASETS.md](docs/DATASETS.md) — roles, access caveats, licence checklist
- [docs/TRAINING.md](docs/TRAINING.md) — curriculum, single/multi‑GPU, resume
- [docs/INFERENCE.md](docs/INFERENCE.md) — offline, indexing, streaming, QA
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — living engineering log
- [docs/DESIGN_REVIEW.md](docs/DESIGN_REVIEW.md) — critical self‑review

## Limitations

- **Untrained.** No quality claims; outputs are structural/evidence‑only until trained.
- **Grounding/confidence uncalibrated** — surfaced as model scores, not probabilities.
- Real‑LLM multimodal generation is stubbed pending the training run; the dummy
  decoder covers the loss path.
- Long‑video scaling relies on the cosine‑retrieval memory baseline (replaceable).

## Roadmap

Stage‑1 pretraining → alignment → reasoning; learned memory importance; hard‑negative
contrastive; question‑conditioned grounding; causal event graph; calibrated confidence.

## License

Apache‑2.0.
