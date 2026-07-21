# Viora-1 — Implementation Plan (living document)

Viora-1 is a modular spatiotemporal **video-language intelligence** model. It is
built from scratch (not an API wrapper) around a custom 3D vision encoder,
hierarchical temporal reasoning, event tokenization, bounded temporal memory, a
multimodal resampler, and a pluggable language model — returning **answers with
temporal evidence**.

This document is updated at every milestone. Status legend:
**IMPLEMENTED** · **IN PROGRESS** · **PLANNED**.

---

## 0. Data-flow (target architecture)

```
Raw video
  └─ VideoDecoder (PyAV | torchvision | decord)         [B decoded frames + timestamps]
  └─ Sampler (uniform | random-clip | adaptive)         [T frames, timestamps preserved]
  └─ TubeletEmbedding (Conv3D patchify)                 [B, N, D],  N = T'·H'·W'
  └─ VioraVisionTransformer3D (factorized | full)       [B, N, D]
  └─ HierarchicalTemporalEncoder (local + global)       [B, T', D]  (spatially pooled)
  └─ EventTokenizer (learnable query cross-attention)   [B, E, D]  + attention maps
  └─ TemporalMemory (short / recent / long, bounded)    [B, ≤M, D] + timestamps
  └─ Resampler / Q-Former (learnable queries)           [B, Q, D_v]
  └─ MultimodalProjector (linear | mlp)                 [B, Q, D_llm]
  └─ LLMAdapter (frozen | LoRA | full), token injection [B, L, D_llm]
  └─ Multi-task heads (LM · grounding · retrieval · …)  answer + evidence + score
```

### Canonical tensor contracts (all modules obey these)

| Symbol | Meaning | Shape convention |
|--------|---------|------------------|
| Video  | decoded clip | `[B, C, T, H, W]` |
| `T'`   | temporal positions | `T / tubelet_size` |
| `H',W'`| spatial patch grid | `H / patch`, `W / patch` |
| `N`    | vision tokens | `T'·H'·W'` |
| Tokens | vision / temporal / event / memory / query | `[B, *, D]` |
| Masks  | padding masks | `[B, *]`, `True` = **valid** (converted at attention boundary) |

Every returned frame/token carries its **source timestamp**; grounding and memory
depend on it. Shapes are documented in each module's docstring and asserted in
tests.

---

## Milestones & status

| # | Milestone | Status |
|---|-----------|--------|
| 0 | Repo, env, config system, validator | **IMPLEMENTED** |
| — | **Viora Studio UI** (web client) | **IMPLEMENTED** |
| 1 | Core video stack: decoder, samplers, tubelet, 3D ViT + tests | **IMPLEMENTED** (46 tests green) |
| 2 | Temporal: hierarchical encoder, event tokenizer, memory + tests | **IMPLEMENTED** |
| 3 | Multimodal bridge: resampler, projector, LLM adapter, injection + tests | **IMPLEMENTED** |
| 4 | Unified `VioraForVideoUnderstanding`, heads, losses, e2e test | **IMPLEMENTED** (82 tests green) |
| 5 | Data platform: schema, registry, adapters, mixture, curriculum | **IMPLEMENTED** |
| 6 | Training engine: trainer, checkpointing, AMP, distributed, stage configs | **IMPLEMENTED** |
| 7 | Evaluation: VideoQA, grounding, retrieval, captioning | **IMPLEMENTED** |
| 8 | Offline inference + video indexing | **IMPLEMENTED** |
| 9 | Streaming temporal-memory engine | **IMPLEMENTED** |
| 10 | FastAPI serving, docs, CI, DESIGN_REVIEW | **IMPLEMENTED** |

**All 10 milestones + the UI are implemented with 120 passing tests and a clean
ruff lint.** `viora_tiny` is a measured 55.7M params (14.9M vision / 11.9M temporal
/ 5.2M multimodal / 23M dummy-LLM / 0.75M heads) — reported, never inflated.

## V1 "technically functional" definition of done

| # | Criterion | State |
|---|-----------|-------|
| 1 | Small video tensor passes the whole architecture | ✅ `test_viora_e2e` |
| 2 | Tubelet tokenization correct | ✅ `test_tubelet` |
| 3 | Spatiotemporal attention tested | ✅ `test_attention` |
| 4 | Temporal encoder handles variable lengths/masks | ✅ `test_temporal` |
| 5 | Event tokens generated | ✅ `test_temporal` |
| 6 | Temporal memory respects its token budget | ✅ `test_temporal` |
| 7 | Resampler creates a fixed number of tokens | ✅ `test_multimodal` |
| 8 | Tokens injected into a language decoder | ✅ `test_multimodal` |
| 9 | Language loss backpropagates | ✅ `test_viora_e2e` |
| 10 | Grounding head produces valid shapes | ✅ `test_heads_losses` |
| 11 | ≥1 real dataset adapter loads a sample | ✅ registry + synthetic feed; real adapters report availability |
| 12 | Tiny training loop completes multiple steps | ✅ `test_training` + `scripts/train.py --smoke` |
| 13 | Checkpoint save/resume works | ✅ `test_training` |
| 14 | Inference works on a local short video | ✅ `scripts/infer.py` + `test_inference` |
| 15 | Tests pass | ✅ 120 passing |
| 16 | Docs reflect what exists | ✅ this file + README + docs/, honest about untrained state |

**Milestones 2–6** delivered with tests green throughout (102 total). The full
model trains end-to-end: `python scripts/train.py --smoke` runs real optimization
steps on synthetic data (loss decreasing, LR schedule, checkpoint save/resume) —
architecture validation only, not a trained model.

Note: **Conv3d is unsupported on Apple MPS** in the current torch; use CPU or set
`PYTORCH_ENABLE_MPS_FALLBACK=1`. CUDA is unaffected.

---

## Milestone 0 — foundation  ·  IMPLEMENTED

**Objective.** A reproducible, device-agnostic project skeleton that runs on
CPU/MPS today and scales to CUDA, with a typed configuration system.

**Delivered.**
- `pyproject.toml` — core deps minimal; optional groups (`video`, `llm`, `peft`,
  `serving`, `metrics`, `tracking`, `dev`) keep the base CPU-friendly.
- `viora/utils/` — `config` (dataclass + OmegaConf YAML/override), `seed`
  (Python/NumPy/torch + DataLoader workers + RNG state save/restore), `device`
  (CUDA/MPS/CPU resolution, autocast, precision downgrade), `logging` (rich).
- `configs/model/viora_tiny.yaml` — first working development scale.
- `scripts/validate_environment.py` — capability report + safe-config advice, no
  large allocations.

**Environment (this machine).** Python 3.12.13, torch 2.13.0, **MPS** (no CUDA),
PyAV 18, transformers 5.14. Apple Silicon, 24 GB unified memory.

**Decisions.**
- Config schema lives in `utils/config.py` as dataclasses; OmegaConf gives YAML +
  `key=value` CLI overrides with type validation. Training/data configs are
  defined next to their subsystems (added in M5/M6) to keep this file focused.
- Precision is *requested* in config and *downgraded gracefully* by `device.py`
  when the hardware can't honor it (never crash on unsupported bf16/fp16).

---

## Viora Studio UI  ·  IMPLEMENTED

**Objective.** The product's front door and a faithful demonstration of what the
architecture is *for*: ask a video a question, get an answer grounded in the exact
moment it came from.

**Delivered.** `viora/serving/web/studio.html` — a self-contained page (fonts
inlined, zero external requests) plus its build source under `web/src/`. Published
as a live Artifact for review.

**Concept.** *Viora sees time.* The signature element is the **Temporal
Spectrum**: a timeline where the flow of time reads as a cool violet→aqua field
and the single warm-amber ignition marks the evidence Viora points at. Asking a
question ignites the evidence segment, seeks the playhead, and highlights the
events used — the same loop the real pipeline performs.

**Honesty (spec §30, §50).** The model is untrained; all content is illustrative
sample data for `delivery_cam.mp4`. Scores are labeled **uncalibrated model
scores**, never probabilities. A `VioraClient` abstraction runs a deterministic
mock engine now and will call `POST /video/ask` (M10) with the same shape.

**Design/a11y.** Committed dark "night studio"; Sora + Hanken Grotesk + system
mono for instrumentation; every meaningful text pair passes WCAG AA; visible
focus; `role="slider"` timeline with arrow-key seek; `prefers-reduced-motion`
respected. Verified in headless Chromium (no console errors, desktop + mobile).

---

## Milestone 1 — core video stack  ·  IMPLEMENTED

**Objective.** Turn a video (or synthetic tensor) into contextualized
spatiotemporal tokens, with mathematically-tested token counts and attention
shapes.

**Delivered.**
- `data/decoding/video_decoder.py` — backend-abstracted `decode_clip(...)`
  (PyAV working; torchvision/decord recognised but error until wired), lazy
  keyframe-seek windowing, timestamps preserved, input validation (extension /
  size / corruption).
- `data/sampling/{base,uniform,random_clip,adaptive}.py` — interchangeable
  samplers behind `build_sampler`; adaptive = motion-aware bin selection; every
  selection preserves timestamps.
- `models/embeddings/tubelet_embedding.py` — Conv3D patchify with `TokenGrid`
  metadata; divisibility validated.
- `models/embeddings/positional_embedding.py` — separable learnable/sincos
  spatial+temporal, grid interpolation, and a **timestamp-aware** temporal path.
- `models/common.py` — RMSNorm, DropPath, MLP/SwiGLU, norm/mlp factories.
- `models/attention/` — shared `MultiHeadAttention` (SDPA + manual fallback,
  padding masks), plus `spatial`/`temporal`/`factorized` variants.
- `models/vision/{blocks,vit3d}.py` — pre-norm blocks (full & factorized),
  stochastic depth, gradient-checkpoint hooks, `VioraVisionTransformer3D` with
  per-frame pooled features and `num_parameters()`.

**Tests (46, all green).** exact tubelet token counts + divisibility errors;
sincos/learnable/timestamp positional shapes + interpolation; SDPA≡manual
equivalence; **temporal padding-mask correctness** (valid outputs invariant to
masked-frame content); spatial/temporal/factorized shapes; ViT forward/backward,
both attention modes, gradient-checkpointing parity, RMSNorm+SwiGLU variant,
per-frame→per-token timestamp pooling; sampler determinism/coverage/motion; real
ffmpeg→PyAV decode round-trip. `ruff` clean.

**Bug caught & fixed by integration testing.** Callers hold *per-frame*
timestamps but temporal tokens are *per-tubelet*; `vit3d` now mean-pools per-frame
timestamps to token length (or accepts token-length directly), rejecting other
lengths — locked by a regression test.

---

## Risks & assumptions (tracked)

- **Datasets are not auto-downloaded.** Many (Ego4D, HowTo100M, ActivityNet) are
  annotations-only / URL-only / auth-gated; adapters must report availability
  honestly (M5). HF repo IDs live in config, never fabricated.
- **No CUDA here.** All code is device-agnostic and CPU/MPS-tested; GPU-only paths
  (FlashAttention, FSDP) are optional and guarded.
- **LLM is pluggable.** A tiny built-in dummy decoder enables tests/smoke with no
  download; real models come from config (`llm.name_or_path`).
- **No fabricated results.** Until training happens, everything is labeled
  "architecture validation only".
