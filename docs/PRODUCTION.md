# Taking Viora to production

The honest recipe real teams use (LLaVA / Video‑LLaVA / Qwen‑VL / VideoChat).
**You do not pretrain the LLM or the vision encoder from scratch** — you stand on
pretrained models and train the connective tissue + LoRA. From‑scratch is a
multi‑month, six‑figure research project; the pragmatic path ships a genuinely
usable model in days for hundreds–thousands of dollars.

## The components (already wired in this repo)

| Piece | Choice | Trained? |
|-------|--------|----------|
| Vision | frozen **SigLIP** per‑frame (`vision.backbone: pretrained`) | frozen |
| Temporal / events / memory | Viora's stack | **trained** |
| Resampler + projector | Viora's bridge | **trained** |
| Language | **Qwen2.5** via `transformers` (0.5B → 7B) | **LoRA** |
| Heads | grounding / retrieval / confidence | **trained** |

Config: [`configs/model/viora_pragmatic.yaml`](../configs/model/viora_pragmatic.yaml).
No Ollama — Ollama is a black‑box chat server; we need embeddings + gradients, so
we load the LLM with Hugging Face `transformers` directly.

Validate it assembles + runs with real weights on your machine:

```bash
python scripts/check_pragmatic.py
```

**Prove it learns locally** (no wrapper, no external API — Viora's own weights):

```bash
# from-scratch model — learns cleanly on this Mac (100% on the synthetic task):
python scripts/train_synthetic.py --steps 800

# production model (frozen SigLIP + frozen Qwen + trainable bridge + LoRA):
python scripts/train_pragmatic_local.py --task color --steps 300   # content -> learns
python scripts/train_pragmatic_local.py --task direction           # spatial -> see note
```

Honest findings on this M5 Pro (~0.5 steps/s):
- The **from-scratch** 3D-ViT model learns the synthetic task well (chance→~0.8–1.0) — its own
  weights, no wrapper. This is the clean "trained locally" proof.
- The **production** model (frozen SigLIP + frozen Qwen + LoRA) learns the answer *format* locally
  but **not vision-grounding in a small budget** — aligning a frozen LLM to a new modality via LoRA
  needs real-scale training (the cloud run), not 300 Mac steps. Debugging this surfaced + fixed a
  real projector **scale-mismatch bug** that would also have hurt the cloud run.
- A **spatial** (direction) task is additionally unlearnable through SigLIP **mean-pooling** (it
  discards position — keep patch tokens). See [DESIGN_REVIEW.md](DESIGN_REVIEW.md).

Bottom line: local training proves the from-scratch architecture *learns*; the production model's
real quality needs the datasets + GPU run below. No shortcut, no wrapper.

## Two‑stage training

**Stage A — alignment** (teach the bridge to speak the LLM's language). Freeze
SigLIP *and* the LLM; train only temporal + resampler + projector on ~1–5M
image/video‑caption pairs. Fast, cheap, stabilizes everything.

**Stage B — instruction tuning** (teach it to answer). Keep SigLIP frozen; enable
**LoRA** on the LLM; train on video‑instruction + QA data (VideoInstruct‑100K,
LLaVA‑Video, NExT‑QA, MSVD‑QA). This is what makes it answer questions well.

```bash
# Stage A (alignment): freeze the LLM
python scripts/train.py --model configs/model/viora_pragmatic.yaml \
  --train configs/training/pragmatic_lora.yaml training.freeze_llm=true \
  'training.loss_weights={lm: 1.0, contrastive: 1.0}'

# Stage B (instruction tuning): LoRA on
python scripts/train.py --model configs/model/viora_pragmatic.yaml \
  --train configs/training/pragmatic_lora.yaml training.freeze_llm=false
```

## Data — what to actually download

Start with **one** downloadable set to prove the loop, then add more:

| Purpose | Dataset | Size | How |
|---------|---------|------|-----|
| QA (start here) | **MSR‑VTT** | ~7 GB | direct download; `prepare_dataset.py --dataset msrvtt` |
| Reasoning QA | NExT‑QA, MSVD‑QA | ~10–40 GB | manual download (VidOR / MSVD) |
| Instructions | VideoInstruct‑100K | annotations + ActivityNet videos | fetch referenced videos |
| Alignment | WebVid / CC3M / LLaVA‑Pretrain | large | scrape URLs / HF |

Each adapter reports availability honestly; nothing auto‑downloads. Convert to
sharded `WebDataset` tars for throughput at scale (recommended for cloud).

## Compute & cost (realistic)

| Where | What you can do | Time | Cost |
|-------|-----------------|------|------|
| **This M5 Pro** | assemble + forward + generate; tiny LoRA validation on a few hundred clips | minutes–hours | $0 |
| **1× A100 (rented)** | usable 0.5–1.5B model on MSR‑VTT/NExT‑QA | ~1 day | ~$30–60 |
| **8× A100/H100** | 7B, multi‑dataset, both stages | 2–4 days | ~$500–3,000 |

The Mac validates correctness; **real quality needs a rented GPU.** The trainer is
AMP + DDP‑ready (see the cloud recipe below).

> MPS note: Conv3d isn't supported on Apple MPS — irrelevant here because the
> pragmatic path uses SigLIP (Conv2d), which runs on MPS. Only the from‑scratch 3D
> ViT hits the Conv3d gap.

## Cloud recipe (multi‑GPU) — one command

On a fresh 8×A100 node:

```bash
git clone <repo> && cd viora
bash scripts/cloud/launch.sh 8 Qwen/Qwen2.5-7B-Instruct
```

That script sets up the CUDA env, validates GPUs, builds shards (synthetic
fallback if none exist), and launches **FSDP** training. Data is streamed from
**WebDataset** tar shards (`viora/data/webdataset_pipeline.py`), split across
ranks. Build shards from your data with:

```bash
python scripts/build_shards.py --synthetic 2000 --out data/shards/train-%06d.tar   # validate
# real data: map your dataset -> (video, {question/answer/caption}) and feed write_video_text_shards
```

Under the hood:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --model configs/model/viora_pragmatic.yaml \
  --train configs/training/pragmatic_lora.yaml \
  --shards 'data/shards/train-{000000..000099}.tar' \
  llm.name_or_path=Qwen/Qwen2.5-7B-Instruct \
  training.fsdp=true training.precision=bf16 training.batch_size=8
```

`TaskAwareMixtureSampler` gives each rank a disjoint, deterministic slice; **FSDP**
(`training.fsdp=true`) shards the 7B LLM across GPUs (`wrap_fsdp` in
`training/distributed.py`, transformer‑layer auto‑wrap). Serve the result:

```bash
VIORA_MODEL_CONFIG=configs/model/viora_pragmatic.yaml VIORA_CHECKPOINT=runs/pragmatic/final.pt \
  uvicorn viora.serving.api:app --host 0.0.0.0 --port 8000
```

## Evaluate

```bash
python scripts/evaluate.py --model ... --checkpoint runs/pragmatic/best.pt \
  --dataset nextqa   # accuracy; grounding mIoU on charades_sta; retrieval R@K on msrvtt
```

Metrics are implemented ([`viora/evaluation`](../viora/evaluation)); the
dataset‑level harness that decodes+runs+scores is the last piece to wire before a
real run — build it first so quality is measurable from step 0.

## Scaling knobs

- **LLM size:** 0.5B (Mac) → 1.5B/3B (1 GPU) → 7B (8 GPU). Just change `llm.name_or_path`.
- **Frames / resolution:** more frames = better temporal reasoning, more memory.
- **Vision:** SigLIP‑base → SigLIP‑large / InternVideo for quality.
- **QLoRA** (4‑bit) to fit bigger LLMs on smaller GPUs (`bitsandbytes` + peft).

## Proven vs. pending

- ✅ Architecture learns (synthetic task: 100% held‑out) — `runs/synth/best.pt`.
- ✅ Real LLM (Qwen2.5) integrates: forward + generation through Viora.
- ✅ Pretrained SigLIP vision path wired; pragmatic model assembles + runs.
- ⏳ The **training run on real data** (cloud) — that's what produces quality.
- ⏳ Dataset‑level eval harness.
