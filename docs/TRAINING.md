# Training

Native‑PyTorch trainer (`viora/training/trainer.py`): mixed precision, gradient
accumulation + clipping, LR warmup/schedule, checkpoint save/resume, periodic
validation, per‑component loss logging, and NaN/Inf detection that surfaces
(never hides) failures. DDP is optional and costs nothing when unused.

## Smoke run (no data)

```bash
python scripts/train.py --smoke
```

A few real optimization steps on synthetic video — verifies the whole pipeline,
the LR schedule, and checkpointing. Runs on CPU in seconds. Architecture
validation only.

## Curriculum

Stages (`viora/data/mixture/curriculum.py`), each with a config in
`configs/training/`:

| Stage | Focus | Data | Objectives |
|-------|-------|------|-----------|
| 1 | Visual/temporal pretraining | Something‑Something v2 | classification, masked video |
| 2 | Video‑language alignment | WebVid, InternVid, MSR‑VTT | contrastive, matching, LM |
| 3 | Long temporal understanding | ActivityNet, Charades‑STA, Ego4D, HowTo100M | grounding, LM, order |
| 4 | Video reasoning | NExT‑QA, TGIF‑QA, MSVD‑QA | LM (QA) |
| 5 | Instruction tuning | VideoInstruct | LM |

Mixture ratios are **examples, not tuned defaults**. Not every stage must run to
exercise the repo.

```bash
python scripts/train.py \
  --model configs/model/viora_tiny.yaml \
  --train configs/training/stage2_alignment.yaml
```

Config overrides are dotlist args: `training.max_steps=50000 training.lr=5e-5`.

## Single vs multi‑GPU

- **Single GPU / CPU / MPS:** just run the script. Precision auto‑downgrades if the
  device can't honor the request (bf16→fp32 on MPS, etc.).
- **Multi‑GPU (DDP):** launch under `torchrun`; `training/distributed.py` reads the
  env, and `TaskAwareMixtureSampler(rank=…, world_size=…)` gives each rank a
  disjoint, deterministic slice of the mixture.

```bash
torchrun --nproc_per_node=8 scripts/train.py --train configs/training/stage2_alignment.yaml
```

## Memory optimization

`freeze_vision` / `freeze_llm`, `gradient_checkpointing`, `precision: bf16|fp16`,
smaller `num_frames` / `image_size` / `tubelet_size`, `grad_accum`. On a small GPU,
freeze the LLM, checkpoint the vision blocks, and accumulate.

## Checkpoints & resume

`save_checkpoint` bundles model + optimizer + scheduler + AMP scaler + step/epoch +
resolved config + RNG states + version (atomic write). Resume with
`training.resume=runs/exp/step_2000.pt`; architecture mismatches raise an
understandable error rather than a shape dump.

> Security: full resume unpickles optimizer/RNG state (`weights_only=False`) and is
> for **your own** checkpoints. `load_checkpoint(..., trusted=False)` loads weights
> only under `weights_only=True` for untrusted files.

## Experiment tracking

Each run writes `run_meta.json` (device, precision, seed, parameter counts,
timestamp) and the resolved config to `output_dir`. TensorBoard is optional
(`training.tensorboard=true`); nothing requires an external account.
