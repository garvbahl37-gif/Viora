# Design: Caption + Q&A Multi‑Task Training (Approach A)

- **Date:** 2026‑07‑22
- **Status:** Approved (design), pending spec review
- **Owner:** garvbahl37‑gif

## Goal

Train a single Qwen‑0.5B‑based Viora model that both **describes** clips and
**answers questions** about them — the "upload a video, ask questions, get
answers" goal — entirely on a free Kaggle T4, resuming from the existing 5k‑step
caption checkpoint rather than restarting.

## Context

- Current model (pragmatic): frozen SigLIP + frozen Qwen2.5‑0.5B + LoRA + trained
  bridge. Trained 5k steps on MSR‑VTT captions → loss ~3.1, on‑topic captions.
  Checkpoints (`final.pt`, `viora-trained.pt`) are on HF `bharatverse11/viora-msrvtt`.
- The collator (`viora/data/collators/video_text_collator.py`) already formats
  `{captions: […]}` as a caption and `{question, answer}` as
  `<video>\nQuestion:…\nAnswer:…`. Inference already has `caption()` and
  `generate_answer()`.

## Non‑goals

- No bigger LLM (Qwen‑1.5B is too slow on a T4 — decided).
- No paid GPU.
- No new datasets beyond MSR‑VTT captions + MSRVTT‑QA (reuses the same videos).
- No grounding/temporal‑evidence supervision (MSR‑VTT has no spans).

## Data design — one combined shard set, each video stored once

MSRVTT‑QA has ~23 Q&A pairs per clip; storing the video per pair would ~23× the
disk. So each video is written **once** with all its captions *and* Q&A pairs:

```json
meta = {
  "video_id": "video0",
  "captions": ["a man is singing", "..."],
  "qa": [["what is the man doing", "singing"], ["how many people", "one"]]
}
```

The collator samples **one caption or one Q&A pair per view** (mix policy below),
so a single shard set trains multi‑task with no training‑loop changes.

**QA source:** fetch MSRVTT‑QA `train_qa.json` (a top‑level list of
`{question, answer, video_id}`) from HF/GitHub — the same kind of fetch used for
captions. `video_id` may be an int or `videoN`; `_resolve_video` already handles
the `0 → video0.mp4` case.

## Code changes (small, tested)

1. **QA parser** — `viora/data/adapters/video_caption.py`:
   `parse_qa_records(path) -> {video_id: [(question, answer), …]}` for the
   top‑level‑list QA format (id/question/answer keys auto‑detected).
2. **Merged builder** — `build_caption_qa_shards(videos, captions_src, qa_src, out, …)`:
   join captions + QA per video, write one combined shard sample per resolvable
   clip via the existing `write_video_text_shards`.
3. **Collator mix** — `_format_text` learns a `qa` list and a **mix ratio**
   (`qa_prob`, default 0.5): if a sample has both, each view randomly becomes a
   caption or a `Question/Answer` example; if only one is present, use it. The
   ratio is a collator attribute so it is config‑settable.

## Training design — resume, don't restart

- Warm‑start from `final.pt` (pulled from HF) via `training.resume=`.
- Train the combined shards for **~15–25k more steps** (target ~20k), same config
  (fp16, batch 4, `num_workers=4`, cosine LR, grad clip 1.0), `save_every=500`,
  `keep_last_checkpoints=3`.
- LoRA stays r=16 on q/k/v/o (proven); revisit rank only if capacity looks binding.

## Multi‑session structure (free T4)

Kaggle output does not survive across sessions, so between sessions the notebook
**pushes the latest checkpoint to the HF repo** and the next session pulls it to
resume exactly. ~20–30 GPU‑hours ≈ 2–4 free sessions. A small `save_to_hf` +
`resume_from_hf` helper wraps `hf_hub_download` / `upload_file`.

## Evaluation (the "advanced" touch)

Hold out a small validation slice (e.g. MSRVTT‑QA `val_qa.json` + a few hundred
captions) and log `val_loss` every N steps via the trainer's existing
`evaluate()` path, so progress is *measured*, not guessed. Optionally a tiny
exact‑match QA accuracy on ~200 val questions for a human‑readable number.

## Testing (all CPU‑runnable)

- `parse_qa_records`: list `{question, answer, video_id}` → grouped pairs.
- `build_caption_qa_shards`: captions + QA → one combined shard/video; read back,
  assert meta carries both `captions` and `qa`.
- collator mix: sample with both → over many calls yields both formats; only‑QA →
  always QA; only‑captions → always caption; `qa_prob` respected at extremes (0/1).

## Inference

Add a notebook cell that loads the trained checkpoint and **answers a typed
question** about a chosen clip via `generate_answer()`, alongside the existing
caption cell.

## Risks & mitigations

- **QA annotation sourcing** (like the caption hunt) → auto‑detect list format,
  fail loud with a clear message if ids don't match filenames.
- **Catastrophic forgetting of captioning** → mixed (not staged) training keeps
  both tasks live every batch; 50/50 default.
- **Cross‑session loss** → HF checkpoint relay; keep‑last‑3 bounds disk.
- **Short MSRVTT‑QA answers** (often one word) → generative LM handles it; EOS
  already supervised so it stops cleanly.

## Rollout steps

1. QA parser + merged builder + tests.
2. Collator mix + tests.
3. HF checkpoint relay helper + notebook cells (build combined shards, resume,
   QA inference).
4. Full suite + lint green; commit; push.
5. User runs it on Kaggle across sessions.
