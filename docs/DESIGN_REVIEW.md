# Design Review

A critical review of Viora‑1 as built. The point is to find what is weak, wrong,
or unproven — not to praise the architecture. Nothing here has been validated by
training; several claims below are hypotheses.

## 1. Architectural bottlenecks

- **The resampler is a hard information bottleneck.** `Q` queries (32 in tiny) must
  summarize the entire video for the LLM. For long or dense videos this almost
  certainly loses fine detail, and there is no evidence yet that `Q=32` is enough
  or that the queries specialize usefully. Question‑conditioning exists (`QFormer`)
  but the unified model currently uses the *unconditional* `PerceiverResampler`.
- **Grounding is not question‑conditioned in the trained path.** The head's query
  is `masked_mean(temporal)` — a global pooled vector — so as trained it predicts a
  video‑level span, not a per‑question span. Question‑conditioning today comes only
  from retrieval. This is a real gap between the intended and implemented behavior.
- **Event tokens carry no explicit timestamp.** Their time is *inferred* from the
  attention map (`event→time`). If attention is diffuse, event timestamps are
  unreliable — and interpretability claims rest on that map being meaningful.
- **Empirical (local, this machine) — three findings from trying to train the
  pragmatic model locally:**
  1. **Fixed bug — projector scale mismatch.** The random-init projector emitted
     visual tokens at ~13× the (frozen) LLM's token-embedding std (0.20 vs 0.015),
     causing exploding gradients. Fixed by scale-matching the projector output to
     the LLM embedding std (`MultimodalProjector.match_embedding_scale`). This
     would also have hurt the cloud run. *Note:* with AdamW (scale-invariant) this
     alone didn't unblock learning — it is a stability fix, not the whole story.
  2. **Frozen-LLM alignment needs real-scale training.** After the fix, the model
     learned the answer *format* (loss 6.5 → ln 4) but not *which* colour from
     vision, in ~300 local steps. Aligning a frozen LLM to a new visual modality
     via LoRA is exactly the alignment stage real VLMs run with millions of
     examples — not reproducible in a tiny local budget. The from-scratch model
     (small, fully-trainable LLM) learned because it fits the task in its capacity.
  3. **Mean-pooling loses position.** A *direction* task is unlearnable through
     SigLIP mean-pooling regardless (it averages away *where* things are); keep
     per-frame **patch tokens** for spatial/motion questions.

## 2. Computational complexity

`T` frames, `H·W` resolution, `N=T'·H'·W'` tokens, `Q` resampler queries, `L`
language tokens. Factorized ViT attention is `O(t·s² + s·t²)` vs full `O(N²)` — the
main win, since `s≫t`. Temporal is `O(T'·w + g·T'²)`. LLM cost is `O(L·…)`,
decoupled from video length by the resampler. **Weakest point:** memory for
activations scales with `N` (all vision tokens are kept through every ViT block);
gradient checkpointing helps time‑for‑memory but the `N`‑sized activation is
unavoidable without token pruning/merging, which is not implemented.

## 3. GPU memory

- Full spatiotemporal attention (`attention_mode: full`) is `O(N²)` memory and will
  OOM quickly at real resolutions; factorized is the only viable default.
- The `[B,N]` boolean mask expansion for full‑attention temporal masking is another
  `O(N²)` term inside SDPA. Fine at tiny scale, wasteful at large `N`.
- No activation offloading, no FSDP sharding of the vision tower, no token
  dropping. These are the obvious next memory levers and are absent.

## 4. Data loader

- Decoding is per‑sample and single‑backend (PyAV). At scale this is likely the
  throughput bottleneck; there is no frame cache, no pre‑decoded shard format
  (WebDataset/tar), and no decord fast path wired.
- The adaptive sampler computes motion on the CPU over full frames — cheap per clip
  but not free, and it runs in the worker, competing with decode.

## 5. Long‑video scaling

- The offline pipeline indexes the *whole* token set; only streaming truly bounds
  memory. Retrieval is a single cosine baseline with **no learned scoring, no
  reranking, and no chunk‑overlap handling** — it will miss evidence that spans
  chunk boundaries.
- Long‑term memory compression is mean‑merge or eviction; mean‑merging semantically
  different events into one slot is lossy and unprincipled. Importance is an
  untrained head, so importance‑aware eviction is currently near‑random.

## 6. Distributed training

- DDP is wired but **untested on real multi‑GPU hardware** (no CUDA here). FSDP is
  claimed as "ready" only structurally — it is not implemented.
- The mixture sampler partitions by `rank::world_size` over a *materialized* index
  list; for very large `num_samples` that list is held in memory per rank. It also
  assumes all ranks share identical dataset sizes/seed — no cross‑rank consistency
  check.

## 7. Dataset imbalance

- Temperature sampling can under‑sample small but important sets (e.g. Charades‑STA
  grounding) or let a huge noisy set (HowTo100M) dominate. The example weights are
  guesses. There is no per‑task loss normalization, so a high‑magnitude loss (LM)
  can swamp low‑magnitude ones (contrastive) regardless of weights.

## 8. Catastrophic forgetting

- The curriculum trains stages sequentially with no rehearsal buffer, no EWC/LwF,
  and no replay of earlier‑stage data. Alignment learned in Stage 2 may erode
  during Stage 4 QA. This is a known risk that is entirely unmitigated in the
  current design.

## 9. Temporal hallucination

- With an untrained (and later, imperfect) model, grounding/retrieval will confidently
  point at wrong segments. The system mitigates *dishonesty* (scores labeled
  uncalibrated, untrained answers marked) but not *hallucination itself*. There is
  no abstention mechanism, no evidence‑sufficiency threshold, and no consistency
  check between the answer and the retrieved evidence.

## 10. Evaluation gaps

- Metrics are implemented and unit‑tested against hand values, but **no end‑to‑end
  benchmark has been run** — there is no dataset‑level eval harness that decodes,
  runs the model, and scores, only the metric functions and a `Trainer.evaluate`
  loop. CIDEr is gated behind an optional dep. There is no calibration evaluation,
  no human‑eval scaffold, and no per‑task leaderboard wiring.

## Highest‑leverage fixes (opinion)

1. Route the question into the resampler **and** the grounding head (close the
   conditioning gap) — likely the biggest quality lever.
2. Add token merging/pruning after the ViT to break the `N`‑sized activation cost.
3. Build a real dataset‑level eval harness before any training, so quality is
   measurable from day one.
4. Add rehearsal/replay to the curriculum to counter forgetting.
5. Learn memory importance and add an abstention/evidence‑sufficiency signal to
   attack temporal hallucination.
