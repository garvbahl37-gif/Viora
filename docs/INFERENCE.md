# Inference

Three paths share the same encoder and the cosine‑retrieval memory baseline:
**offline indexing + QA**, **streaming**, and the **HTTP API**.

## Offline: index once, ask many

```bash
python scripts/infer.py index --video meeting.mp4 --out meeting.viora
python scripts/infer.py ask   --index meeting.viora \
       --question "What decision was made about the launch date?"
```

Indexing decodes + samples frames, runs the encoder, and serializes clip/event/
temporal tokens + timestamps to a versioned **`.viora`** file using **safetensors**
(safe to load — no code execution). Asking:

1. embeds the question (real tokenizer if present, else hashed ids);
2. retrieves the temporal regions most similar to it (cosine in the retrieval
   space) → **evidence segments with real timestamps**;
3. adds the grounding head's global span (ordered + clamped);
4. returns a structured answer.

```jsonc
{
  "answer": "...", "score": 0.41, "score_type": "uncalibrated_model_score",
  "evidence": [{ "start": 41.2, "end": 46.8, "score": 0.41 }],
  "events_used": [41.2, 44.0], "diagnostics": { "model_trained": false }
}
```

`score_type` is always explicit; when the model is untrained the answer is
evidence‑only and says so. No fabricated confidence.

## Streaming: successive chunks, bounded memory

```python
from viora.inference.streaming import StreamingVioraEngine
engine = StreamingVioraEngine(model)
engine.add_video_chunk(chunk1)          # encode once → events → memory
engine.add_video_chunk(chunk2)
engine.ask("what happened after the door opened?")
engine.get_memory_summary()             # tier counts, budget, stream time
engine.reset_memory()
```

Each chunk is encoded once; event tokens are timestamped via the event→time
attention and written to bounded `TemporalMemory` (short‑term verbatim, long‑term
compressed). The whole video is **never reprocessed**; memory respects its token
budget across chunks. Answers retrieve from memory.

## HTTP API

```bash
uvicorn viora.serving.api:app --reload    # http://127.0.0.1:8000
```

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Viora Studio web client |
| `GET /health` | model / device / trained flag / version |
| `POST /video/index` | upload a video → `index_id` (type + size validated) |
| `POST /video/ask` | `{index_id, question}` → answer + evidence |
| `POST /video/events` | temporal moments for an index |
| `POST /video/summarize` | structural summary |

Uploads are validated (extension allow‑list, size cap), written to a temp file, and
never expose arbitrary filesystem paths. Set `VIORA_CHECKPOINT` to serve trained
weights; otherwise responses are honestly marked untrained.

## Notes & limits

- Real‑LLM multimodal generation (continuing from the visual prefix) is stubbed
  pending training; the dummy decoder covers the loss/graph path.
- Retrieval is a deliberately simple, **replaceable** cosine baseline
  (`inference/memory_manager.py`).
- For very long videos, prefer streaming + retrieval over indexing the full token
  sequence.
