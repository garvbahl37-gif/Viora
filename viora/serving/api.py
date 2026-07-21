"""FastAPI serving app.

Serves the Viora Studio web client at ``/`` and a JSON API for indexing and
asking questions. The model is loaded lazily (so ``/`` and ``/health`` are instant)
and defaults to an **untrained** tiny model unless ``VIORA_CHECKPOINT`` is set —
responses are labeled ``model_trained`` accordingly and scores are uncalibrated.

Run:  uvicorn viora.serving.api:app --reload
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from viora import __version__  # noqa: E402
from viora.data.decoding.video_decoder import VALID_EXTENSIONS  # noqa: E402
from viora.serving.schemas import (  # noqa: E402
    AskRequest,
    AskResponse,
    EventsResponse,
    HealthResponse,
    IndexResponse,
    MomentModel,
    SummarizeRequest,
    SummarizeResponse,
)

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB API cap

app = FastAPI(title="Viora-1", version=__version__)

# lazily-built singletons
_pipeline = None
_indices: dict[str, object] = {}


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from viora.inference.pipeline import VioraInferencePipeline
        from viora.models.viora import VioraForVideoUnderstanding
        from viora.training.checkpointing import load_checkpoint
        from viora.utils.config import load_config

        cfg = load_config(os.environ.get("VIORA_MODEL_CONFIG", "configs/model/viora_tiny.yaml"))
        model = VioraForVideoUnderstanding(cfg)
        ckpt = os.environ.get("VIORA_CHECKPOINT")
        if ckpt:
            load_checkpoint(ckpt, model, map_location="cpu")
        _pipeline = VioraInferencePipeline(model, device="cpu")
    return _pipeline


def _model_trained() -> bool:
    return bool(os.environ.get("VIORA_CHECKPOINT"))


# --------------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def studio() -> FileResponse:
    return FileResponse(WEB_DIR / "studio.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        model="viora_tiny", device="cpu", model_trained=_model_trained(), version=__version__
    )


@app.post("/video/index", response_model=IndexResponse)
async def index_video(file: UploadFile = File(...)) -> IndexResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VALID_EXTENSIONS:
        raise HTTPException(400, f"unsupported file type '{suffix}'")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")
    if not data:
        raise HTTPException(400, "empty file")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    index_id = uuid.uuid4().hex[:12]
    try:
        idx = _get_pipeline().index(tmp_path)
    except Exception as e:  # noqa: BLE001 - surface decode errors as 400
        raise HTTPException(400, f"could not index video: {e}") from e
    finally:
        os.unlink(tmp_path)

    _indices[index_id] = idx
    return IndexResponse(
        index_id=index_id, duration=idx.duration,
        num_temporal_tokens=int(idx.temporal.shape[0]),
        num_frames=int(idx.metadata.get("num_frames", 0)),
    )


def _require_index(index_id: str):
    if index_id not in _indices:
        raise HTTPException(404, f"unknown index_id '{index_id}' (index a video first)")
    return _indices[index_id]


@app.post("/video/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    idx = _require_index(req.index_id)
    ans = _get_pipeline().ask(idx, req.question, top_k=req.top_k)
    return AskResponse(
        answer=ans.answer,
        evidence=[{"start": e.start, "end": e.end, "score": e.score} for e in ans.evidence],
        score=ans.score, score_type=ans.score_type, events_used=ans.events_used,
        model_trained=bool(ans.diagnostics.get("model_trained", False)),
    )


@app.post("/video/events", response_model=EventsResponse)
def events(req: SummarizeRequest) -> EventsResponse:
    idx = _require_index(req.index_id)
    moments = [MomentModel(timestamp=round(float(t), 2), index=i) for i, t in enumerate(idx.timestamps)]
    return EventsResponse(index_id=req.index_id, duration=idx.duration, moments=moments)


@app.post("/video/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    idx = _require_index(req.index_id)
    moments = [MomentModel(timestamp=round(float(t), 2), index=i) for i, t in enumerate(idx.timestamps)]
    trained = _model_trained()
    summary = (
        f"{len(moments)} temporal segments span {idx.duration:.1f}s."
        if trained
        else f"{len(moments)} temporal segments over {idx.duration:.1f}s "
             f"(untrained model — structural summary only, no generated narrative)."
    )
    return SummarizeResponse(summary=summary, moments=moments, model_trained=trained)
