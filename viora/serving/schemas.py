"""Pydantic request/response schemas for the serving API.

The ``AskResponse`` shape matches what the Viora Studio front-end already expects
(answer + evidence + uncalibrated score + events), so the UI's client can call the
real API unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    device: str
    model_trained: bool
    version: str


class IndexResponse(BaseModel):
    index_id: str
    duration: float
    num_temporal_tokens: int
    num_frames: int


class EvidenceModel(BaseModel):
    start: float
    end: float
    score: float


class AskRequest(BaseModel):
    index_id: str
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    evidence: list[EvidenceModel]
    score: float
    score_type: str
    events_used: list[float]
    model_trained: bool


class CaptionResponse(BaseModel):
    caption: str
    confidence: float          # uncalibrated mean token probability
    model_trained: bool


class MomentModel(BaseModel):
    timestamp: float
    index: int


class EventsResponse(BaseModel):
    index_id: str
    duration: float
    moments: list[MomentModel]


class SummarizeRequest(BaseModel):
    index_id: str


class SummarizeResponse(BaseModel):
    summary: str
    moments: list[MomentModel]
    model_trained: bool
