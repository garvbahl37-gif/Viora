"""Task heads: temporal grounding, retrieval, classification, confidence."""

from viora.models.heads.classification import ClassificationHead
from viora.models.heads.confidence import ConfidenceHead
from viora.models.heads.retrieval import RetrievalHead
from viora.models.heads.temporal_grounding import GroundingOutput, TemporalGroundingHead

__all__ = [
    "TemporalGroundingHead",
    "GroundingOutput",
    "RetrievalHead",
    "ClassificationHead",
    "ConfidenceHead",
]
