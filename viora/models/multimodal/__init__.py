"""Multimodal bridge: resampler / Q-Former, projector, visual-token injection."""

from viora.models.multimodal.projector import MultimodalProjector
from viora.models.multimodal.qformer import QFormer
from viora.models.multimodal.resampler import PerceiverResampler
from viora.models.multimodal.token_injection import MultimodalBatch, build_multimodal_inputs

__all__ = [
    "PerceiverResampler",
    "QFormer",
    "MultimodalProjector",
    "build_multimodal_inputs",
    "MultimodalBatch",
]
