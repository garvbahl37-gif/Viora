"""Viora-1: a modular spatiotemporal video-language intelligence model.

The package is organised into subsystems that mirror the data flow:

    data      -> decoding, sampling, dataset adapters, mixture sampling
    models    -> embeddings, attention, vision (3D ViT), temporal, multimodal, heads
    losses    -> multi-task objectives
    training  -> trainer, checkpointing, schedulers, distributed
    evaluation-> task-aware metrics
    inference -> offline pipeline, indexing, streaming
    serving   -> FastAPI app
    utils     -> config, logging, seed, device

Top-level imports are kept light (only the version and config) so importing
``viora`` never pulls in torch-heavy submodules until they are needed.
"""

from __future__ import annotations

__version__ = "0.1.0"

from viora.utils.config import VioraConfig, load_config

__all__ = ["__version__", "VioraConfig", "load_config"]
