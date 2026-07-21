"""Data platform: schema, registry, decoding, sampling, datasets, mixture, collation."""

from viora.data.collators import VideoTextCollator
from viora.data.datasets import SyntheticVideoDataset
from viora.data.decoding import DecodedClip, VideoDecoder, VideoMetadata
from viora.data.mixture import STAGES, TaskAwareMixtureSampler, get_stage
from viora.data.registry import DEFAULT_REGISTRY, Availability, DatasetSpec, VioraDatasetRegistry
from viora.data.sampling import FrameSampler, build_sampler
from viora.data.schema import Task, VideoSample, validate_sample

__all__ = [
    "VideoDecoder",
    "DecodedClip",
    "VideoMetadata",
    "FrameSampler",
    "build_sampler",
    "Task",
    "VideoSample",
    "validate_sample",
    "VioraDatasetRegistry",
    "DEFAULT_REGISTRY",
    "DatasetSpec",
    "Availability",
    "SyntheticVideoDataset",
    "TaskAwareMixtureSampler",
    "STAGES",
    "get_stage",
    "VideoTextCollator",
]
