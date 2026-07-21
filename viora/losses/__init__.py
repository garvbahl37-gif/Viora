"""Multi-task objectives for Viora."""

from viora.losses.captioning import CaptioningLoss, captioning_loss
from viora.losses.contrastive import VideoTextContrastiveLoss
from viora.losses.masked_video import MaskedVideoModelingLoss, masked_feature_loss
from viora.losses.matching import VideoTextMatchingLoss, sample_hard_negatives
from viora.losses.multitask import LossBreakdown, MultiTaskLossManager
from viora.losses.temporal_grounding import TemporalGroundingLoss
from viora.losses.temporal_order import TemporalOrderLoss

__all__ = [
    "VideoTextContrastiveLoss",
    "VideoTextMatchingLoss",
    "sample_hard_negatives",
    "CaptioningLoss",
    "captioning_loss",
    "TemporalGroundingLoss",
    "TemporalOrderLoss",
    "MaskedVideoModelingLoss",
    "masked_feature_loss",
    "MultiTaskLossManager",
    "LossBreakdown",
]
