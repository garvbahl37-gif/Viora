"""Task heads and multi-task losses: shapes, ranges, and NaN handling."""

from __future__ import annotations

import torch

from viora.losses.contrastive import VideoTextContrastiveLoss
from viora.losses.masked_video import masked_feature_loss
from viora.losses.matching import VideoTextMatchingLoss, sample_hard_negatives
from viora.losses.multitask import MultiTaskLossManager
from viora.losses.temporal_grounding import TemporalGroundingLoss
from viora.losses.temporal_order import TemporalOrderLoss
from viora.models.heads.classification import ClassificationHead
from viora.models.heads.confidence import ConfidenceHead
from viora.models.heads.retrieval import RetrievalHead
from viora.models.heads.temporal_grounding import TemporalGroundingHead
from viora.utils.config import GroundingConfig


# ------------------------------------------------------------------- heads
def test_grounding_head_shapes_and_ranges():
    head = TemporalGroundingHead(GroundingConfig(num_bins=32, hidden_dim=24), dim=24)
    feats = torch.randn(2, 7, 24)
    query = torch.randn(2, 24)
    out = head(feats, query)
    assert out.start_logits.shape == (2, 32) and out.end_logits.shape == (2, 32)
    assert out.start_norm.shape == (2,) and out.confidence.shape == (2,)
    assert bool((out.start_norm >= 0).all() and (out.start_norm <= 1).all())
    start_s, end_s = out.to_seconds(torch.tensor([60.0, 30.0]))
    assert start_s.shape == (2,)


def test_retrieval_head_normalized():
    head = RetrievalHead(video_dim=24, text_dim=32, proj_dim=16)
    v, t = head(torch.randn(4, 24), torch.randn(4, 32))
    assert v.shape == (4, 16) and t.shape == (4, 16)
    assert torch.allclose(v.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_classification_and_confidence_heads():
    assert ClassificationHead(24, 10)(torch.randn(3, 24)).shape == (3, 10)
    c = ConfidenceHead(24)(torch.randn(3, 24))
    assert c.shape == (3,) and bool((c >= 0).all() and (c <= 1).all())


# ------------------------------------------------------------------ losses
def test_contrastive_loss_lower_when_aligned():
    loss_fn = VideoTextContrastiveLoss(learnable=False)
    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    aligned = loss_fn(x, x)                        # perfect match
    shuffled = loss_fn(x, x[torch.randperm(8)])    # mismatched
    assert aligned < shuffled


def test_matching_loss_and_hard_negatives():
    logits = torch.randn(6, 2)
    labels = torch.randint(0, 2, (6,))
    assert VideoTextMatchingLoss()(logits, labels).item() >= 0
    sim = torch.randn(5, 5)
    neg = sample_hard_negatives(sim)
    assert neg.shape == (5,) and bool((neg != torch.arange(5)).all())


def test_grounding_loss_and_targets():
    lf = TemporalGroundingLoss()
    sl, el = torch.randn(2, 16), torch.randn(2, 16)
    sb, eb = lf.targets_from_seconds(
        torch.tensor([10.0, 5.0]), torch.tensor([20.0, 15.0]), torch.tensor([40.0, 40.0]), 16
    )
    assert sb.max() < 16 and eb.max() < 16
    assert lf(sl, el, sb, eb).item() >= 0


def test_temporal_order_loss():
    lf = TemporalOrderLoss(dim=16)
    a, b = torch.randn(5, 16), torch.randn(5, 16)
    label = torch.randint(0, 2, (5,)).float()
    assert lf(a, b, label).item() >= 0


def test_masked_video_loss_ignores_unmasked():
    pred = torch.zeros(1, 4, 8)
    target = torch.ones(1, 4, 8)
    mask = torch.tensor([[True, False, True, False]])
    loss = masked_feature_loss(pred, target, mask)
    assert loss.item() > 0  # only masked positions contribute


def test_multitask_manager_weights_and_nan():
    mgr = MultiTaskLossManager({"lm": 1.0, "grounding": 0.5})
    out = mgr.combine({"lm": torch.tensor(2.0), "grounding": torch.tensor(4.0)})
    assert abs(out.total.item() - (1.0 * 2.0 + 0.5 * 4.0)) < 1e-5
    # a NaN component is surfaced, not silently summed in
    out2 = mgr.combine({"lm": torch.tensor(1.0), "grounding": torch.tensor(float("nan"))})
    assert "grounding" in out2.nonfinite
    assert torch.isfinite(out2.total)
