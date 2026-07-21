"""Evaluation metrics verified against hand-computed values."""

from __future__ import annotations

import torch

from viora.evaluation.captioning import corpus_bleu, rouge_l
from viora.evaluation.evaluator import Evaluator, measure_systems
from viora.evaluation.retrieval import recall_at_k
from viora.evaluation.temporal_grounding import grounding_metrics, temporal_iou
from viora.evaluation.videoqa import multiple_choice_accuracy, normalize_answer, qa_accuracy


def test_qa_normalization_and_accuracy():
    assert normalize_answer("The Dog.") == "dog"
    assert qa_accuracy(["the dog", "a cat"], ["Dog", "dog"]) == 0.5


def test_multiple_choice_accuracy():
    assert multiple_choice_accuracy([0, 1, 2], [0, 1, 3]) == 2 / 3


def test_temporal_iou_exact():
    assert temporal_iou((0, 10), (0, 10)) == 1.0
    assert temporal_iou((0, 10), (5, 15)) == 5 / 15  # inter 5, union 15
    assert temporal_iou((0, 5), (10, 15)) == 0.0


def test_grounding_metrics():
    preds = [(0, 10), (0, 10)]
    refs = [(0, 10), (5, 15)]  # IoUs: 1.0, 0.333
    m = grounding_metrics(preds, refs, thresholds=(0.5,))
    assert abs(m["mIoU"] - (1.0 + 1 / 3) / 2) < 1e-6
    assert m["R@0.5"] == 0.5


def test_recall_at_k_perfect_and_worst():
    perfect = torch.eye(5)  # diagonal is max -> rank 0
    r = recall_at_k(perfect, ks=(1,))
    assert r["R@1"] == 1.0 and r["median_rank"] == 1.0
    # a matrix where the correct answer is always ranked last
    bad = torch.eye(4) * -1.0 + 1.0  # diagonal is min
    r2 = recall_at_k(bad, ks=(1,))
    assert r2["R@1"] == 0.0


def test_bleu_and_rouge():
    assert corpus_bleu(["the cat sat"], ["the cat sat"]) > 0.99  # exact match
    assert corpus_bleu(["dog"], ["the cat sat on the mat"]) < 0.5
    assert rouge_l(["the cat sat"], ["the cat sat"]) > 0.99
    assert 0.0 <= rouge_l(["the dog"], ["the cat"]) < 1.0


def test_evaluator_dispatch():
    ev = Evaluator()
    assert ev.videoqa(["a"], ["a"]).metrics["accuracy"] == 1.0
    assert ev.grounding([(0, 10)], [(0, 10)]).metrics["mIoU"] == 1.0
    assert ev.retrieval(torch.eye(3)).metrics["R@1"] == 1.0


def test_measure_systems_returns_timing():
    out = measure_systems(lambda: sum(range(1000)), iters=3)
    assert out["latency_ms"] >= 0 and out["throughput_per_s"] > 0
