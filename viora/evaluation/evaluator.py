"""Evaluation orchestration.

Dispatches per-task quality metrics and keeps **systems** metrics (latency,
throughput, peak memory) separate from model-quality metrics — the two answer
different questions and must not be conflated.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from viora.evaluation.captioning import captioning_metrics
from viora.evaluation.retrieval import recall_at_k
from viora.evaluation.temporal_grounding import Segment, grounding_metrics
from viora.evaluation.videoqa import multiple_choice_accuracy, qa_accuracy


@dataclass
class EvaluationResult:
    task: str
    metrics: dict[str, float]
    num_samples: int
    systems: dict[str, float] = field(default_factory=dict)


def measure_systems(fn: Callable[[], object], *, warmup: int = 1, iters: int = 5) -> dict[str, float]:
    """Time ``fn`` (already-bound) and report latency / throughput / peak GPU memory."""
    cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    out = {
        "latency_ms": 1000.0 * elapsed / iters,
        "throughput_per_s": iters / elapsed if elapsed > 0 else 0.0,
    }
    if cuda:
        out["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    return out


class Evaluator:
    """Thin dispatcher returning :class:`EvaluationResult` per task."""

    def videoqa(self, predictions: list[str], references: list[str]) -> EvaluationResult:
        return EvaluationResult("video_qa", {"accuracy": qa_accuracy(predictions, references)}, len(predictions))

    def multiple_choice(self, pred_idx: list[int], gt_idx: list[int]) -> EvaluationResult:
        return EvaluationResult("video_qa_mc", {"accuracy": multiple_choice_accuracy(pred_idx, gt_idx)}, len(pred_idx))

    def grounding(self, predictions: list[Segment], references: list[Segment]) -> EvaluationResult:
        return EvaluationResult("temporal_grounding", grounding_metrics(predictions, references), len(predictions))

    def retrieval(self, similarity: torch.Tensor) -> EvaluationResult:
        return EvaluationResult("retrieval", recall_at_k(similarity), int(similarity.shape[0]))

    def captioning(self, candidates: list[str], references: list[str]) -> EvaluationResult:
        return EvaluationResult("captioning", captioning_metrics(candidates, references), len(candidates))
