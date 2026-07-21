"""Task-aware evaluation: VideoQA, temporal grounding, retrieval, captioning."""

from viora.evaluation.captioning import captioning_metrics, corpus_bleu, rouge_l
from viora.evaluation.evaluator import EvaluationResult, Evaluator, measure_systems
from viora.evaluation.retrieval import recall_at_k
from viora.evaluation.temporal_grounding import grounding_metrics, temporal_iou
from viora.evaluation.videoqa import multiple_choice_accuracy, normalize_answer, qa_accuracy

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "measure_systems",
    "qa_accuracy",
    "multiple_choice_accuracy",
    "normalize_answer",
    "grounding_metrics",
    "temporal_iou",
    "recall_at_k",
    "captioning_metrics",
    "corpus_bleu",
    "rouge_l",
]
