"""VideoQA metrics: open-ended accuracy (normalized exact match) and MC accuracy."""

from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style)."""
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def qa_accuracy(predictions: list[str], references: list[str]) -> float:
    """Fraction of normalized exact matches."""
    if not predictions:
        return 0.0
    correct = sum(normalize_answer(p) == normalize_answer(r) for p, r in zip(predictions, references, strict=True))
    return correct / len(predictions)


def multiple_choice_accuracy(pred_idx: list[int], gt_idx: list[int]) -> float:
    if not pred_idx:
        return 0.0
    return sum(p == g for p, g in zip(pred_idx, gt_idx, strict=True)) / len(pred_idx)
