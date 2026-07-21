"""Captioning metrics: BLEU-n and ROUGE-L implemented in-repo (no heavy deps).

CIDEr is only reported when ``pycocoevalcap`` is installed — we never invent it.
"""

from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(candidates: list[str], references: list[str], max_n: int = 4) -> float:
    """Corpus BLEU-``max_n`` with a brevity penalty (single reference per candidate)."""
    if not candidates:
        return 0.0
    clipped = [0] * max_n
    totals = [0] * max_n
    cand_len = ref_len = 0
    for cand, ref in zip(candidates, references, strict=True):
        ct, rt = cand.split(), ref.split()
        cand_len += len(ct)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            cand_ng = _ngrams(ct, n)
            ref_ng = _ngrams(rt, n)
            overlap = sum(min(c, ref_ng[g]) for g, c in cand_ng.items())
            clipped[n - 1] += overlap
            totals[n - 1] += max(0, len(ct) - n + 1)
    # only n-gram orders that actually occur (short sentences lack high-order grams)
    precisions = [clipped[i] / totals[i] for i in range(max_n) if totals[i] > 0]
    if not precisions or min(precisions) <= 0:
        return 0.0
    log_p = sum(math.log(p) for p in precisions) / len(precisions)
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(1, cand_len))
    return bp * math.exp(log_p)


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(candidates: list[str], references: list[str], beta: float = 1.2) -> float:
    """Mean sentence-level ROUGE-L F-measure."""
    if not candidates:
        return 0.0
    scores = []
    for cand, ref in zip(candidates, references, strict=True):
        ct, rt = cand.split(), ref.split()
        if not ct or not rt:
            scores.append(0.0)
            continue
        lcs = _lcs(ct, rt)
        prec, rec = lcs / len(ct), lcs / len(rt)
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append((1 + beta**2) * prec * rec / (rec + beta**2 * prec))
    return sum(scores) / len(scores)


def captioning_metrics(candidates: list[str], references: list[str]) -> dict[str, float]:
    out = {
        "BLEU-4": corpus_bleu(candidates, references, 4),
        "ROUGE-L": rouge_l(candidates, references),
    }
    try:  # CIDEr only if the optional dependency is present
        from pycocoevalcap.cider.cider import Cider  # noqa: F401

        out["CIDEr"] = _cider(candidates, references)
    except ImportError:
        pass  # not reported rather than fabricated
    return out


def _cider(candidates: list[str], references: list[str]) -> float:  # pragma: no cover
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [r] for i, r in enumerate(references)}
    res = {i: [c] for i, c in enumerate(candidates)}
    score, _ = Cider().compute_score(gts, res)
    return float(score)
