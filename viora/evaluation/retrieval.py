"""Video-text retrieval metrics: Recall@K from a similarity matrix."""

from __future__ import annotations

import torch


def recall_at_k(similarity: torch.Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    """``similarity`` ``[N, N]`` (row = query, col = candidate); diagonal is the match.

    Returns Recall@K and the median rank of the correct candidate.
    """
    n = similarity.shape[0]
    # rank of the true (diagonal) match per row, 0-indexed
    order = similarity.argsort(dim=1, descending=True)
    correct = torch.arange(n, device=similarity.device).unsqueeze(1)
    ranks = (order == correct).float().argmax(dim=1)  # [N]
    out = {f"R@{k}": float((ranks < k).float().mean()) for k in ks if k <= n}
    out["median_rank"] = float(ranks.median()) + 1
    return out
