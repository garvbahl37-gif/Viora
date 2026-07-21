"""Video-text matching loss (binary: is this (video, text) pair a true match?).

Complements the contrastive loss with a fine-grained pairwise decision, typically
trained with in-batch hard negatives. The loss is a plain cross-entropy over
2-class matching logits; :func:`sample_hard_negatives` builds negative pairs from
a similarity matrix.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VideoTextMatchingLoss(nn.Module):
    def forward(self, match_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """``match_logits``: ``[N, 2]``; ``labels``: ``[N]`` in {0,1}."""
        return F.cross_entropy(match_logits, labels)


@torch.no_grad()
def sample_hard_negatives(similarity: torch.Tensor) -> torch.Tensor:
    """Given ``[B, B]`` similarities, return, per row, the hardest negative index.

    The diagonal (true pair) is masked out; the most similar off-diagonal entry is
    the hardest negative.
    """
    sim = similarity.clone()
    sim.fill_diagonal_(float("-inf"))
    return sim.argmax(dim=1)
