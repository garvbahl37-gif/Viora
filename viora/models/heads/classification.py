"""Classification head for action recognition (pooled video feature -> class logits)."""

from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """``[B, D] -> [B, num_classes]``."""
        return self.fc(self.drop(pooled))
