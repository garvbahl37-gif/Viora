"""Multimodal projector: map resampled visual tokens into the LLM embedding space.

The vision stack works in ``vision_dim``; the LLM expects ``llm_hidden``. This is
the only learned bridge between the two spaces, so it is kept simple and explicit
(``linear`` or ``mlp``) and its output dim is resolved from the LLM at build time.
"""

from __future__ import annotations

import torch
from torch import nn

from viora.utils.config import ProjectorConfig

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class MultimodalProjector(nn.Module):
    def __init__(self, cfg: ProjectorConfig, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be resolved (> 0) from the LLM hidden size")
        self.input_dim = input_dim
        self.output_dim = output_dim

        if cfg.type == "linear":
            self.proj: nn.Module = nn.Linear(input_dim, output_dim)
        elif cfg.type == "mlp":
            hidden = cfg.hidden_dim or output_dim
            act = _ACT.get(cfg.activation, nn.GELU)
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden), act()]
            for _ in range(max(0, cfg.depth - 2)):
                layers += [nn.Linear(hidden, hidden), act()]
            layers.append(nn.Linear(hidden, output_dim))
            self.proj = nn.Sequential(*layers)
        else:
            raise ValueError(f"unknown projector type '{cfg.type}' (expected linear|mlp)")

        # Normalize + scale the visual tokens to the LLM's embedding scale. Without this,
        # random-init projector outputs are ~10x larger than a (frozen) LLM's token
        # embeddings, causing exploding gradients that grad-clipping crushes -> no learning.
        # The scale is learnable; set match_embedding_scale() to the LLM embed std at build.
        self.out_norm = nn.LayerNorm(output_dim)

    def match_embedding_scale(self, std: float) -> None:
        """Init the output scale so visual tokens start at the LLM's embedding std."""
        with torch.no_grad():
            self.out_norm.weight.fill_(float(std))
            self.out_norm.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, Q, input_dim] -> [B, Q, output_dim]`` at the LLM's embedding scale."""
        return self.out_norm(self.proj(x))
