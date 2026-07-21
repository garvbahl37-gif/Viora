"""Q-Former — a question-conditioned resampler (BLIP-2 style).

Like :class:`PerceiverResampler`, but learnable queries are concatenated with the
(projected) **question** embeddings before self-attention, so the queries can be
*conditioned on what is being asked* while cross-attending to the visual tokens.
Only the query positions are returned. This enables question-conditioned temporal
retrieval — a deliberate step beyond an unconditional resampler.
"""

from __future__ import annotations

import torch
from torch import nn

from viora.models.common import TransformerBlock, make_norm
from viora.utils.config import ResamplerConfig


class QFormer(nn.Module):
    def __init__(
        self, cfg: ResamplerConfig, input_dim: int | None = None, text_dim: int | None = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_queries = cfg.num_queries
        self.input_dim = input_dim or cfg.dim
        self.queries = nn.Parameter(torch.zeros(1, cfg.num_queries, cfg.dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.input_proj = (
            nn.Linear(self.input_dim, cfg.dim) if self.input_dim != cfg.dim else nn.Identity()
        )
        self.text_proj = nn.Linear(text_dim, cfg.dim) if text_dim else None
        self.blocks = nn.ModuleList(
            TransformerBlock(
                cfg.dim, cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                norm_type=cfg.norm_type, use_sdpa=cfg.use_sdpa, cross_attention=True,
            )
            for _ in range(cfg.num_layers)
        )
        self.norm = make_norm(cfg.norm_type, cfg.dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_embeds: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``visual_tokens``: ``[B, M, input_dim]``; ``text_embeds``: ``[B, Lt, text_dim]``.

        Returns the ``[B, Q, D]`` query outputs.
        """
        b = visual_tokens.shape[0]
        ctx = self.input_proj(visual_tokens)
        q = self.queries.expand(b, -1, -1)

        self_kpm = None
        seq = q
        if text_embeds is not None and self.text_proj is not None:
            t = self.text_proj(text_embeds)
            seq = torch.cat([q, t], dim=1)  # [B, Q+Lt, D]
            q_mask = torch.ones(b, self.num_queries, dtype=torch.bool, device=q.device)
            tm = text_mask if text_mask is not None else torch.ones(
                b, t.shape[1], dtype=torch.bool, device=q.device
            )
            self_kpm = torch.cat([q_mask, tm], dim=1)

        for block in self.blocks:
            seq = block(
                seq, context=ctx,
                self_key_padding_mask=self_kpm, context_key_padding_mask=visual_mask,
            )
        return self.norm(seq[:, : self.num_queries])
