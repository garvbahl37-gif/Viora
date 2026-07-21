"""VioraVisionTransformer3D — the spatiotemporal video encoder.

Pipeline: tubelet embedding -> separable positional embedding -> ``depth``
pre-norm blocks (full or factorized attention) -> final norm. Exposes both the
full token sequence ``[B, N, D]`` and per-frame pooled features ``[B, T', D]``
consumed by the hierarchical temporal encoder downstream.

Implemented in-repo (not a wrapped pretrained ViT); every dimension comes from
:class:`~viora.utils.config.VisionConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange, reduce
from torch import nn

from viora.models.common import make_norm
from viora.models.embeddings.positional_embedding import SpatioTemporalPositionalEmbedding
from viora.models.embeddings.tubelet_embedding import TokenGrid, TubeletEmbedding
from viora.models.vision.blocks import SpatioTemporalBlock
from viora.utils.config import VisionConfig


@dataclass
class VisionOutput:
    tokens: torch.Tensor          # [B, N, D] contextualized spatiotemporal tokens
    grid: TokenGrid               # layout of tokens
    frame_features: torch.Tensor  # [B, T', D] spatially-pooled per-temporal-position
    temporal_mask: torch.Tensor | None  # [B, T'] True = valid (passthrough)


class VioraVisionTransformer3D(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tubelet = TubeletEmbedding(
            in_channels=cfg.in_channels,
            embed_dim=cfg.dim,
            tubelet_size=cfg.tubelet_size,
            patch_size=cfg.patch_size,
        )
        base_grid = self.tubelet.grid_for(cfg.num_frames, cfg.image_size, cfg.image_size)
        self.base_grid = base_grid

        self.pos = SpatioTemporalPositionalEmbedding(
            cfg.dim, base_grid,
            spatial=cfg.pos_embed_spatial,
            temporal=cfg.pos_embed_temporal,
            dropout=cfg.drop_rate,
        )

        # linearly-scaled stochastic depth across blocks
        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()
        self.blocks = nn.ModuleList(
            SpatioTemporalBlock(
                cfg.dim, cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                attention_mode=cfg.attention_mode,
                norm_type=cfg.norm_type,
                mlp_type=cfg.mlp_type,
                qkv_bias=cfg.qkv_bias,
                drop=cfg.drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[i],
                use_sdpa=cfg.use_sdpa,
                gradient_checkpointing=cfg.gradient_checkpointing,
            )
            for i in range(cfg.depth)
        )
        self.norm = make_norm(cfg.norm_type, cfg.dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        for blk in self.blocks:
            blk.gradient_checkpointing = enabled

    def _reconcile_timestamps(
        self, timestamps: torch.Tensor | None, t_tokens: int, batch: int
    ) -> torch.Tensor | None:
        """Map timestamps to per-temporal-token length ``T'``.

        Accepts either per-token timestamps (length ``T'``) or per-frame
        timestamps (length ``T' * tubelet_size``); the latter are mean-pooled over
        each tubelet so callers can pass raw decoded-frame timestamps directly.
        """
        if timestamps is None:
            return None
        if timestamps.dim() != 2 or timestamps.shape[0] != batch:
            raise ValueError(f"timestamps must be [B, *], got {tuple(timestamps.shape)}")
        n = timestamps.shape[1]
        if n == t_tokens:
            return timestamps
        if n == t_tokens * self.cfg.tubelet_size:
            return timestamps.reshape(batch, t_tokens, self.cfg.tubelet_size).mean(dim=2)
        raise ValueError(
            f"timestamps length {n} must be T'={t_tokens} or "
            f"T={t_tokens * self.cfg.tubelet_size} (frames)"
        )

    def forward(
        self,
        video: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> VisionOutput:
        """Args:
            video: ``[B, C, T, H, W]``.
            temporal_mask: ``[B, T']`` bool over *token* temporal positions
                (``T' = T / tubelet_size``); ``True`` = valid.
            timestamps: optional ``[B, T']`` seconds for timestamp-aware temporal
                positional embedding.

        Returns:
            :class:`VisionOutput`.
        """
        tokens, grid = self.tubelet(video)  # [B, N, D]
        timestamps = self._reconcile_timestamps(timestamps, grid.t, video.shape[0])
        tokens = self.pos(tokens, grid, timestamps=timestamps)
        for blk in self.blocks:
            tokens = blk(tokens, grid, temporal_mask=temporal_mask)
        tokens = self.norm(tokens)

        # per-temporal-position features = mean over spatial tokens
        frame_features = reduce(
            rearrange(tokens, "b (t s) d -> b t s d", t=grid.t),
            "b t s d -> b t d", "mean",
        )
        return VisionOutput(tokens=tokens, grid=grid, frame_features=frame_features,
                            temporal_mask=temporal_mask)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
