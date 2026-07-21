"""Pretrained per-frame vision encoder (the pragmatic vision path).

Instead of pretraining Viora's 3D ViT from scratch (which needs TBs of video +
GPUs), initialise vision from a pretrained image encoder (SigLIP / CLIP): run each
frame through the frozen backbone, project to Viora's dim, and feed the per-frame
features into the hierarchical temporal encoder / event tokenizer / resampler
exactly as the 3D ViT's ``frame_features`` would. This is how shipping video-LMs
(LLaVA/Video-LLaVA/Qwen-VL-style) get to quality without vision pretraining.

Returns a :class:`~viora.models.vision.vit3d.VisionOutput`-compatible object so it
is a drop-in alternative to the 3D ViT downstream. Requires ``transformers``; the
video transform must match the backbone's expected size + normalization (SigLIP:
224px, [-1,1]).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn

from viora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FrameEncoderOutput:
    frame_features: torch.Tensor          # [B, T, D]
    tokens: torch.Tensor | None = None    # patch tokens if kept (else None)
    grid: object | None = None
    temporal_mask: torch.Tensor | None = None


class PretrainedFrameEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_dim: int,
        *,
        freeze: bool = True,
        pool: str = "mean",     # "mean" over patches | "cls" pooler
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover
            raise ImportError("pretrained vision needs transformers; `uv pip install transformers`") from e

        td = getattr(torch, dtype, torch.float32)
        try:
            full = AutoModel.from_pretrained(model_name, dtype=td)
        except TypeError:  # pragma: no cover - older transformers
            full = AutoModel.from_pretrained(model_name, torch_dtype=td)
        self.vision = getattr(full, "vision_model", full)
        self.hidden = getattr(self.vision.config, "hidden_size", None) or full.config.hidden_size
        self.pool = pool
        self.proj = nn.Linear(self.hidden, out_dim)
        self.frozen = freeze
        if freeze:
            for p in self.vision.parameters():
                p.requires_grad_(False)
            self.vision.eval()
        logger.info("PretrainedFrameEncoder: %s (hidden=%d, frozen=%s) -> %d",
                    model_name, self.hidden, freeze, out_dim)

    def train(self, mode: bool = True):  # keep frozen backbone in eval
        super().train(mode)
        if self.frozen:
            self.vision.eval()
        return self

    def forward(
        self, video: torch.Tensor, temporal_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,  # noqa: ARG002 - kept for interface parity
    ) -> FrameEncoderOutput:
        """``video``: ``[B, C, T, H, W]`` (preprocessed for the backbone) -> per-frame features."""
        b, c, t, h, w = video.shape
        frames = rearrange(video, "b c t h w -> (b t) c h w")
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.vision(pixel_values=frames)
        if self.pool == "cls" and getattr(out, "pooler_output", None) is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state.mean(dim=1)  # mean over patches
        feat = self.proj(feat.to(self.proj.weight.dtype))
        frame_features = rearrange(feat, "(b t) d -> b t d", b=b)
        return FrameEncoderOutput(frame_features=frame_features, temporal_mask=temporal_mask)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
