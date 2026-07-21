"""VioraForVideoUnderstanding — the unified video-language model.

Forward pipeline (shapes in brackets)::

    video [B,C,T,H,W]
      -> VioraVisionTransformer3D            tokens [B,N,D], frame_features [B,T',D]
      -> HierarchicalTemporalEncoder         [B,T',D]
      -> EventTokenizer                      event_tokens [B,E,D] (+attention)
      -> concat(temporal, events)            [B,T'+E,D]
      -> PerceiverResampler                  [B,Q,D]
      -> MultimodalProjector                 [B,Q,D_llm]
      -> inject at <video> + LLMAdapter      language-model loss
      -> TemporalGroundingHead               start/end + confidence

Every path returns the same :class:`VioraOutput`. Losses are combined by
:class:`MultiTaskLossManager` and reported per-component. Temporal memory is used
by the streaming/inference paths (M9), not this single-clip training forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from viora.losses.contrastive import VideoTextContrastiveLoss
from viora.losses.multitask import MultiTaskLossManager
from viora.losses.temporal_grounding import TemporalGroundingLoss
from viora.models.heads.confidence import ConfidenceHead
from viora.models.heads.retrieval import RetrievalHead
from viora.models.heads.temporal_grounding import GroundingOutput, TemporalGroundingHead
from viora.models.language.llm_adapter import LLMAdapter
from viora.models.multimodal.projector import MultimodalProjector
from viora.models.multimodal.resampler import PerceiverResampler
from viora.models.multimodal.token_injection import build_multimodal_inputs
from viora.models.temporal.event_tokenizer import EventTokenizer
from viora.models.temporal.hierarchical_encoder import HierarchicalTemporalEncoder
from viora.models.temporal.temporal_memory import TemporalMemory
from viora.models.temporal.temporal_pooling import SpatialPool, masked_mean
from viora.models.vision.vit3d import VioraVisionTransformer3D
from viora.utils.config import VioraConfig
from viora.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_LOSS_WEIGHTS = {"lm": 1.0, "grounding": 1.0, "contrastive": 1.0, "order": 0.5}


@dataclass
class VioraOutput:
    loss: torch.Tensor | None = None
    losses: dict[str, float] = field(default_factory=dict)
    logits: torch.Tensor | None = None
    temporal_predictions: GroundingOutput | None = None
    event_attention: torch.Tensor | None = None
    video_embed: torch.Tensor | None = None
    text_embed: torch.Tensor | None = None
    resampled_tokens: torch.Tensor | None = None
    injected_labels: torch.Tensor | None = None  # labels after visual-token expansion
    diagnostics: dict = field(default_factory=dict)


class VioraForVideoUnderstanding(nn.Module):
    def __init__(self, cfg: VioraConfig, loss_weights: dict[str, float] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.vision.dim
        if not (cfg.temporal.dim == cfg.event.dim == cfg.resampler.dim == d):
            raise ValueError("vision/temporal/event/resampler dims must match for token fusion")

        if cfg.vision.backbone == "pretrained":
            from viora.models.vision.pretrained import PretrainedFrameEncoder

            self.vision = PretrainedFrameEncoder(
                cfg.vision.pretrained_name, d, freeze=cfg.vision.freeze_backbone,
                pool=cfg.vision.pretrained_pool,
            )
            self.spatial_pool = None  # per-frame features already; no spatial tokens
        else:
            self.vision = VioraVisionTransformer3D(cfg.vision)
            self.spatial_pool = (
                None if cfg.temporal.spatial_pool == "mean"
                else SpatialPool(d, cfg.temporal.spatial_pool, num_heads=cfg.temporal.num_heads)
            )
        self.temporal = HierarchicalTemporalEncoder(cfg.temporal)
        self.event = EventTokenizer(cfg.event)
        self.memory = TemporalMemory(cfg.memory) if cfg.memory.enabled else None
        self.resampler = PerceiverResampler(cfg.resampler, input_dim=d)
        self.llm = LLMAdapter(cfg.llm)

        out_dim = cfg.projector.output_dim or self.llm.hidden_size
        self.projector = MultimodalProjector(cfg.projector, input_dim=cfg.resampler.dim, output_dim=out_dim)
        # scale-match the visual tokens to the LLM's embedding space (stabilizes training)
        with torch.no_grad():
            embed_std = float(self.llm.get_input_embeddings().weight.std())
        self.projector.match_embedding_scale(embed_std if embed_std > 0 else 0.02)

        self.grounding = (
            TemporalGroundingHead(cfg.grounding, cfg.temporal.dim, query_dim=cfg.temporal.dim)
            if cfg.grounding.enabled else None
        )
        self.retrieval = RetrievalHead(d, self.llm.hidden_size)
        self.confidence = ConfidenceHead(d)

        # loss modules / manager
        self.contrastive = VideoTextContrastiveLoss()
        self.grounding_loss = TemporalGroundingLoss()
        self.loss_manager = MultiTaskLossManager(loss_weights or DEFAULT_LOSS_WEIGHTS)

    # -------------------------------------------------------------- encoding
    def encode_video(
        self,
        video: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the vision+temporal+event+resampler stack; returns intermediate tensors."""
        vout = self.vision(video, temporal_mask=temporal_mask, timestamps=timestamps)
        frame_feats = (
            self.spatial_pool(vout.tokens, vout.grid) if self.spatial_pool is not None
            else vout.frame_features
        )  # [B, T', D]
        temporal = self.temporal(frame_feats, temporal_mask=temporal_mask)  # [B, T', D]
        events = self.event(temporal, temporal_mask=temporal_mask)
        fused = torch.cat([temporal, events.event_tokens], dim=1)  # [B, T'+E, D]
        resampled = self.resampler(fused)  # [B, Q, D]
        return {
            "temporal": temporal,
            "event_tokens": events.event_tokens,
            "event_attention": events.event_attention,
            "resampled": resampled,
        }

    # --------------------------------------------------------------- forward
    def forward(
        self,
        video: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        grounding_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
        compute_contrastive: bool = False,
    ) -> VioraOutput:
        enc = self.encode_video(video, temporal_mask, timestamps)
        temporal, resampled = enc["temporal"], enc["resampled"]
        projected = self.projector(resampled)  # [B, Q, D_llm]

        losses: dict[str, torch.Tensor | None] = {}
        logits = None
        injected_labels = None

        # language modeling (VideoQA / captioning / instruction following)
        if input_ids is not None:
            mm = build_multimodal_inputs(
                input_ids, projected, self.llm.get_input_embeddings(),
                video_token_id=self.llm.video_token_id,
                attention_mask=attention_mask, labels=labels,
            )
            lm_out = self.llm(mm.inputs_embeds, mm.attention_mask, mm.labels)
            logits = lm_out.logits
            injected_labels = mm.labels
            if mm.labels is not None:
                losses["lm"] = lm_out.loss

        # temporal grounding (evidence)
        grounding_out = None
        if self.grounding is not None:
            query = masked_mean(temporal, temporal_mask)  # [B, D]
            grounding_out = self.grounding(temporal, query)
            if grounding_targets is not None:
                sb, eb = grounding_targets
                losses["grounding"] = self.grounding_loss(
                    grounding_out.start_logits, grounding_out.end_logits, sb, eb
                )

        # video-text alignment (optional)
        video_embed = text_embed = None
        if compute_contrastive and input_ids is not None:
            vid_summary = resampled.mean(dim=1)  # [B, D]
            txt_summary = self.llm.get_input_embeddings()(input_ids).mean(dim=1)  # [B, D_llm]
            video_embed, text_embed = self.retrieval(vid_summary, txt_summary)
            losses["contrastive"] = self.contrastive(video_embed, text_embed)

        breakdown = self.loss_manager.combine(losses)
        total = breakdown.total if losses else None

        return VioraOutput(
            loss=total,
            losses=breakdown.components,
            logits=logits,
            temporal_predictions=grounding_out,
            event_attention=enc["event_attention"],
            video_embed=video_embed,
            text_embed=text_embed,
            resampled_tokens=resampled,
            injected_labels=injected_labels,
            diagnostics={"nonfinite_losses": breakdown.nonfinite, "loss_weights": breakdown.weights},
        )

    # ------------------------------------------------------------ bookkeeping
    def count_parameters(self) -> dict[str, int]:
        """Report parameter counts by subsystem (total & trainable)."""
        def n(module: nn.Module | None, trainable: bool = False) -> int:
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable)

        heads = nn.ModuleList([m for m in (self.grounding, self.retrieval, self.confidence) if m])
        return {
            "total": n(self),
            "trainable": n(self, trainable=True),
            "vision": n(self.vision),
            "temporal": n(self.temporal) + n(self.event) + n(self.memory),
            "multimodal": n(self.resampler) + n(self.projector),
            "llm": n(self.llm),
            "heads": n(heads),
        }
