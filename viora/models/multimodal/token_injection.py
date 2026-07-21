"""Embedding-level injection of visual tokens into a language sequence.

Given a text sequence containing a single ``<video>`` placeholder token, replace
that placeholder with the ``Q`` projected visual embeddings (LLaVA-style
expansion). We operate on ``inputs_embeds`` — never by stringifying embeddings —
and rebuild the attention mask (visual tokens are always attended) and labels
(visual positions are set to ``ignore_index`` so they contribute no LM loss).

If a sequence has no placeholder, visual tokens are prepended. Sequences are
right-padded to a common length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MultimodalBatch:
    inputs_embeds: torch.Tensor       # [B, L', D]
    attention_mask: torch.Tensor      # [B, L'] (1 = attend)
    labels: torch.Tensor | None       # [B, L'] (ignore_index on visual/pad)


def build_multimodal_inputs(
    input_ids: torch.Tensor,             # [B, L]
    visual_embeds: torch.Tensor,         # [B, Q, D]
    embed_layer: nn.Module,              # token -> embedding
    *,
    video_token_id: int,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    ignore_index: int = -100,
    pad_token_id: int = 0,
) -> MultimodalBatch:
    """Splice ``visual_embeds`` into each sequence at its ``<video>`` placeholder."""
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
    b, length = input_ids.shape
    q = visual_embeds.shape[1]
    device = visual_embeds.device
    text_embeds = embed_layer(input_ids)  # [B, L, D]
    if attention_mask is None:
        attention_mask = torch.ones(b, length, dtype=torch.long, device=device)

    seqs, masks, lbls = [], [], []
    for i in range(b):
        placeholders = (input_ids[i] == video_token_id).nonzero(as_tuple=True)[0]
        p = int(placeholders[0]) if placeholders.numel() > 0 else 0
        keep_placeholder = placeholders.numel() == 0  # no placeholder -> prepend, keep all text

        before = slice(0, p)
        after = slice(p if keep_placeholder else p + 1, length)

        emb = torch.cat([text_embeds[i, before], visual_embeds[i], text_embeds[i, after]], dim=0)
        vis_mask = torch.ones(q, dtype=attention_mask.dtype, device=device)
        msk = torch.cat([attention_mask[i, before], vis_mask, attention_mask[i, after]], dim=0)
        seqs.append(emb)
        masks.append(msk)
        if labels is not None:
            vis_lbl = torch.full((q,), ignore_index, dtype=labels.dtype, device=device)
            lbls.append(torch.cat([labels[i, before], vis_lbl, labels[i, after]], dim=0))

    max_len = max(s.shape[0] for s in seqs)
    dim = visual_embeds.shape[2]
    pad_vec = embed_layer(torch.full((1,), pad_token_id, dtype=torch.long, device=device))[0]

    out_emb = torch.zeros(b, max_len, dim, device=device, dtype=visual_embeds.dtype)
    out_mask = torch.zeros(b, max_len, dtype=attention_mask.dtype, device=device)
    out_lbl = (
        torch.full((b, max_len), ignore_index, dtype=labels.dtype, device=device)
        if labels is not None
        else None
    )
    for i, (emb, msk) in enumerate(zip(seqs, masks, strict=True)):
        n = emb.shape[0]
        out_emb[i, :n] = emb
        out_emb[i, n:] = pad_vec
        out_mask[i, :n] = msk
        if out_lbl is not None:
            out_lbl[i, :n] = lbls[i]

    return MultimodalBatch(inputs_embeds=out_emb, attention_mask=out_mask, labels=out_lbl)
