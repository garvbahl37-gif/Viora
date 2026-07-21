"""Attention primitives shared by every attention variant in Viora.

A single :class:`MultiHeadAttention` handles both self- and cross-attention and
both the fused (``scaled_dot_product_attention``) and a numerically-equivalent
manual path, selected by ``use_sdpa``. Mask convention throughout Viora:
``key_padding_mask[B, Nk]`` with ``True`` = **valid** (kept); we convert to SDPA's
boolean-attend semantics at this boundary so no other module handles masks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional cross-attention and key padding.

    Args:
        dim: query/output dimension.
        num_heads: attention heads (``dim`` must be divisible by it).
        kv_dim: key/value input dimension for cross-attention (defaults to ``dim``).
        qkv_bias: bias on the q/k/v projections.
        attn_drop / proj_drop: dropout probabilities.
        use_sdpa: use the fused kernel when True, else the manual path.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        kv_dim: int | None = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_sdpa = use_sdpa
        kv_dim = kv_dim or dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        return rearrange(t, "b n (h d) -> b h n d", h=self.num_heads)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: queries ``[B, Nq, D]``.
            context: keys/values ``[B, Nk, kv_dim]``; ``None`` -> self-attention.
            key_padding_mask: ``[B, Nk]`` bool, ``True`` = valid.
            attn_mask: boolean ``[Nq, Nk]`` / ``[B, Nq, Nk]`` (``True`` = attend),
                e.g. a local-window band; combined (AND) with ``key_padding_mask``.
            return_weights: if True, also return head-averaged attention weights
                ``[B, Nq, Nk]`` (uses the manual path; for interpretability).

        Returns:
            ``[B, Nq, D]``, or ``(out, weights)`` when ``return_weights``.
        """
        ctx = x if context is None else context
        q = self._split(self.q_proj(x))  # [B, H, Nq, d]
        k = self._split(self.k_proj(ctx))  # [B, H, Nk, d]
        v = self._split(self.v_proj(ctx))  # [B, H, Nk, d]

        mask = None  # broadcastable to [B, H, Nq, Nk], True = attend
        if attn_mask is not None:
            m = attn_mask.to(torch.bool)
            m = m[None, None] if m.dim() == 2 else m[:, None]  # -> [*,1,Nq,Nk]
            mask = m
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :].to(torch.bool)  # [B,1,1,Nk]
            mask = kpm if mask is None else (mask & kpm)

        if self.use_sdpa and not return_weights:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop if self.training else 0.0,
            )
            weights = None
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,Nq,Nk]
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            probs = scores.softmax(dim=-1)
            # guard fully-masked rows (all -inf -> NaN after softmax)
            probs = torch.nan_to_num(probs)
            if self.attn_drop and self.training:
                probs = F.dropout(probs, p=self.attn_drop)
            out = torch.matmul(probs, v)  # [B,H,Nq,d]
            weights = probs.mean(dim=1)  # [B, Nq, Nk] head-averaged

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj_drop(self.proj(out))
        if return_weights:
            return out, weights
        return out
