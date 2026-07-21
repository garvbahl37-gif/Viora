"""Shared neural building blocks used across Viora's encoders.

Kept dependency-free (only ``torch``) so any subpackage can import it without
creating cycles: norm factory, RMSNorm, stochastic depth, and the two MLP
variants (standard GELU-MLP and SwiGLU).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def make_norm(kind: str, dim: int) -> nn.Module:
    """Return a normalization layer by name: ``"layernorm"`` or ``"rmsnorm"``."""
    kind = kind.lower()
    if kind in ("layernorm", "ln"):
        return nn.LayerNorm(dim)
    if kind in ("rmsnorm", "rms"):
        return RMSNorm(dim)
    raise ValueError(f"unknown norm '{kind}' (expected layernorm|rmsnorm)")


def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth (drops whole residual branches)."""
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    # broadcast mask over all non-batch dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x * mask / keep


class DropPath(nn.Module):
    """Module wrapper around :func:`drop_path`."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class MLP(nn.Module):
    """Standard transformer feed-forward: Linear -> act -> drop -> Linear."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        *,
        act: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward (GLU with SiLU gate); hidden scaled by 2/3 per convention."""

    def __init__(self, dim: int, hidden_dim: int | None = None, *, drop: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


def build_mlp(kind: str, dim: int, mlp_ratio: float, drop: float = 0.0) -> nn.Module:
    """Feed-forward factory: ``"mlp"`` or ``"swiglu"``."""
    hidden = int(dim * mlp_ratio)
    if kind == "mlp":
        return MLP(dim, hidden, drop=drop)
    if kind == "swiglu":
        return SwiGLU(dim, hidden, drop=drop)
    raise ValueError(f"unknown mlp type '{kind}' (expected mlp|swiglu)")


class TransformerBlock(nn.Module):
    """Generic pre-norm transformer block over a token sequence ``[B, N, D]``.

    Optionally cross-attends to a ``context`` sequence (queries=``x``, keys/values
    =``context``) between self-attention and the feed-forward — the shape used by
    the event tokenizer and the resampler. Import is local to avoid a cycle
    (``attention_utils`` depends on nothing internal).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        norm_type: str = "layernorm",
        mlp_type: str = "mlp",
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_sdpa: bool = True,
        cross_attention: bool = False,
        kv_dim: int | None = None,
    ) -> None:
        super().__init__()
        from viora.models.attention.attention_utils import MultiHeadAttention

        self.norm1 = make_norm(norm_type, dim)
        self.self_attn = MultiHeadAttention(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=drop, use_sdpa=use_sdpa,
        )
        self.cross_attn = None
        if cross_attention:
            self.norm_ctx = make_norm(norm_type, dim)
            self.cross_attn = MultiHeadAttention(
                dim, num_heads, kv_dim=kv_dim, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop, use_sdpa=use_sdpa,
            )
        self.norm2 = make_norm(norm_type, dim)
        self.mlp = build_mlp(mlp_type, dim, mlp_ratio, drop=drop)
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        context_key_padding_mask: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        return_cross_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        x = x + self.drop_path(
            self.self_attn(
                self.norm1(x),
                key_padding_mask=self_key_padding_mask,
                attn_mask=self_attn_mask,
            )
        )
        weights = None
        if self.cross_attn is not None and context is not None:
            attn_out = self.cross_attn(
                self.norm_ctx(x), context=context,
                key_padding_mask=context_key_padding_mask,
                return_weights=return_cross_weights,
            )
            if return_cross_weights:
                attn_out, weights = attn_out
            x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_cross_weights:
            return x, weights
        return x
