from __future__ import annotations

"""
Attention modules for Gemma-3-270M (educational).

Implements Multi-Query Attention (MQA) with:
- Global (full) causal attention
- Sliding window causal attention
- RoPE (rotary position embeddings) applied to Q/K
- QK scaling via `query_pre_attn_scalar`

Shazeer notation used for clarity (B=batch, T=seq_len, H=query heads, K=kv heads, D=head_dim).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Gemma3TextConfig


def _build_causal_mask(t_q: int, t_k: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular mask of shape [1, 1, T_q, T_k] with -inf above diagonal."""
    i = torch.arange(t_q, device=device)[:, None]
    j = torch.arange(t_k, device=device)[None, :]
    mask = (i < j).to(torch.bool)  # True where masked (future positions)
    out = torch.zeros((1, 1, t_q, t_k), device=device)
    out.masked_fill_(mask, float("-inf"))
    return out


def _apply_sliding_window(mask: torch.Tensor, window: int) -> torch.Tensor:
    """Augment a causal mask to also mask tokens older than `window`.
    `mask` is additive [1,1,T,T] with -inf for disallowed positions.
    """
    b, h, t_q, t_k = mask.shape
    assert b == 1 and h == 1 and t_q == t_k
    device = mask.device
    i = torch.arange(t_q, device=device)[:, None]
    j = torch.arange(t_k, device=device)[None, :]
    too_old = (i - j) > window
    mask = mask.masked_fill(too_old, float("-inf"))
    return mask


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits input tensor in half along the last dim, then concats the negated second half with the first half.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _build_rope_cache(
    seq_len: int, dim: int, base_theta: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (cos, sin) with shape [1, 1, seq_len, dim].
    Uses pairwise dims (half) for rotary, then duplicates to full dim for efficiency.
    """
    assert dim % 2 == 0, "Embedding dim has to be even for RoPE computation...."
    half = dim // 2
    inv_freq = 1.0 / (
        base_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("p,d->pd", pos, inv_freq)  # [seq_len, half]
    cos = freqs.cos().repeat_interleave(2, dim=-1)  # [seq_len, dim]
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B,H,T,D] or [B,K,T,D]; cos/sin: [1,1,T,D]
    return (x * cos) + (_rope_rotate_half(x) * sin)


@dataclass
class _MQAParams:
    num_heads: int
    num_kv: int
    head_dim: int
    hidden_size: int
    query_pre_attn_scalar: float
    sliding_window: Optional[int]


class _MultiQueryAttentionBase(nn.Module):
    """Base MQA with RoPE and causal/sliding masks; projects back to hidden_size."""

    def __init__(self, config: Gemma3TextConfig, sliding_window: Optional[int], *, rope_theta: float):
        super().__init__()
        self.params = _MQAParams(
            num_heads=config.num_attention_heads,
            num_kv=config.num_key_value_heads,
            head_dim=config.head_dim,
            hidden_size=config.hidden_size,
            query_pre_attn_scalar=config.head_dim**-0.5,
            sliding_window=sliding_window,
        )
        H, K, D, HS = (
            self.params.num_heads,
            self.params.num_kv,
            self.params.head_dim,
            self.params.hidden_size,
        )

        # Projections for Q, K, V, O respectively... no biases.
        self.q_proj = nn.Linear(HS, H * D, bias=False)
        self.k_proj = nn.Linear(HS, K * D, bias=False)
        self.v_proj = nn.Linear(HS, K * D, bias=False)
        self.o_proj = nn.Linear(H * D, HS, bias=False)

        # Select RoPE base per attention type (passed by subclass)
        self.rope_theta = float(rope_theta)

    # Friendly aliases for tests/printing...
    @property
    def num_heads(self) -> int:
        return self.params.num_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.params.num_kv

    @property
    def head_dim(self) -> int:
        return self.params.head_dim

    def _make_mask(self, t_q: int, t_k: int, device: torch.device) -> torch.Tensor:
        """The appropriate causal/sliding window mask for given lengths."""
        mask = _build_causal_mask(t_q, t_k, device)
        if self.params.sliding_window is not None:
            win = int(self.params.sliding_window)
            mask = _apply_sliding_window(mask, win)
        return mask

    def forward(
        self,
        x: torch.Tensor,  # [B,T,HS]
        attention_mask: Optional[torch.Tensor] = None,  # additive [B,1,1,T]
    ) -> torch.Tensor:
        B, T, HS = x.shape
        H, K, D = self.params.num_heads, self.params.num_kv, self.params.head_dim

        # Projection applications on x...
        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, K, D)
        v = self.v_proj(x).view(B, T, K, D)

        # RoPE for Q/K
        device = x.device
        cos, sin = _build_rope_cache(T, D, self.rope_theta, device)
        # [B,H,T,D] ; [B,K,T,D]
        q = _apply_rope(q.permute(0, 2, 1, 3).contiguous(), cos, sin)
        k = _apply_rope(k.permute(0, 2, 1, 3).contiguous(), cos, sin)  # treat K as heads dim=K

        # Scaled dot-product with MQA broadcasting (K/V shared across heads)
        scale = self.params.query_pre_attn_scalar
        # q: [B,H,T,D], k: [B,K,T,D] → kT: [B,K,D,T]; broadcast K=1 over H
        attn_scores = torch.matmul(q * scale, k.transpose(-2, -1))  # [B,H,T,T] (since K=1)

        # Masks
        mask = self._make_mask(T, T, device)  # [1,1,T,T]
        attn_scores = attn_scores + mask  # broadcast over B,H
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask  # [B,1,1,T]

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

        # Values: v [B,K,T,D] → broadcast to [B,H,T,D]
        v = v.permute(0, 2, 1, 3).contiguous()  # [B,K,T,D]
        # Expand along heads (K=1 → repeat to H)
        if K == 1:
            v = v.expand(B, H, T, D)
        else:
            # If future variants use grouped KV, tile accordingly
            v = v.repeat_interleave(H // K, dim=1)

        ctx = torch.matmul(attn_weights, v)  # [B,H,T,D]
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)  # [B,T,H*D]
        out = self.o_proj(ctx)  # [B,T,HS]
        return out


class GlobalAttention(_MultiQueryAttentionBase):
    """Full causal attention (no sliding window)."""

    def __init__(self, config: Gemma3TextConfig):
        # Global attention uses the larger RoPE base
        super().__init__(config, sliding_window=None, rope_theta=config.rope_theta)


class SlidingWindowAttention(_MultiQueryAttentionBase):
    """Sliding window causal attention with window size from config or arg."""

    def __init__(self, config: Gemma3TextConfig, window_size: Optional[int] = None):
        if window_size is None:
            window_size = int(config.sliding_window)
        # Sliding/local attention uses the local RoPE base (10k)
        super().__init__(config, sliding_window=window_size, rope_theta=config.rope_local_theta)


__all__ = [
    "GlobalAttention",
    "SlidingWindowAttention",
]
