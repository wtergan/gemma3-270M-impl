from __future__ import annotations

"""
Transformer block for Gemma-3-270M (educational).

Composition:
- RMSNorm (pre-attention)
- Self-Attention (Sliding or Global)
- Residual add
- RMSNorm (pre-MLP)
- MLP (SwiGLU-style)
- Residual add

Forward preserves shape [B, T, D]. This module implements the common pre-norm
pattern used in modern transformers. Decode-mode KV integration is deferred to
the model assembly for simplicity in this educational scaffold.
"""

import torch
import torch.nn as nn

from .config import Gemma3TextConfig
from .attention import SlidingWindowAttention, GlobalAttention
from .mlp import Gemma3MLP


class RMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm variant.
    y = x / rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight


class Gemma3Block(nn.Module):
    """Transformer block mirroring Gemma3's norm ordering."""

    def __init__(self, config: Gemma3TextConfig, layer_type: str = "sliding_attention"):
        super().__init__()
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError("layer_type must be 'sliding_attention' or 'full_attention'")

        d_model = int(config.hidden_size)
        self.input_layernorm = RMSNorm(d_model)
        if layer_type == "sliding_attention":
            self.attention = SlidingWindowAttention(config)
        else:
            self.attention = GlobalAttention(config)
        self.post_attention_layernorm = RMSNorm(d_model)
        self.pre_feedforward_layernorm = RMSNorm(d_model)
        self.post_feedforward_layernorm = RMSNorm(d_model)
        self.mlp = Gemma3MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        attn_input = self.input_layernorm(x)
        attn_out = self.attention(attn_input, attention_mask=attention_mask)
        attn_out = self.post_attention_layernorm(attn_out)
        x = residual + attn_out

        residual = x
        mlp_input = self.pre_feedforward_layernorm(x)
        mlp_out = self.mlp(mlp_input)
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        x = residual + mlp_out
        return x


__all__ = ["Gemma3Block", "RMSNorm"]
