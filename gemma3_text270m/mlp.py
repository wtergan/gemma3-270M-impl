from __future__ import annotations

"""
Feed-forward network (MLP) for Gemma-3-270M (educational).

Implements a SwiGLU-style gated MLP using PyTorch GELU with the tanh
approximation ("gelu_pytorch_tanh"). Dimensions follow the locked 270M spec:
  - hidden_size (D): 640
  - intermediate_size (FF): 2048

Shape (Shazeer notation):
  x: [B, T, D]
  gate = Linear(D, FF)(x)
  up   = Linear(D, FF)(x)
  act  = GELU(up, approximate='tanh')
  y    = Linear(FF, D)(gate * act)

No bias terms are used in projections for clarity and to match spec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Gemma3TextConfig


class Gemma3MLP(nn.Module):
    """
    SwiGLU-style MLP with GELU(tanh) activation.
    - gate_proj: Linear(D, FF, bias=False)
    - up_proj:   Linear(D, FF, bias=False)
    - down_proj: Linear(FF, D, bias=False)
    """

    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        d_model = int(config.hidden_size)
        d_ff = int(config.intermediate_size)

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated MLP. Returns tensor of same shape as input (B, T, D)."""
        gate = self.gate_proj(x)  # [B, T, FF]
        up = self.up_proj(x)  # [B, T, FF]
        act = F.gelu(up, approximate="tanh")  # [B, T, FF]
        hidden = gate * act  # [B, T, FF]
        out = self.down_proj(hidden)  # [B, T, D]
        return out


__all__ = ["Gemma3MLP"]
