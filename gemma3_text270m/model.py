from __future__ import annotations

"""
Gemma3ForCausalLM model assembly (educational).

Assembles embeddings, 18 transformer blocks with a 5:1 sliding:global pattern
repeated 3 times, final RMSNorm, and a tied LM head.

Forward implements prefill-style processing over full input sequences.
"""

import torch
import torch.nn as nn

from .config import Gemma3TextConfig
from .block import Gemma3Block, RMSNorm


def _layer_types_5to1(num_layers: int) -> list[str]:
    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    out: list[str] = []
    # Sees if there are enough to add full 6-layer pattern
    while len(out) + 6 <= num_layers:
        out.extend(pattern)
    # If not divisible by 6, pad with sliding then full as needed
    while len(out) < num_layers:
        out.append("sliding_attention" if (len(out) % 6) != 5 else "full_attention")
    return out[:num_layers]


class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        D = int(config.hidden_size)
        V = int(config.vocab_size)
        L = int(config.num_hidden_layers)

        self.embed_tokens = nn.Embedding(V, D)
        layer_types = _layer_types_5to1(L)
        self.layers = nn.ModuleList([Gemma3Block(config, layer_type=lt) for lt in layer_types])
        self.norm = RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)

        # Tie weights b/w embedding and LM head... reduces params, improves sample efficiency in smaller models
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """embed --> blocks --> norm --> lm_head"""
        # input_ids: [B, T]
        x = self.embed_tokens(input_ids)  # [B, T, D]
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits


__all__ = ["Gemma3ForCausalLM"]
