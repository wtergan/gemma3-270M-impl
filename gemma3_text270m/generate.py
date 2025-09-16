from __future__ import annotations

"""
Text generation utilities for Gemma-3-270M (educational).

Implements a simple prefill + decode loop using the assembled model and
tokenizer. For clarity, this baseline recomputes attention over the full
context each step; KV cache integration can be added later for speed.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn.functional as F

from .config import Gemma3TextConfig
from .model import Gemma3ForCausalLM
from .tokenizer import Gemma3Tokenizer


def _sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> int:
    """Return next token id sampled from logits [V]. Greedy if temperature<=0."""
    if temperature is None or temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / float(temperature)
    probs = F.softmax(logits, dim=-1)

    # Top-k filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, probs.numel())
        vals, idx = torch.topk(probs, k)
        mask = torch.full_like(probs, float("-inf"))
        # log of the top-k probs, -inf elsewhere 
        mask[idx] = torch.log(vals)
        probs = F.softmax(mask, dim=-1)

    # Nucleus (top-p) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= top_p
        # Ensure at least one token kept
        if not torch.any(keep):
            keep[0] = True
        filtered = torch.where(
            keep, torch.log(sorted_probs), torch.full_like(sorted_probs, float("-inf"))
        )
        probs = torch.zeros_like(probs)
        probs[sorted_idx] = F.softmax(filtered, dim=-1)

    # Sample from the filtered distribution
    next_id = torch.multinomial(probs, num_samples=1)
    return int(next_id.item())


@dataclass
class Gemma3Generator:
    model: Gemma3ForCausalLM
    tokenizer: Gemma3Tokenizer
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = next(self.model.parameters()).device

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_at_eos: bool = True,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text from a prompt string.

        - temperature <= 0 switches to greedy decoding
        - top_k/top_p are optional sampling constraints
        - stream_callback, if provided, is called with decoded text chunks
        """
        self.model.eval()
        with torch.no_grad():
            # Tokenize prompt without EOS; prepend BOS
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            ids = [self.tokenizer.bos_id] + ids
            input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, T]

            # Enforce configured context length on initial prompt
            ctx_len = int(getattr(self.model.config, "context_length", 32768))
            if input_ids.shape[1] > ctx_len:
                input_ids = input_ids[:, -ctx_len:]

            for _ in range(max_new_tokens):
                # Ensure input never exceeds context_length during decode
                if input_ids.shape[1] > ctx_len:
                    input_ids = input_ids[:, -ctx_len:]
                logits = self.model(input_ids)  # [1, T, V]
                next_logits = logits[0, -1]  # [V]
                next_id = _sample_logits(
                    next_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=self.device)], dim=1)

                if stop_at_eos and next_id == self.tokenizer.eos_id:
                    break

            # Decode excluding prompt BOS
            gen_ids = input_ids[0].tolist()
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if stream_callback is not None:
                stream_callback(text)
            return text
