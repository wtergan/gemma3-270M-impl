"""
gemma3_text270m
-----------------
Educational, from-scratch implementation scaffold for the Gemma-3-270M text model.

This package provides a clean, modular codebase that will progressively implement
the core components needed to understand and run a compact LLM:
- Configuration system (Gemma3TextConfig)
- Tokenizer integration (SentencePiece / HF tokenizers)
- Attention (MQA with sliding/global variants and RoPE)
- Feed-forward network (SwiGLU-style)
- KV cache utilities
- Model assembly (Gemma3ForCausalLM)
- Utilities for text generation and HF weight loading (no Transformers dependency)

The initial version only establishes the package structure and metadata.
"""

from .config import Gemma3TextConfig
from .tokenizer import Gemma3Tokenizer
from .attention import SlidingWindowAttention, GlobalAttention
from .mlp import Gemma3MLP
from .kvcache import KVCache, RingBufferKVCache, FullSequenceKVCache
from .block import Gemma3Block
from .model import Gemma3ForCausalLM
from .hf_loader import load_weights_into
from .generate import Gemma3Generator

__all__ = [
    "Gemma3TextConfig",
    "Gemma3Tokenizer",
    "SlidingWindowAttention",
    "GlobalAttention",
    "Gemma3MLP",
    "KVCache",
    "RingBufferKVCache",
    "FullSequenceKVCache",
    "Gemma3Block",
    "Gemma3ForCausalLM",
    "load_weights_into",
    "Gemma3Generator",
]

# Package version
__version__ = "0.1.0"
