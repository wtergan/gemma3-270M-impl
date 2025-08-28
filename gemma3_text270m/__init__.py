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

__all__ = [
    "Gemma3TextConfig",
]

# Package version
__version__ = "0.1.0"
