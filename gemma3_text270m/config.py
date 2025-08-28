from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
import json
from pathlib import Path


@dataclass
class Gemma3TextConfig:
    """Configuration for the Gemma-3-270M text model.

    This configuration is intentionally locked to the 270M variant. Loading a
    mismatched configuration will raise a ValueError.

    HuggingFace compatibility: can be constructed directly from a HF-style
    config.json dictionary via `from_hf_dict` or from a file via `from_json_file`.
    """

    # Locked architecture parameters for Gemma-3-270M
    vocab_size: int = 262_144
    hidden_size: int = 640
    intermediate_size: int = 2_048
    num_hidden_layers: int = 18
    num_attention_heads: int = 4
    num_key_value_heads: int = 1  # MQA
    head_dim: int = 256

    # Common tokens (Gemma-3 convention)
    bos_token_id: int = 2
    eos_token_id: int = 1
    pad_token_id: int = 0

    # Attention/windowing defaults (used later by attention/KV cache)
    sliding_window: int = 4096

    # Rotary positional embeddings base (placeholder; may be adjusted later)
    rope_theta: float = 10000.0

    # Misc metadata
    model_type: str = field(default="gemma3_text", init=False)

    def __post_init__(self) -> None:
        self._validate_locked_spec()

    # --- HuggingFace loading helpers ---
    @classmethod
    def from_hf_dict(cls, data: Mapping[str, Any]) -> "Gemma3TextConfig":
        """Create a config from a HuggingFace-style config.json mapping.

        Only keys relevant to this 270M variant are read; others are ignored.
        Validation ensures the loaded values match the locked specification.
        """

        kwargs: dict[str, Any] = {}

        # Map permissively, fall back to locked defaults
        for key, attr in [
            ("vocab_size", "vocab_size"),
            ("hidden_size", "hidden_size"),
            ("intermediate_size", "intermediate_size"),
            ("num_hidden_layers", "num_hidden_layers"),
            ("num_attention_heads", "num_attention_heads"),
            ("num_key_value_heads", "num_key_value_heads"),
            ("head_dim", "head_dim"),
            ("bos_token_id", "bos_token_id"),
            ("eos_token_id", "eos_token_id"),
            ("pad_token_id", "pad_token_id"),
        ]:
            if key in data:
                kwargs[attr] = data[key]

        # Optional window/rope keys
        if "sliding_window" in data:
            kwargs["sliding_window"] = data["sliding_window"]
        if "rope_theta" in data:
            kwargs["rope_theta"] = data["rope_theta"]

        cfg = cls(**kwargs)  # __post_init__ enforces locked spec
        return cfg

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Gemma3TextConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_hf_dict(data)

    # --- Validation ---
    def _validate_locked_spec(self) -> None:
        """Ensure parameters match the 270M published specification.

        This class is intentionally strict to avoid silent misuse with other
        Gemma-3 sizes. If you intend to support multiple sizes, factor this
        into a general config with a variant field.
        """

        expected = {
            "vocab_size": 262_144,
            "hidden_size": 640,
            "intermediate_size": 2_048,
            "num_hidden_layers": 18,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
        }

        for name, val in expected.items():
            got = getattr(self, name)
            if got != val:
                raise ValueError(
                    f"Invalid {name}={got} for Gemma-3-270M; expected {val}. "
                    "This config is locked to the 270M variant."
                )

        # Basic token id sanity
        for tok_name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            tok = getattr(self, tok_name)
            if not isinstance(tok, int) or tok < 0:
                raise ValueError(f"{tok_name} must be a non-negative int; got {tok!r}")

