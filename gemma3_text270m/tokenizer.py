from __future__ import annotations

"""
Tokenizer utilities for Gemma-3-270M.

This module provides a small wrapper class `Gemma3Tokenizer` that supports two
backends without depending on the Transformers library:
- SentencePiece (`.model` files) via the `sentencepiece` package
- Hugging Face Tokenizers (`.json` files) via the `tokenizers` package

It standardizes encode/decode and special token handling to align with the
Gemma-3 text model conventions (PAD=0, EOS=1, BOS=2).

Notes:
- Loading from HuggingFace Hub is supported via `huggingface_hub` but is only
  used when explicitly requested.
- The class adds BOS/EOS tokens manually when `add_special_tokens=True` to
  ensure consistent behavior across backends.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import json
import os
import importlib

try:
    from typing import Literal
except ImportError:  # Python <3.8 typing backport not required here
    Literal = str  # type: ignore[assignment]


Backend = "Literal['sentencepiece','hf_tokenizers']"


# --- Dynamic imports to avoid hard deps... ---
def _import_sentencepiece():
    return importlib.import_module("sentencepiece")


def _import_tokenizers():
    return importlib.import_module("tokenizers")


def _import_hf_hub():
    return importlib.import_module("huggingface_hub")


# --- File detection helpers, for HF tokenizers and spm respectively ---
def _is_json_tokenizer_path(path: Path) -> bool:
    return path.is_file() and path.suffix == ".json"


def _is_spm_model_path(path: Path) -> bool:
    return path.is_file() and path.suffix in {".model", ".spm"}


def _search_tokenizer_file(path: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Searches for a tokenizer file under `path` if `path` is a directory.
    Returns (file_path, backend) if found; (None, None) otherwise.
    """
    if path.is_file():
        if _is_json_tokenizer_path(path):
            return path, "hf_tokenizers"
        if _is_spm_model_path(path):
            return path, "sentencepiece"
        return None, None

    if path.is_dir():
        # Prefer HF tokenizers JSON, then SentencePiece
        json_path = path / "tokenizer.json"
        if _is_json_tokenizer_path(json_path):
            return json_path, "hf_tokenizers"
        spm_path = path / "spiece.model"
        if _is_spm_model_path(spm_path):
            return spm_path, "sentencepiece"
    return None, None


@dataclass
class Gemma3Tokenizer:
    """
    Unified tokenizer wrapper for Gemma-3 text models.

    Parameters
    - backend: Which backend is active (auto-detected when loading)
    - bos_id/eos_id/pad_id: Special token ids (Gemma-3 default BOS=2, EOS=1, PAD=0)
    - Note: One of `_tok` (HF Tokenizers) or `_sp` (SentencePieceProcessor) is set.
    """

    backend: Optional[str] = None
    bos_id: int = 2
    eos_id: int = 1
    pad_id: int = 0

    # Internal handles to backend instances...
    _tok: object | None = None  # HF Tokenizers Tokenizer
    _sp: object | None = None  # SentencePieceProcessor

    # ----- Construction helpers -----
    @staticmethod
    def detect_backend(path: str | os.PathLike[str]) -> Optional[str]:
        p = Path(path)
        _, b = _search_tokenizer_file(p)
        return b

    @classmethod
    def from_local_path(cls, path: str | os.PathLike[str]) -> "Gemma3Tokenizer":
        p = Path(path)
        file_path, backend = _search_tokenizer_file(p)
        if file_path is None or backend is None:
            raise FileNotFoundError(f"No supported tokenizer file found at {p!s}")
        tok = cls()
        tok._load_backend(file_path, backend)
        return tok

    @classmethod
    def from_hf_repo(
        cls,
        repo_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "Gemma3Tokenizer":
        """
        Load tokenizer files from a HuggingFace repository using hf_hub.
        This requires `huggingface_hub` and network access unless files are cached.
        """
        hub = _import_hf_hub()
        # Choose filename if not provided
        candidate_filenames = [
            filename,
            "tokenizer.json",
            "spiece.model",
        ]
        last_err: Optional[Exception] = None
        for fname in filter(None, candidate_filenames):
            try:
                fpath = hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,  # type: ignore[arg-type]
                    revision=revision,
                    local_files_only=local_files_only,
                )
                return cls.from_local_path(fpath)
            except Exception as e:  # noqa: BLE001 - surface final error below
                last_err = e
                continue
        raise RuntimeError(
            f"Unable to download tokenizer from repo '{repo_id}'. Last error: {last_err}"
        )

    # ----- Core API -----
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids: List[int]
        if self._tok is not None:
            # HF Tokenizers API
            ids = list(self._tok.encode(text).ids)  # type: ignore[attr-defined]
        elif self._sp is not None:
            # SentencePiece API
            ids = list(self._sp.EncodeAsIds(text))  # type: ignore[attr-defined]
        else:
            # Minimal whitespace fallback (not intended for production)
            pieces = text.strip().split() if text else []
            # Map tokens to fake ids deterministically based on stable hashing
            ids = [abs(hash(t)) % 10000 + 10 for t in pieces]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]

        if self._tok is not None:
            return self._tok.decode(list(ids))  # type: ignore[attr-defined]
        if self._sp is not None:
            return self._sp.DecodeIds(list(ids))  # type: ignore[attr-defined]
        # Fallback: join placeholders (order preserved but irreversible)
        return " ".join(f"<tok:{i}>" for i in ids)

    # ----- Utilities -----
    def apply_chat_template(
        self,
        messages: Sequence[dict],
        add_generation_prompt: bool = False,
    ) -> str:
        """Very small chat template for educational testing.

        messages: list of {"role": "system|user|assistant", "content": str}
        """
        lines: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        if add_generation_prompt:
            lines.append("assistant:")
        return "\n".join(lines)

    # ----- Internal helpers -----
    def _load_backend(self, file_path: Path, backend: str) -> None:
        self.backend = backend
        if backend == "hf_tokenizers":
            toks = _import_tokenizers()
            # Load from JSON file; avoid Transformers
            self._tok = toks.Tokenizer.from_file(str(file_path))
            self._sp = None
        elif backend == "sentencepiece":
            spm = _import_sentencepiece()
            sp = spm.SentencePieceProcessor()
            loaded = sp.Load(str(file_path))
            if not loaded:  # SentencePiece returns bool
                raise RuntimeError(f"Failed to load SentencePiece model: {file_path!s}")
            self._sp = sp
            self._tok = None
        else:
            raise ValueError(f"Unsupported backend: {backend!r}")

    # Expose special tokens as read-only properties for clarity
    @property
    def special_tokens(self) -> dict:
        return {"pad": self.pad_id, "eos": self.eos_id, "bos": self.bos_id}
