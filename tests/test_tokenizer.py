import json
from pathlib import Path

import pytest

from gemma3_text270m.tokenizer import Gemma3Tokenizer


def _write_minimal_hf_tokenizer(tmp_path: Path) -> Path:
    """Create a minimal HF Tokenizers JSON (WordLevel) with special tokens.

    Vocabulary:
    0:[PAD], 1:[EOS], 2:[BOS], 3:hello, 4:world
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "hello": 3, "world": 4}
    model = WordLevel(vocab=vocab, unk_token="[UNK]")
    tok = Tokenizer(model)
    tok.pre_tokenizer = Whitespace()
    j = tok.to_str()
    path = tmp_path / "tokenizer.json"
    path.write_text(j, encoding="utf-8")
    return path


def test_backend_detection_json_vs_model(tmp_path: Path):
    # JSON tokenizer
    jpath = _write_minimal_hf_tokenizer(tmp_path)
    assert Gemma3Tokenizer.detect_backend(jpath) == "hf_tokenizers"

    # SPM model detection via extension (not loading actual model)
    spm = tmp_path / "spiece.model"
    spm.write_bytes(b"dummy")
    assert Gemma3Tokenizer.detect_backend(spm) == "sentencepiece"


def test_encode_decode_roundtrip_hf_tokenizers(tmp_path: Path):
    jpath = _write_minimal_hf_tokenizer(tmp_path)
    tok = Gemma3Tokenizer.from_local_path(jpath)

    text = "hello world"
    ids = tok.encode(text, add_special_tokens=True)
    # Expect BOS at start and EOS at end
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id

    detok = tok.decode(ids, skip_special_tokens=True)
    assert detok.strip() == text


def test_special_token_ids_default(tmp_path: Path):
    jpath = _write_minimal_hf_tokenizer(tmp_path)
    tok = Gemma3Tokenizer.from_local_path(jpath)
    assert tok.bos_id == 2
    assert tok.eos_id == 1
    assert tok.pad_id == 0


def test_sentencepiece_missing_dependency_raises(tmp_path: Path, monkeypatch):
    # Force import failure for sentencepiece to validate error path
    def fake_import(name):
        if name == "sentencepiece":
            raise ImportError("sentencepiece not installed")
        return __import__(name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    spm = tmp_path / "spiece.model"
    spm.write_bytes(b"dummy")

    # Loading should fail due to missing dependency
    with pytest.raises(ImportError):
        Gemma3Tokenizer.from_local_path(spm)

