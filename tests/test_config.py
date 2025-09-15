import json
from pathlib import Path
import pytest

from gemma3_text270m.config import Gemma3TextConfig


def test_default_locked_values():
    cfg = Gemma3TextConfig()
    assert cfg.vocab_size == 262_144
    assert cfg.hidden_size == 640
    assert cfg.intermediate_size == 2_048
    assert cfg.num_hidden_layers == 18
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 1
    assert cfg.head_dim == 256


def test_from_hf_dict_valid():
    data = {
        "vocab_size": 262_144,
        "hidden_size": 640,
        "intermediate_size": 2_048,
        "num_hidden_layers": 18,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    }
    cfg = Gemma3TextConfig.from_hf_dict(data)
    assert isinstance(cfg, Gemma3TextConfig)


def test_from_hf_dict_invalid_raises():
    bad = {
        "vocab_size": 32000,  # wrong
    }
    with pytest.raises(ValueError):
        Gemma3TextConfig.from_hf_dict(bad)


def test_from_json_file(tmp_path: Path):
    p = tmp_path / "config.json"
    payload = {
        "vocab_size": 262_144,
        "hidden_size": 640,
        "intermediate_size": 2_048,
        "num_hidden_layers": 18,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    cfg = Gemma3TextConfig.from_json_file(p)
    assert cfg.hidden_size == 640
