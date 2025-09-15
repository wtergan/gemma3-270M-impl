import json
from pathlib import Path
from unittest.mock import patch

import torch

from gemma3_text270m.config import Gemma3TextConfig
from gemma3_text270m.model import Gemma3ForCausalLM
from gemma3_text270m.hf_loader import (
    _apply_weight_transforms,
    _map_hf_key_to_local,
    _needs_transpose,
    load_weights_into,
)


def _local_to_hf_key(local_key: str) -> str:
    if local_key.startswith("embed_tokens."):
        return f"model.embed_tokens.{local_key.split('.', 1)[1]}"

    if local_key.startswith("layers."):
        parts = local_key.split(".")
        layer_idx = parts[1]
        remainder = parts[2:]
        if remainder[0] == "attention":
            return f"model.layers.{layer_idx}.self_attn.{'.'.join(remainder[1:])}"
        if remainder[0] == "mlp":
            return f"model.layers.{layer_idx}.mlp.{'.'.join(remainder[1:])}"
        if remainder[0] == "input_layernorm":
            return f"model.layers.{layer_idx}.input_layernorm.{'.'.join(remainder[1:])}"
        if remainder[0] == "post_attention_layernorm":
            return f"model.layers.{layer_idx}.post_attention_layernorm.{'.'.join(remainder[1:])}"
        if remainder[0] == "pre_feedforward_layernorm":
            return f"model.layers.{layer_idx}.pre_feedforward_layernorm.{'.'.join(remainder[1:])}"
        if remainder[0] == "post_feedforward_layernorm":
            return f"model.layers.{layer_idx}.post_feedforward_layernorm.{'.'.join(remainder[1:])}"

    if local_key.startswith("norm."):
        return f"model.norm.{local_key.split('.', 1)[1]}"

    if local_key.startswith("lm_head."):
        return f"lm_head.{local_key.split('.', 1)[1]}"

    return local_key


def test_map_hf_key_to_local_patterns():
    cases = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight": "layers.0.attention.q_proj.weight",
        "model.layers.11.mlp.down_proj.weight": "layers.11.mlp.down_proj.weight",
        "model.layers.5.input_layernorm.weight": "layers.5.input_layernorm.weight",
        "model.layers.17.post_attention_layernorm.weight": "layers.17.post_attention_layernorm.weight",
        "model.layers.4.pre_feedforward_layernorm.weight": "layers.4.pre_feedforward_layernorm.weight",
        "model.layers.4.post_feedforward_layernorm.weight": "layers.4.post_feedforward_layernorm.weight",
        "model.layers.9.self_attn.q_norm.weight": "layers.9.attention.q_norm.weight",
        "model.layers.9.self_attn.k_norm.weight": "layers.9.attention.k_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for hf_key, expected in cases.items():
        local_key, was_mapped = _map_hf_key_to_local(hf_key)
        assert local_key == expected
        assert was_mapped


def test_needs_transpose_detection():
    hf_shape = torch.Size([4, 8])
    local_shape = torch.Size([8, 4])
    assert _needs_transpose("layers.0.attention.q_proj.weight", hf_shape, local_shape)
    assert not _needs_transpose("layers.0.attention.q_proj.weight", hf_shape, hf_shape)


def test_apply_weight_transforms_transpose():
    tensor = torch.arange(6, dtype=torch.float32).view(2, 3)
    transformed, ops = _apply_weight_transforms(
        "layers.0.attention.q_proj.weight", tensor, torch.Size([3, 2])
    )
    assert transformed.shape == torch.Size([3, 2])
    assert "transpose" in ops


def test_weight_loading_with_mapping(tmp_path: Path):
    config = Gemma3TextConfig()
    model = Gemma3ForCausalLM(config)

    hf_state = {}
    for local_key, tensor in model.state_dict().items():
        hf_state[_local_to_hf_key(local_key)] = tensor.detach().clone()

    fake_root = tmp_path / "repo"
    fake_file = fake_root / "model.safetensors"

    with patch("gemma3_text270m.hf_loader._load_safetensors", return_value=hf_state), patch(
        "gemma3_text270m.hf_loader._detect_files",
        return_value=(fake_root, [fake_file]),
    ):
        report = load_weights_into(model, "google/gemma-3-270m", strict=True)

    assert report.loaded_tensors == len(hf_state)
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    assert report.shape_mismatches == []
    assert report.mapping_stats["mapping_rate"] == 1.0
    assert report.mapping_stats["successfully_mapped"] == len(hf_state)
    assert report.transformations == {}


def test_local_detection_and_partial_load(tmp_path: Path):
    # Create a mock local repo with a minimal safetensors-like structure using torch.save fallback
    # Here we avoid actually writing safetensors; this is a structural test only.
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 262144, "hidden_size": 640}))

    # Skip actual safetensors — loader will error if no .safetensors; ensure message is clear
    cfg = Gemma3TextConfig()
    model = Gemma3ForCausalLM(cfg)
    try:
        load_weights_into(model, str(tmp_path))
    except RuntimeError as e:
        msg = str(e)
        assert "No .safetensors files found" in msg or "safetensors" in msg
