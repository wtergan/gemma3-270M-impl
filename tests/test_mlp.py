import torch

from gemma3_text270m.config import Gemma3TextConfig
from gemma3_text270m.mlp import Gemma3MLP


def test_mlp_shapes_and_weights():
    cfg = Gemma3TextConfig()
    mlp = Gemma3MLP(cfg)

    # Weight shapes follow [out_features, in_features]
    assert mlp.gate_proj.weight.shape == (cfg.intermediate_size, cfg.hidden_size)
    assert mlp.up_proj.weight.shape == (cfg.intermediate_size, cfg.hidden_size)
    assert mlp.down_proj.weight.shape == (cfg.hidden_size, cfg.intermediate_size)

    # No bias terms
    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None


def test_mlp_forward_shape_and_gating_zero_case():
    cfg = Gemma3TextConfig()
    mlp = Gemma3MLP(cfg)
    x = torch.zeros(2, 4, cfg.hidden_size)

    # With zero input, GELU(0) = 0 so output should be exactly zeros
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.zeros_like(y))


def test_mlp_forward_random():
    cfg = Gemma3TextConfig()
    mlp = Gemma3MLP(cfg)
    x = torch.randn(1, 3, cfg.hidden_size)
    y = mlp(x)
    assert y.shape == x.shape

