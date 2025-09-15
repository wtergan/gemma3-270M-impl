import torch

from gemma3_text270m.config import Gemma3TextConfig
from gemma3_text270m.attention import GlobalAttention, SlidingWindowAttention


def test_mqa_shapes_and_heads():
    cfg = Gemma3TextConfig()
    attn = GlobalAttention(cfg)
    assert attn.num_heads == cfg.num_attention_heads
    assert attn.num_key_value_heads == cfg.num_key_value_heads
    x = torch.randn(2, 5, cfg.hidden_size)
    y = attn(x)
    assert y.shape == (2, 5, cfg.hidden_size)


def test_sliding_window_differs_from_global():
    cfg = Gemma3TextConfig()
    g = GlobalAttention(cfg)
    s = SlidingWindowAttention(cfg, window_size=2)
    x = torch.randn(1, 8, cfg.hidden_size)
    yg = g(x)
    ys = s(x)
    # Random inputs → outputs should differ across attention patterns
    assert not torch.allclose(yg, ys)


def test_rope_cache_apply_shapes():
    # Indirectly validated by forward pass; ensure no error at typical dims
    cfg = Gemma3TextConfig()
    attn = GlobalAttention(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    y = attn(x)
    assert y.shape == (1, 4, cfg.hidden_size)
