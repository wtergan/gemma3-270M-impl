import torch

from gemma3_text270m.config import Gemma3TextConfig
from gemma3_text270m.block import Gemma3Block
from gemma3_text270m.attention import SlidingWindowAttention, GlobalAttention


def test_block_forward_shapes_sliding_and_global():
    cfg = Gemma3TextConfig()
    x = torch.randn(2, 5, cfg.hidden_size)

    blk_s = Gemma3Block(cfg, layer_type="sliding_attention")
    y_s = blk_s(x)
    assert y_s.shape == x.shape
    assert isinstance(blk_s.attention, SlidingWindowAttention)

    blk_g = Gemma3Block(cfg, layer_type="full_attention")
    y_g = blk_g(x)
    assert y_g.shape == x.shape
    assert isinstance(blk_g.attention, GlobalAttention)


def test_block_residual_nontrivial_transform():
    cfg = Gemma3TextConfig()
    x = torch.randn(1, 3, cfg.hidden_size)
    blk = Gemma3Block(cfg, layer_type="full_attention")
    y = blk(x)
    assert y.shape == x.shape
    # With random weights it's unlikely to be exactly equal
    assert not torch.allclose(y, x)
