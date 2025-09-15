import torch

from gemma3_text270m.config import Gemma3TextConfig
from gemma3_text270m.model import Gemma3ForCausalLM
from gemma3_text270m.block import Gemma3Block
from gemma3_text270m.attention import SlidingWindowAttention, GlobalAttention


def test_model_assembly_and_layer_types():
    cfg = Gemma3TextConfig()
    model = Gemma3ForCausalLM(cfg)
    assert len(model.layers) == cfg.num_hidden_layers
    # Check 5:1 sliding:global pattern for first 6 layers and repeated
    for i in range(cfg.num_hidden_layers):
        layer = model.layers[i]
        assert isinstance(layer, Gemma3Block)
        if (i % 6) == 5:
            assert isinstance(layer.attention, GlobalAttention)
        else:
            assert isinstance(layer.attention, SlidingWindowAttention)


def test_model_forward_and_weight_tying():
    cfg = Gemma3TextConfig()
    model = Gemma3ForCausalLM(cfg)
    assert model.lm_head.weight is model.embed_tokens.weight

    B, T = 2, 7
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size)
