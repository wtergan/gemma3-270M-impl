import torch

from gemma3_text270m.config import Gemma3TextConfig
from gemma3_text270m.kvcache import RingBufferKVCache, FullSequenceKVCache


def test_ring_buffer_capacity_and_append_get():
    cfg = Gemma3TextConfig()
    cap = 8
    cache = RingBufferKVCache(cfg, capacity=cap, device=torch.device("cpu"))

    B, K, D = 1, cfg.num_key_value_heads, cfg.head_dim
    # Append more than capacity tokens, T=1 each time
    steps = cap + 3
    for t in range(steps):
        k_new = torch.full((B, K, 1, D), float(t + 1))
        v_new = torch.full((B, K, 1, D), float(t + 1))
        cache.append(k_new, v_new)

    k, v = cache.get()
    assert k.shape == (B, K, cap, D)
    assert v.shape == (B, K, cap, D)
    # Should contain the last `cap` values in order
    # First element corresponds to t = steps - cap + 1
    first_val = steps - cap + 1
    assert torch.isclose(k[0, 0, 0, 0], torch.tensor(float(first_val)))
    assert torch.isclose(k[0, 0, -1, 0], torch.tensor(float(steps)))


def test_full_sequence_cache_concat():
    cfg = Gemma3TextConfig()
    cache = FullSequenceKVCache(cfg)
    B, K, D = 2, cfg.num_key_value_heads, cfg.head_dim

    k1 = torch.randn(B, K, 1, D)
    v1 = torch.randn(B, K, 1, D)
    cache.append(k1, v1)
    k2 = torch.randn(B, K, 3, D)
    v2 = torch.randn(B, K, 3, D)
    cache.append(k2, v2)

    k, v = cache.get()
    assert k.shape == (B, K, 4, D)
    assert v.shape == (B, K, 4, D)
