from __future__ import annotations

"""
KV cache utilities for Gemma-3-270M (educational).

Provides two cache strategies for autoregressive decoding:
- RingBufferKVCache: keeps only the most recent `capacity` tokens (sliding window)
- FullSequenceKVCache: keeps the entire sequence history

Tensor shapes follow MQA (single KV head) defaults but are general:
  K/V cache: [B, Kvh, L, Dh]
  New K/V:   [B, Kvh, T, Dh] (T usually 1 during decode)
"""

from typing import Optional, Tuple
import torch
from .config import Gemma3TextConfig


class KVCache:
    """
    Abstract KV cache interface. Subclasses must implement `reset`, `append`, and `get`.
    """

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:  # pragma: no cover
        raise NotImplementedError

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    @property
    def capacity(self) -> Optional[int]:  # pragma: no cover
        return None


class RingBufferKVCache(KVCache):
    """
    Ring-buffer KV cache for sliding-window attention layers.

    Stores at most `capacity` tokens. Appends wrap around and overwrite the
    oldest entries. Retrieval returns keys/values in chronological order.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
        *,
        capacity: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.kvh = int(config.num_key_value_heads)
        self.dh = int(config.head_dim)
        self._capacity = int(capacity if capacity is not None else config.sliding_window)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self._pos = 0
        self._size = 0
        self._allocated = False
        self._bsz = int(batch_size) if batch_size is not None else None

        self._k: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None

    def _ensure_alloc(self, bsz: int) -> None:
        if self._allocated:
            if self._bsz != bsz:
                # Reallocate if batch size changes
                self._bsz = bsz
                self._k = torch.zeros(
                    (bsz, self.kvh, self._capacity, self.dh), device=self.device, dtype=self.dtype
                )
                self._v = torch.zeros(
                    (bsz, self.kvh, self._capacity, self.dh), device=self.device, dtype=self.dtype
                )
                self._pos = 0
                self._size = 0
            return
        self._bsz = bsz
        self._k = torch.zeros(
            (bsz, self.kvh, self._capacity, self.dh), device=self.device, dtype=self.dtype
        )
        self._v = torch.zeros(
            (bsz, self.kvh, self._capacity, self.dh), device=self.device, dtype=self.dtype
        )
        self._pos = 0
        self._size = 0
        self._allocated = True

    def reset(self) -> None:
        self._pos = 0
        self._size = 0
        # Keep allocated buffers for reuse
        if self._k is not None:
            self._k.zero_()
        if self._v is not None:
            self._v.zero_()

    @torch.no_grad()
    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        # Expect [B, Kvh, T, Dh]
        if k_new.dim() != 4:
            raise ValueError(f"k_new must be 4D [B,Kvh,T,Dh]; got {k_new.shape}")
        if v_new.shape != k_new.shape:
            raise ValueError("k_new and v_new must have identical shapes")
        bsz, kvh, t, dh = k_new.shape
        if kvh != self.kvh or dh != self.dh:
            raise ValueError("KV head count or head dim mismatch with cache")

        self._ensure_alloc(bsz)
        assert self._k is not None and self._v is not None

        # Write sequentially with wrap-around
        for i in range(t):
            write_idx = self._pos
            self._k[:, :, write_idx, :] = k_new[:, :, i, :]
            self._v[:, :, write_idx, :] = v_new[:, :, i, :]
            self._pos = (self._pos + 1) % self._capacity
            self._size = min(self._size + 1, self._capacity)

    def _gather_ordered(self, buf: torch.Tensor) -> torch.Tensor:
        # Return last `_size` entries in chronological order along length dim
        if self._size == 0:
            # Return an empty view for consistency
            return buf[:, :, :0, :]
        start = (self._pos - self._size) % self._capacity
        idx = (torch.arange(self._size, device=buf.device) + start) % self._capacity
        return buf.index_select(dim=2, index=idx)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._allocated or self._k is None or self._v is None:
            raise RuntimeError("Cache not allocated/initialized")
        k = self._gather_ordered(self._k)
        v = self._gather_ordered(self._v)
        return k, v

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size


class FullSequenceKVCache(KVCache):
    """
    Full sequence KV cache for global attention layers.

    Stores the entire history by concatenating along the sequence length
    dimension. Simpler but with linear memory growth.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.kvh = int(config.num_key_value_heads)
        self.dh = int(config.head_dim)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self._k: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._k = None
        self._v = None

    @torch.no_grad()
    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        # Expect [B, Kvh, T, Dh]
        if k_new.dim() != 4:
            raise ValueError(f"k_new must be 4D [B,Kvh,T,Dh]; got {k_new.shape}")
        if v_new.shape != k_new.shape:
            raise ValueError("k_new and v_new must have identical shapes")
        bsz, kvh, t, dh = k_new.shape
        if kvh != self.kvh or dh != self.dh:
            raise ValueError("KV head count or head dim mismatch with cache")

        if self._k is None:
            self._k = k_new.to(device=self.device, dtype=self.dtype).clone()
            self._v = v_new.to(device=self.device, dtype=self.dtype).clone()
            return

        if self._k.shape[0] != bsz:
            # Reinitialize on batch size change for simplicity
            self._k = k_new.to(device=self.device, dtype=self.dtype).clone()
            self._v = v_new.to(device=self.device, dtype=self.dtype).clone()
            return

        self._k = torch.cat([self._k, k_new.to(self.device, self.dtype)], dim=2)
        self._v = torch.cat([self._v, v_new.to(self.device, self.dtype)], dim=2)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._k is None or self._v is None:
            raise RuntimeError("Cache is empty")
        return self._k, self._v

    @property
    def capacity(self) -> Optional[int]:
        return None


__all__ = ["KVCache", "RingBufferKVCache", "FullSequenceKVCache"]
