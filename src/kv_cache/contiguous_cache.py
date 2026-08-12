"""Contiguous KV cache — one reservation per sequence, sized up front.

Storage is layer-major so read() is a pure view: [n_layers, B, H, max_seq_len, Hd].
"""
import torch

from src.kv_cache.base import BaseKVCache


class ContiguousKVCache(BaseKVCache):
    def __init__(self, config, batch_size, max_seq_len, device, dtype=torch.float16):
        self.n_layers, self.n_heads, self.head_dim = config.n_layers, config.n_heads, config.head_dim
        self.batch_size, self.max_seq_len = batch_size, max_seq_len
        self.device, self.dtype = device, dtype

        shape = (self.n_layers, batch_size, self.n_heads, max_seq_len, self.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

        self.seq_len = 0    # logical length, driven by advance()
        self.filled = 0     # written extent, driven by write()

    def write(self, layer_idx, k, v):
        T = k.shape[2]
        start = self.seq_len
        if start + T > self.max_seq_len:
            raise RuntimeError("sequence exceeds max_seq_len")
        self.k[layer_idx, :, :, start:start + T, :] = k
        self.v[layer_idx, :, :, start:start + T, :] = v
        # read() must see tokens written in THIS forward; advance() lags a forward.
        self.filled = start + T

    def read(self, layer_idx):
        n = self.filled
        return (self.k[layer_idx, :, :, :n, :],
                self.v[layer_idx, :, :, :n, :])   # views, no copy

    def advance(self, amount=1):
        self.seq_len += amount

    def free(self):
        self.seq_len = 0
        self.filled = 0

    def reset(self):
        self.k.zero_()
        self.v.zero_()
        self.seq_len = 0
        self.filled = 0

    def memory_bytes(self):
        return self.k.numel() * self.k.element_size() * 2   # k + v

    def fragmentation_ratio(self):
        """Reservation waste as a fraction of total allocation (see BaseKVCache).

        allocated_b is the up-front reservation, which exists from construction —
        so an unwritten cache is 1.0 (all of it wasted), unlike paged, which
        allocates nothing until the first write.
        """
        alloc = [self.max_seq_len] * self.batch_size
        used = self.lengths                      # written extent, not seq_len
        total = sum(alloc)
        return (total - sum(used)) / total if total else 0.0

    @property
    def lengths(self):
        return [self.filled] * self.batch_size
