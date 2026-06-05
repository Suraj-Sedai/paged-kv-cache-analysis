import torch

from src.kv_cache.base import BaseKVCache


class ContiguousKVCache(BaseKVCache):
    def __init__(self, config, max_seq_len, device, dtype=torch.float16):
        self.n_layers, self.n_heads, self.head_dim = config.n_layers, config.n_heads, config.head_dim
        self.max_seq_len, self.device, self.dtype = max_seq_len, device, dtype
        self.k = {}            # seq_id -> [n_layers, n_heads, max_seq_len, head_dim]
        self.v = {}
        self.seq_len = {}      # logical length, driven by advance()
        self.filled = {}       # actual written extent, driven by write()

    def _ensure(self, seq_id):
        if seq_id not in self.k:
            shape = (self.n_layers, self.n_heads, self.max_seq_len, self.head_dim)
            self.k[seq_id] = torch.zeros(shape, device=self.device, dtype=self.dtype)
            self.v[seq_id] = torch.zeros(shape, device=self.device, dtype=self.dtype)
            self.seq_len[seq_id] = 0
            self.filled[seq_id] = 0

    def write(self, layer_idx, seq_id, k, v):
        self._ensure(seq_id)
        start = self.seq_len[seq_id]
        T = k.shape[1]
        if start + T > self.max_seq_len:
            raise RuntimeError(f"seq {seq_id} exceeds max_seq_len")
        self.k[seq_id][layer_idx, :, start:start+T, :] = k
        self.v[seq_id][layer_idx, :, start:start+T, :] = v
        # read must see tokens written in THIS forward; advance() lags by a forward.
        self.filled[seq_id] = start + T

    def read(self, layer_idx, seq_id):
        n = self.filled[seq_id]
        return self.k[seq_id][layer_idx, :, :n, :], self.v[seq_id][layer_idx, :, :n, :]   # view, no copy

    def advance(self, seq_id, amount=1):
        self.seq_len[seq_id] += amount

    def reset(self):
        self.k = {}
        self.v = {}
        self.seq_len = {}
        self.filled = {}

    def free(self, seq_id):
        self.k.pop(seq_id, None); self.v.pop(seq_id, None)
        self.seq_len.pop(seq_id, None); self.filled.pop(seq_id, None)

    def memory_bytes(self):
        elem = torch.empty((), dtype=self.dtype).element_size()   # dtype-accurate, matches paged
        per = self.n_layers * self.n_heads * self.max_seq_len * self.head_dim * elem
        return 2 * len(self.k) * per   # k + v

    def fragmentation_ratio(self, seq_id):
        if seq_id not in self.seq_len:
            return 0.0
        return (self.max_seq_len - self.seq_len[seq_id]) / self.max_seq_len