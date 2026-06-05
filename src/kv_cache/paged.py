
"""Paged KV Cache implementation."""
import torch
from src.kv_cache.base import BaseKVCache


class PagedKVCache(BaseKVCache):
    """Manages KV cache using fixed-size pages to reduce fragmentation.

    Storage is the page pool only — read() reconstructs a sequence's K/V by
    gathering its physical pages on demand (the real paged-attention cost). No
    shadow contiguous copy is kept, so memory_bytes() reflects true occupancy.

    dtype is caller-chosen and should match the contiguous cache so a layout
    comparison isolates layout (not precision); precision is a separate axis.
    """

    def __init__(self, n_layers, n_heads, dim_head, block_size, num_blocks, device, dtype=torch.float32):
        self.n_layers, self.n_heads, self.head_dim = n_layers, n_heads, dim_head
        self.page_size, self.num_blocks = block_size, num_blocks
        self.device, self.dtype = device, dtype

        # pool: [num_blocks, n_layers, n_heads, block_size, head_dim] — NO batch dim
        self.k_pages = torch.zeros(num_blocks, n_layers, n_heads, block_size, dim_head, device=device, dtype=dtype)
        self.v_pages = torch.zeros(num_blocks, n_layers, n_heads, block_size, dim_head, device=device, dtype=dtype)

        self.free_pages = list(range(num_blocks))   # single global pool
        self.page_table = {}                        # seq_id -> [physical_page, ...]
        self.seq_len    = {}                        # seq_id -> logical length (advance())
        self.filled     = {}                        # seq_id -> written extent (write())

    def _ensure_capacity(self, seq_id, total_len):
        pages = self.page_table.setdefault(seq_id, [])
        need = (total_len + self.page_size - 1) // self.page_size
        while len(pages) < need:
            if not self.free_pages:
                raise RuntimeError("OOM: no free pages")
            pages.append(self.free_pages.pop())

    def free(self, seq_id):                          # call on sequence completion
        self.free_pages.extend(self.page_table.pop(seq_id, []))
        self.seq_len.pop(seq_id, None)
        self.filled.pop(seq_id, None)

    def write(self, layer_idx, seq_id, k, v):
        t_new = k.shape[1]
        start = self.seq_len.get(seq_id, 0)
        self._ensure_capacity(seq_id, start + t_new)
        pages, written = self.page_table[seq_id], 0
        while written < t_new:
            pos = start + written
            lp, off = pos // self.page_size, pos % self.page_size
            chunk = min(self.page_size - off, t_new - written)
            phys = pages[lp]
            self.k_pages[phys, layer_idx, :, off:off+chunk, :].copy_(k[:, written:written+chunk, :])
            self.v_pages[phys, layer_idx, :, off:off+chunk, :].copy_(v[:, written:written+chunk, :])
            written += chunk
        self.filled[seq_id] = start + t_new

    def read(self, layer_idx, seq_id):
        """Gather the sequence's physical pages into a contiguous [H, n, Hd] view."""
        n = self.filled[seq_id]                       # KeyError if never written / freed
        pages = self.page_table[seq_id]
        need = (n + self.page_size - 1) // self.page_size
        phys = pages[:need]
        k = self.k_pages[phys, layer_idx]             # [P, H, page_size, Hd]
        v = self.v_pages[phys, layer_idx]
        P, H, ps, Hd = k.shape
        k = k.permute(1, 0, 2, 3).reshape(H, P * ps, Hd)[:, :n, :]
        v = v.permute(1, 0, 2, 3).reshape(H, P * ps, Hd)[:, :n, :]
        return k, v

    def reset(self):
        self.free_pages = list(range(self.num_blocks))
        self.page_table = {}
        self.seq_len = {}
        self.filled = {}

    def advance(self, seq_id, amount=1):
        self.seq_len[seq_id] = self.seq_len.get(seq_id, 0) + amount

    def memory_bytes(self):
        used = sum(len(p) for p in self.page_table.values())   # physical pages held now
        per_page = self.k_pages[0].numel() * self.k_pages.element_size()
        return 2 * used * per_page                              # k + v

    def fragmentation_ratio(self, seq_id) -> float:
        total = self.seq_len.get(seq_id, 0)
        if total == 0:
            return 0.0
        rem = total % self.page_size
        return 0.0 if rem == 0 else (self.page_size - rem) / self.page_size
