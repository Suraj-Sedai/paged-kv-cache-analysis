"""Paged KV cache — fixed-size blocks handed out on demand via a block table.

Storage is the page pool only; read() gathers a sequence's physical pages, so
memory_bytes() reflects true occupancy with no shadow contiguous copy.

MEASUREMENT CAVEAT (load-bearing — keep this in the paper): read() materialises
the full history into a dense [B, H, n, Hd] tensor. A real PagedAttention kernel
consumes the block table inside attention and never builds this tensor. So the
latency measured against this class is the cost of paging *without a fused
kernel*, not an intrinsic property of the paged layout. That is a legitimate
quantity — it is the software-overhead argument vAttention (ASPLOS'25) makes
qualitatively — but it must be labelled as such, not reported as "paged is slow".

The block table is mirrored into a [B, max_pages] int64 tensor so write/read are
single vectorised ops instead of a Python loop over the batch.
"""
import torch

from src.kv_cache.base import BaseKVCache


class PagedKVCache(BaseKVCache):
    def __init__(self, n_layers, n_heads, dim_head, block_size, num_blocks,
                 batch_size, device, dtype=torch.float16):
        self.n_layers, self.n_heads, self.head_dim = n_layers, n_heads, dim_head
        self.page_size, self.num_blocks = block_size, num_blocks
        self.batch_size, self.device, self.dtype = batch_size, device, dtype

        # pool: [num_blocks, n_layers, n_heads, block_size, head_dim]
        self.k_pages = torch.zeros(num_blocks, n_layers, n_heads, block_size, dim_head,
                                   device=device, dtype=dtype)
        self.v_pages = torch.zeros(num_blocks, n_layers, n_heads, block_size, dim_head,
                                   device=device, dtype=dtype)

        self.next_free = 0                       # single global bump pointer into the pool
        self.n_pages = 0                         # pages allocated per sequence (uniform)
        self.page_table = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        self.seq_len = 0
        self.filled = 0

    def _ensure_capacity(self, total_len):
        need = (total_len + self.page_size - 1) // self.page_size
        if need <= self.n_pages:
            return
        extra = need - self.n_pages
        if self.next_free + extra * self.batch_size > self.num_blocks:
            raise RuntimeError("OOM: no free pages")
        new = torch.arange(self.next_free, self.next_free + extra * self.batch_size,
                           device=self.device, dtype=torch.long).view(self.batch_size, extra)
        self.next_free += extra * self.batch_size
        self.page_table = torch.cat([self.page_table, new], dim=1)
        self.n_pages = need

    def _index(self, total_positions, start=0):
        """(phys [B, T], off [T]) for logical positions start .. start+T-1."""
        pos = torch.arange(start, start + total_positions, device=self.device)
        return self.page_table[:, pos // self.page_size], pos % self.page_size

    def write(self, layer_idx, k, v):
        T = k.shape[2]
        start = self.seq_len
        self._ensure_capacity(start + T)
        phys, off = self._index(T, start)
        # advanced indices separated by a slice -> result is [B, T, H, Hd]
        self.k_pages[phys, layer_idx, :, off, :] = k.permute(0, 2, 1, 3)
        self.v_pages[phys, layer_idx, :, off, :] = v.permute(0, 2, 1, 3)
        self.filled = start + T

    def read(self, layer_idx):
        n = self.filled
        phys, off = self._index(n)
        k = self.k_pages[phys, layer_idx, :, off, :]   # [B, n, H, Hd]
        v = self.v_pages[phys, layer_idx, :, off, :]
        return k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)   # [B, H, n, Hd]

    def advance(self, amount=1):
        self.seq_len += amount

    def free(self):
        self.next_free = 0
        self.n_pages = 0
        self.page_table = torch.zeros(self.batch_size, 0, dtype=torch.long, device=self.device)
        self.seq_len = 0
        self.filled = 0

    def reset(self):
        self.free()

    def memory_bytes(self):
        used = self.n_pages * self.batch_size
        per_page = self.k_pages[0].numel() * self.k_pages.element_size()
        return 2 * used * per_page   # k + v

    def fragmentation_ratio(self):
        """Page waste as a fraction of total allocation (see BaseKVCache).

        Supersedes the old last-block definition, which divided the final page's
        unused slots by page_size rather than by the allocation. That reported
        waste as a fraction of ONE page, so it was ~constant in sequence length
        and overstated true waste by n_pages x (12x at seq 188 / page 16, 132x
        at seq 2108). Waste here is bounded by one page and correctly amortises.

        Unhanded pool blocks are over-provisioning, not fragmentation, and are
        excluded — the denominator counts only pages this batch actually holds.
        """
        alloc = [self.n_pages * self.page_size] * self.batch_size
        used = self.lengths                      # written extent, not seq_len
        total = sum(alloc)
        return (total - sum(used)) / total if total else 0.0

    @property
    def lengths(self):
        return [self.filled] * self.batch_size
