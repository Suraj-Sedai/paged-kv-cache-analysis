
"""Paged KV Cache implementation."""
import torch
from src.kv_cache.base import BaseKVCache


class PagedKVCache(BaseKVCache):
    """Manages KV cache using fixed-size pages to reduce fragmentation."""

    def __init__(
        self,
        n_layers,
        n_heads,
        dim_head,
        block_size,
        num_blocks,
        device,
        batch_size,
        dtype=torch.float32,
    ):
        self.n_layers = n_layers
        self.num_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = dim_head
        self.page_size = block_size
        self.max_pages = num_blocks
        self.max_seq_len = block_size * num_blocks
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.curr_len = 0

        self.k_pages = [
            torch.zeros(
                num_blocks,
                batch_size,
                n_heads,
                block_size,
                dim_head,
                device=device,
                dtype=dtype,
            )
            for _ in range(n_layers)
        ]
        self.v_pages = [
            torch.zeros(
                num_blocks,
                batch_size,
                n_heads,
                block_size,
                dim_head,
                device=device,
                dtype=dtype,
            )
            for _ in range(n_layers)
        ]
        self.page_table = [[None] * self.max_pages for _ in range(self.n_layers)]
        self.free_pages = [list(range(self.max_pages)) for _ in range(self.n_layers)]

    def _num_used_pages(self):
        if self.curr_len == 0:
            return 0
        return (self.curr_len + self.page_size - 1) // self.page_size

    def _get_or_allocate_page(self, layer_id, logical_page_idx):
        physical_page = self.page_table[layer_id][logical_page_idx]
        if physical_page is None:
            if not self.free_pages[layer_id]:
                raise ValueError("KV cache out of pages!")
            physical_page = self.free_pages[layer_id].pop(0)
            self.page_table[layer_id][logical_page_idx] = physical_page
        return physical_page

    def _write_tokens(self, page_store, layer_id, tensor_new):
        _, _, t_new, _ = tensor_new.shape
        write_pos = self.curr_len
        tokens_written = 0

        while tokens_written < t_new:
            logical_page_idx = write_pos // self.page_size
            page_offset = write_pos % self.page_size
            physical_page = self._get_or_allocate_page(layer_id, logical_page_idx)

            chunk_size = min(self.page_size - page_offset, t_new - tokens_written)
            page_store[layer_id][physical_page, :, :, page_offset:page_offset + chunk_size, :].copy_(
                tensor_new[:, :, tokens_written:tokens_written + chunk_size, :]
            )

            write_pos += chunk_size
            tokens_written += chunk_size

    def _materialize(self, page_store, layer_id, total_len):
        if total_len == 0:
            return torch.empty(
                self.batch_size,
                self.n_heads,
                0,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

        output = torch.empty(
            self.batch_size,
            self.n_heads,
            total_len,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        read_pos = 0
        logical_page_count = (total_len + self.page_size - 1) // self.page_size
        for logical_page_idx in range(logical_page_count):
            physical_page = self.page_table[layer_id][logical_page_idx]
            if physical_page is None:
                raise ValueError("KV cache page table is missing data for requested sequence length.")

            chunk_size = min(self.page_size, total_len - read_pos)
            output[:, :, read_pos:read_pos + chunk_size, :].copy_(
                page_store[layer_id][physical_page, :, :, :chunk_size, :]
            )
            read_pos += chunk_size

        return output

    def write(self, layer_idx, k, v, start_pos):
        """Update cache with new keys and values."""
        _, _, t_new, _ = k.shape
        if self.curr_len + t_new > self.max_seq_len:
            raise ValueError("KV cache out of capacity!")

        self._write_tokens(self.k_pages, layer_idx, k)
        self._write_tokens(self.v_pages, layer_idx, v)
        
        self.advance(t_new)

    def read(self, layer_idx, upto_pos):
        """Write new K/V into pages and return a contiguous tensor for attention."""
        _, _, t_new, _ = k.shape
        if self.curr_len + t_new > self.max_seq_len:
            raise ValueError("KV cache out of capacity!")

        self.write(layer_idx, k, v, self.curr_len)
        end_pos = self.curr_len + t_new
        return (
            self._materialize(self.k_pages, layer_idx, end_pos),
            self._materialize(self.v_pages, layer_idx, end_pos),
        )

    def reset(self):
        """Reset cache metadata to the empty state."""
        self.curr_len = 0
        self.page_table = [[None] * self.max_pages for _ in range(self.n_layers)]
        self.free_pages = [list(range(self.max_pages)) for _ in range(self.n_layers)]

    def advance(self, amount):
        """Advance the current position by a certain amount."""
        new_len = self.curr_len + amount
        if new_len > self.max_seq_len:
            raise ValueError("Cannot advance KV cache beyond maximum sequence length.")
        self.curr_len = new_len

    def memory_bytes(self) -> int:
        """Calculate total memory usage of the KV cache."""
        #sum of k_pages + v_pages size
        k_bytes = sum(page.numel() * page.element_size() for page in self.k_pages)
        v_bytes = sum(page.numel() * page.element_size() for page in self.v_pages)
        return k_bytes + v_bytes
    
    def fragmentation_ratio(self) -> float:
        """Calculate the fragmentation ratio of the cache."""
        used_pages = self._num_used_pages()
        total_pages = self.max_pages
        return used_pages / total_pages if total_pages > 0 else 0.0
    
