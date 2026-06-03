"""
Verifies that PagedKVCache and ContiguousKVCache produce identical outputs
for the same inputs.

PREREQUISITE — two bugs in paged.py must be fixed before these tests pass:

  1. PagedKVCache.write() must NOT call self.advance() internally.
     The model calls advance() once after all layers write; if write() also
     calls it, curr_len grows n_layers times too fast and each layer writes
     to the wrong position.

  2. PagedKVCache must expose a `current_seq_len` property (or rename
     curr_len). model.py reads kv_cache.current_seq_len to compute start_pos.

Fix for bug 1 — remove the last line from write():
    def write(self, layer_idx, k, v, start_pos):
        ...
        self._write_tokens(self.k_pages, layer_idx, k)
        self._write_tokens(self.v_pages, layer_idx, v)
        # self.advance(t_new)   <-- delete this line

Fix for bug 2 — add a property to PagedKVCache:
    @property
    def current_seq_len(self):
        return self.curr_len
"""

import pytest
import torch

from src.model_core.config import ModelConfig
from src.kv_cache.contiguous_cache import ContiguousKVCache
from src.kv_cache.paged import PagedKVCache

N_LAYERS  = 2
N_HEADS   = 2
D_MODEL   = 16
BATCH     = 1
PAGE_SIZE = 4
NUM_PAGES = 8   # max_seq_len = 32


@pytest.fixture
def cfg():
    return ModelConfig(n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL)


@pytest.fixture
def contiguous(cfg):
    return ContiguousKVCache(cfg, batch_size=BATCH, max_seq_len=NUM_PAGES * PAGE_SIZE, device="cpu")


@pytest.fixture
def paged(cfg):
    return PagedKVCache(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dim_head=cfg.head_dim,
        block_size=PAGE_SIZE,
        num_blocks=NUM_PAGES,
        device="cpu",
        batch_size=BATCH,
    )


def rand_kv_all_layers(T, head_dim):
    return [
        (torch.randn(BATCH, N_HEADS, T, head_dim),
         torch.randn(BATCH, N_HEADS, T, head_dim))
        for _ in range(N_LAYERS)
    ]


def forward_step(cache, kvs, start_pos):
    """Mirrors what model.py does: write all layers, then advance once."""
    T = kvs[0][0].size(2)
    for layer_idx, (k, v) in enumerate(kvs):
        cache.write(layer_idx, k, v, start_pos)
    cache.advance(T)


# --- equivalence tests ---

def test_prefill_outputs_match(contiguous, paged, cfg):
    T = 6
    kvs = rand_kv_all_layers(T, cfg.head_dim)
    forward_step(contiguous, kvs, start_pos=0)
    forward_step(paged,      kvs, start_pos=0)

    for layer_idx, (k, v) in enumerate(kvs):
        k_c, v_c = contiguous.read(layer_idx, upto_pos=T)
        k_p, v_p = paged.read(layer_idx,      upto_pos=T)
        assert torch.allclose(k_c, k_p), f"K mismatch at layer {layer_idx}"
        assert torch.allclose(v_c, v_p), f"V mismatch at layer {layer_idx}"


def test_prefill_then_decode_outputs_match(contiguous, paged, cfg):
    prefill_len = 6
    kvs_pre = rand_kv_all_layers(prefill_len, cfg.head_dim)
    forward_step(contiguous, kvs_pre, start_pos=0)
    forward_step(paged,      kvs_pre, start_pos=0)

    kvs_dec = rand_kv_all_layers(1, cfg.head_dim)
    forward_step(contiguous, kvs_dec, start_pos=prefill_len)
    forward_step(paged,      kvs_dec, start_pos=prefill_len)

    total = prefill_len + 1
    for layer_idx in range(N_LAYERS):
        k_c, v_c = contiguous.read(layer_idx, upto_pos=total)
        k_p, v_p = paged.read(layer_idx,      upto_pos=total)
        assert torch.allclose(k_c, k_p), f"K mismatch at layer {layer_idx}"
        assert torch.allclose(v_c, v_p), f"V mismatch at layer {layer_idx}"


def test_reset_then_reuse_matches(contiguous, paged, cfg):
    kvs = rand_kv_all_layers(4, cfg.head_dim)
    forward_step(contiguous, kvs, start_pos=0)
    forward_step(paged,      kvs, start_pos=0)

    contiguous.reset()
    paged.reset()

    kvs2 = rand_kv_all_layers(4, cfg.head_dim)
    forward_step(contiguous, kvs2, start_pos=0)
    forward_step(paged,      kvs2, start_pos=0)

    for layer_idx, (k, v) in enumerate(kvs2):
        k_c, v_c = contiguous.read(layer_idx, upto_pos=4)
        k_p, v_p = paged.read(layer_idx,      upto_pos=4)
        assert torch.allclose(k_c, k_p), f"K mismatch at layer {layer_idx} after reset"
        assert torch.allclose(v_c, v_p), f"V mismatch at layer {layer_idx} after reset"
