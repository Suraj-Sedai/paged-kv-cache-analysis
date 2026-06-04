import pytest
import torch

from src.model_core.config import ModelConfig
from src.kv_cache.contiguous_cache import ContiguousKVCache
from src.kv_cache.paged import PagedKVCache

N_LAYERS  = 2
N_HEADS   = 2
D_MODEL   = 16
PAGE_SIZE = 4
NUM_PAGES = 8   # capacity = 32 tokens
SEQ_ID    = 0


@pytest.fixture
def cfg():
    return ModelConfig(n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL)


@pytest.fixture
def contiguous(cfg):
    # float32 so reads compare exactly against paged's live tensors
    return ContiguousKVCache(cfg, max_seq_len=NUM_PAGES * PAGE_SIZE, device="cpu", dtype=torch.float32)


@pytest.fixture
def paged(cfg):
    return PagedKVCache(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dim_head=cfg.head_dim,
        block_size=PAGE_SIZE,
        num_blocks=NUM_PAGES,
        device="cpu",
        dtype=torch.float16,
    )


def rand_kv_all_layers(T, head_dim):
    # per-sequence layout: [n_heads, T, head_dim]
    return [
        (torch.randn(N_HEADS, T, head_dim),
         torch.randn(N_HEADS, T, head_dim))
        for _ in range(N_LAYERS)
    ]


def forward_step(cache, kvs, seq_id):
    """Mirrors what model.py does: write all layers, then advance once."""
    T = kvs[0][0].size(1)   # [n_heads, T, head_dim]
    for layer_idx, (k, v) in enumerate(kvs):
        cache.write(layer_idx, seq_id, k, v)
    cache.advance(seq_id, T)


# --- equivalence tests ---

def test_prefill_outputs_match(contiguous, paged, cfg):
    kvs = rand_kv_all_layers(6, cfg.head_dim)
    forward_step(contiguous, kvs, SEQ_ID)
    forward_step(paged,      kvs, SEQ_ID)

    for layer_idx in range(N_LAYERS):
        k_c, v_c = contiguous.read(layer_idx, SEQ_ID)
        k_p, v_p = paged.read(layer_idx,      SEQ_ID)
        assert torch.allclose(k_c, k_p), f"K mismatch at layer {layer_idx}"
        assert torch.allclose(v_c, v_p), f"V mismatch at layer {layer_idx}"


def test_prefill_then_decode_outputs_match(contiguous, paged, cfg):
    kvs_pre = rand_kv_all_layers(6, cfg.head_dim)
    forward_step(contiguous, kvs_pre, SEQ_ID)
    forward_step(paged,      kvs_pre, SEQ_ID)

    kvs_dec = rand_kv_all_layers(1, cfg.head_dim)
    forward_step(contiguous, kvs_dec, SEQ_ID)
    forward_step(paged,      kvs_dec, SEQ_ID)

    for layer_idx in range(N_LAYERS):
        k_c, v_c = contiguous.read(layer_idx, SEQ_ID)
        k_p, v_p = paged.read(layer_idx,      SEQ_ID)
        assert k_c.shape[1] == 7
        assert torch.allclose(k_c, k_p), f"K mismatch at layer {layer_idx}"
        assert torch.allclose(v_c, v_p), f"V mismatch at layer {layer_idx}"


def test_reset_then_reuse_matches(contiguous, paged, cfg):
    kvs = rand_kv_all_layers(4, cfg.head_dim)
    forward_step(contiguous, kvs, SEQ_ID)
    forward_step(paged,      kvs, SEQ_ID)

    contiguous.reset()
    paged.reset()

    kvs2 = rand_kv_all_layers(4, cfg.head_dim)
    forward_step(contiguous, kvs2, SEQ_ID)
    forward_step(paged,      kvs2, SEQ_ID)

    for layer_idx in range(N_LAYERS):
        k_c, v_c = contiguous.read(layer_idx, SEQ_ID)
        k_p, v_p = paged.read(layer_idx,      SEQ_ID)
        assert torch.allclose(k_c, k_p), f"K mismatch at layer {layer_idx} after reset"
        assert torch.allclose(v_c, v_p), f"V mismatch at layer {layer_idx} after reset"
