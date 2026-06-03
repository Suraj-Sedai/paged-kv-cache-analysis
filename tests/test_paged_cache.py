import pytest
import torch

from src.kv_cache.paged import PagedKVCache

N_LAYERS  = 2
N_HEADS   = 2
HEAD_DIM  = 8
BATCH     = 1
PAGE_SIZE = 4
NUM_PAGES = 8   # max_seq_len = 32


@pytest.fixture
def cache():
    return PagedKVCache(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dim_head=HEAD_DIM,
        block_size=PAGE_SIZE,
        num_blocks=NUM_PAGES,
        device="cpu",
        batch_size=BATCH,
    )


def rand_kv(T):
    k = torch.randn(BATCH, N_HEADS, T, HEAD_DIM)
    v = torch.randn(BATCH, N_HEADS, T, HEAD_DIM)
    return k, v


# --- write / read ---

def test_write_and_read_back(cache):
    k, v = rand_kv(4)
    cache.write(0, k, v, start_pos=0)
    k_out, v_out = cache.read(0, upto_pos=4)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_write_spanning_page_boundary(cache):
    # 6 tokens crosses the first page boundary (page_size=4)
    k, v = rand_kv(6)
    cache.write(0, k, v, start_pos=0)
    k_out, v_out = cache.read(0, upto_pos=6)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_read_empty_returns_zero_length_tensor(cache):
    k_out, v_out = cache.read(0, upto_pos=0)
    assert k_out.shape == (BATCH, N_HEADS, 0, HEAD_DIM)
    assert v_out.shape == (BATCH, N_HEADS, 0, HEAD_DIM)


def test_overflow_raises(cache):
    k, v = rand_kv(NUM_PAGES * PAGE_SIZE + 1)
    with pytest.raises(ValueError):
        cache.write(0, k, v, start_pos=0)


# --- reset ---

def test_reset_clears_state(cache):
    k, v = rand_kv(4)
    cache.write(0, k, v, start_pos=0)
    cache.reset()
    assert cache.curr_len == 0
    assert all(p is None for p in cache.page_table[0])
    assert len(cache.free_pages[0]) == NUM_PAGES


# --- memory_bytes ---

def test_memory_bytes(cache):
    # k_pages + v_pages: each is n_layers tensors of shape [num_pages, batch, n_heads, page_size, head_dim]
    expected = 2 * N_LAYERS * NUM_PAGES * BATCH * N_HEADS * PAGE_SIZE * HEAD_DIM * 4
    assert cache.memory_bytes() == expected


# --- fragmentation_ratio ---

def test_fragmentation_ratio_empty(cache):
    assert cache.fragmentation_ratio() == pytest.approx(0.0)


def test_fragmentation_ratio_one_page(cache):
    k, v = rand_kv(4)
    cache.write(0, k, v, start_pos=0)
    cache.advance(4)
    # 1 page used out of NUM_PAGES
    assert cache.fragmentation_ratio() == pytest.approx(1 / NUM_PAGES)


def test_fragmentation_ratio_full(cache):
    k, v = rand_kv(NUM_PAGES * PAGE_SIZE)
    cache.write(0, k, v, start_pos=0)
    cache.advance(NUM_PAGES * PAGE_SIZE)
    assert cache.fragmentation_ratio() == pytest.approx(1.0)
