import pytest
import torch

from src.kv_cache.paged import PagedKVCache

N_LAYERS  = 2
N_HEADS   = 2
HEAD_DIM  = 8
PAGE_SIZE = 4
NUM_PAGES = 8   # shared pool capacity = 32 tokens
SEQ_ID    = 0


@pytest.fixture
def cache():
    # dtype must be float16 (page storage); read() returns the live tensors,
    # which keep the written dtype, so float32 inputs round-trip exactly.
    return PagedKVCache(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dim_head=HEAD_DIM,
        block_size=PAGE_SIZE,
        num_blocks=NUM_PAGES,
        device="cpu",
        dtype=torch.float16,
    )


def rand_kv(T):
    # per-sequence layout: [n_heads, T, head_dim] (no batch dim)
    k = torch.randn(N_HEADS, T, HEAD_DIM)
    v = torch.randn(N_HEADS, T, HEAD_DIM)
    return k, v


# --- write / read ---

def test_write_and_read_back(cache):
    k, v = rand_kv(4)
    cache.write(0, SEQ_ID, k, v)
    k_out, v_out = cache.read(0, SEQ_ID)
    assert k_out.shape == (N_HEADS, 4, HEAD_DIM)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_write_spanning_page_boundary(cache):
    # 6 tokens cross the first page boundary (page_size=4) -> 2 pages
    k, v = rand_kv(6)
    cache.write(0, SEQ_ID, k, v)
    k_out, v_out = cache.read(0, SEQ_ID)
    assert k_out.shape[1] == 6
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)
    assert len(cache.page_table[SEQ_ID]) == 2


def test_sequential_writes_accumulate(cache):
    k1, v1 = rand_kv(4)
    k2, v2 = rand_kv(2)
    cache.write(0, SEQ_ID, k1, v1)
    cache.advance(SEQ_ID, 4)
    cache.write(0, SEQ_ID, k2, v2)
    k_out, _ = cache.read(0, SEQ_ID)
    assert k_out.shape[1] == 6
    assert torch.allclose(k_out[:, :4, :], k1)
    assert torch.allclose(k_out[:, 4:, :], k2)


def test_sequences_are_independent(cache):
    k0, v0 = rand_kv(4)
    k1, v1 = rand_kv(4)
    cache.write(0, 0, k0, v0)
    cache.write(0, 1, k1, v1)
    assert torch.allclose(cache.read(0, 0)[0], k0)
    assert torch.allclose(cache.read(0, 1)[0], k1)


def test_overflow_raises(cache):
    # pool holds NUM_PAGES*PAGE_SIZE tokens; one more needs a 9th page
    k, v = rand_kv(NUM_PAGES * PAGE_SIZE + 1)
    with pytest.raises(RuntimeError):
        cache.write(0, SEQ_ID, k, v)


# --- free ---

def test_free_returns_pages_to_pool(cache):
    k, v = rand_kv(6)            # 2 pages
    cache.write(0, SEQ_ID, k, v)
    assert len(cache.free_pages) == NUM_PAGES - 2
    cache.free(SEQ_ID)
    assert len(cache.free_pages) == NUM_PAGES
    assert SEQ_ID not in cache.page_table
    with pytest.raises(KeyError):
        cache.read(0, SEQ_ID)


# --- reset ---

def test_reset_clears_state(cache):
    cache.write(0, SEQ_ID, *rand_kv(4))
    cache.advance(SEQ_ID, 4)
    cache.reset()
    assert len(cache.free_pages) == NUM_PAGES
    assert cache.page_table == {}
    assert cache.seq_len == {}
    assert cache.memory_bytes() == 0


# --- memory_bytes (physical pages held, fp16) ---

def test_memory_bytes_empty(cache):
    assert cache.memory_bytes() == 0


def test_memory_bytes_counts_held_pages(cache):
    # per page: n_layers * n_heads * page_size * head_dim slots @ 2 bytes (fp16),
    # times 2 for k + v pools.
    per_page = N_LAYERS * N_HEADS * PAGE_SIZE * HEAD_DIM * 2
    cache.write(0, SEQ_ID, *rand_kv(6))   # 2 pages held
    assert cache.memory_bytes() == 2 * 2 * per_page


# --- fragmentation_ratio (last-page internal waste, by logical length) ---

def test_fragmentation_ratio_empty(cache):
    assert cache.fragmentation_ratio(SEQ_ID) == pytest.approx(0.0)


def test_fragmentation_ratio_full_page(cache):
    cache.write(0, SEQ_ID, *rand_kv(4))
    cache.advance(SEQ_ID, 4)            # exactly one full page -> no waste
    assert cache.fragmentation_ratio(SEQ_ID) == pytest.approx(0.0)


def test_fragmentation_ratio_partial_page(cache):
    cache.write(0, SEQ_ID, *rand_kv(6))
    cache.advance(SEQ_ID, 6)           # 6 % 4 = 2 -> last page half empty
    assert cache.fragmentation_ratio(SEQ_ID) == pytest.approx(0.5)
