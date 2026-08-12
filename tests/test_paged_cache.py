"""Unit tests for PagedKVCache under the batched interface."""
import pytest
import torch

from src.kv_cache.paged import PagedKVCache

N_LAYERS, N_HEADS, HEAD_DIM = 2, 2, 8
PAGE_SIZE, NUM_PAGES, BATCH = 4, 16, 2


@pytest.fixture
def cache():
    return PagedKVCache(n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=HEAD_DIM,
                        block_size=PAGE_SIZE, num_blocks=NUM_PAGES,
                        batch_size=BATCH, device="cpu", dtype=torch.float32)


def rand_kv(T):
    return torch.randn(BATCH, N_HEADS, T, HEAD_DIM), torch.randn(BATCH, N_HEADS, T, HEAD_DIM)


def test_write_and_read_back(cache):
    k, v = rand_kv(4)
    cache.write(0, k, v)
    k_out, v_out = cache.read(0)
    assert k_out.shape == (BATCH, N_HEADS, 4, HEAD_DIM)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_write_spanning_page_boundary(cache):
    k, v = rand_kv(6)          # 6 tokens at page_size 4 -> 2 pages
    cache.write(0, k, v)
    k_out, _ = cache.read(0)
    assert k_out.shape[2] == 6
    assert torch.allclose(k_out, k)
    assert cache.page_table.shape == (BATCH, 2)


def test_sequential_writes_accumulate(cache):
    k1, v1 = rand_kv(4)
    k2, v2 = rand_kv(3)
    cache.write(0, k1, v1)
    cache.advance(4)
    cache.write(0, k2, v2)
    k_out, _ = cache.read(0)
    assert k_out.shape[2] == 7
    assert torch.allclose(k_out[:, :, :4, :], k1)
    assert torch.allclose(k_out[:, :, 4:, :], k2)


def test_layers_are_independent(cache):
    k0, v0 = rand_kv(4)
    k1, v1 = rand_kv(4)
    cache.write(0, k0, v0)
    cache.write(1, k1, v1)
    assert torch.allclose(cache.read(0)[0], k0)
    assert torch.allclose(cache.read(1)[0], k1)


def test_sequences_are_independent(cache):
    """Batch elements must not alias. The vectorised gather makes this a real
    failure mode; the old per-seq loop made it impossible by construction."""
    k = torch.arange(BATCH, dtype=torch.float32).view(BATCH, 1, 1, 1).expand(
        BATCH, N_HEADS, 4, HEAD_DIM).contiguous()
    cache.write(0, k, k)
    out, _ = cache.read(0)
    for b in range(BATCH):
        assert torch.all(out[b] == b)


def test_overflow_raises(cache):
    # capacity = NUM_PAGES pages shared across BATCH seqs -> 8 pages each = 32 tokens
    with pytest.raises(RuntimeError, match="OOM"):
        cache.write(0, *rand_kv(NUM_PAGES * PAGE_SIZE))


def test_free_releases_pages(cache):
    cache.write(0, *rand_kv(6))
    cache.advance(6)
    assert cache.memory_bytes() > 0
    cache.free()
    assert cache.memory_bytes() == 0
    assert cache.page_table.shape == (BATCH, 0)
    assert cache.seq_len == 0


def test_reset_clears_state(cache):
    cache.write(0, *rand_kv(4))
    cache.advance(4)
    cache.reset()
    assert cache.memory_bytes() == 0
    assert cache.seq_len == 0
    assert cache.n_pages == 0


def test_memory_bytes_empty(cache):
    assert cache.memory_bytes() == 0


def test_memory_bytes_counts_held_pages(cache):
    per_page = N_LAYERS * N_HEADS * PAGE_SIZE * HEAD_DIM * cache.k_pages.element_size()
    cache.write(0, *rand_kv(6))          # 2 pages per sequence
    assert cache.memory_bytes() == 2 * (2 * BATCH) * per_page   # (k+v) * pages * per_page


@pytest.mark.parametrize("seq_len,expected", [
    (0, 0.0),        # nothing written -> no pages held -> 0 by convention
    (4, 0.0),        # exactly one full page
    (5, 3 / 8),      # 2 pages (8 slots) allocated, 5 written
    (6, 2 / 8),      # was 0.50 under last-block: overstated by n_pages (2x)
    (8, 0.0),        # exactly two full pages
])
def test_fragmentation_ratio_is_waste_over_allocation(cache, seq_len, expected):
    if seq_len:
        cache.write(0, *rand_kv(seq_len))
    assert cache.fragmentation_ratio() == pytest.approx(expected)


def test_fragmentation_ratio_tracks_writes_not_advance(cache):
    """Allocation is driven by write(); advance() only moves the logical cursor.
    Reading seq_len here would cross counters, so advance() must not move the
    ratio. (Under the old last-block definition this went 0.0 -> 0.5.)"""
    cache.write(0, *rand_kv(6))
    before = cache.fragmentation_ratio()
    cache.advance(6)
    assert cache.fragmentation_ratio() == pytest.approx(before)
