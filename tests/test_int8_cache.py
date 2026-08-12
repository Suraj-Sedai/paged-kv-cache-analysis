"""Unit tests for INT8KVCache under the batched interface.

Two things are pinned here because both are load-bearing for H3:
  - the memory ratio (1 + 2/head_dim)/2 against an FP16 baseline
  - the quantization error bound, which must include the fp16 scale rounding
"""
import pytest
import torch

from src.kv_cache.paged import PagedKVCache
from src.kv_cache.int8_paged import INT8KVCache

N_LAYERS, N_HEADS, HEAD_DIM = 2, 2, 64
PAGE_SIZE, NUM_PAGES, BATCH = 4, 16, 2


@pytest.fixture
def cache():
    return INT8KVCache(n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=HEAD_DIM,
                       block_size=PAGE_SIZE, num_blocks=NUM_PAGES,
                       batch_size=BATCH, device="cpu", compute_dtype=torch.float32)


@pytest.fixture
def fp16_paged():
    return PagedKVCache(n_layers=N_LAYERS, n_heads=N_HEADS, dim_head=HEAD_DIM,
                        block_size=PAGE_SIZE, num_blocks=NUM_PAGES,
                        batch_size=BATCH, device="cpu", dtype=torch.float16)


def rand_kv(T):
    return torch.randn(BATCH, N_HEADS, T, HEAD_DIM), torch.randn(BATCH, N_HEADS, T, HEAD_DIM)


def quant_tolerance(ref):
    """Rounding to the int8 grid (<= scale/2 = amax/254) PLUS the fp16 rounding
    of the stored scale, which costs another ~2^-11 relative at large codes."""
    amax = ref.abs().amax(dim=-1, keepdim=True)
    return amax / 254.0 + amax * 2 ** -10


def test_roundtrip_within_quant_error(cache):
    k, v = rand_kv(6)
    cache.write(0, k, v)
    k_out, v_out = cache.read(0)
    assert k_out.shape == (BATCH, N_HEADS, 6, HEAD_DIM)
    assert torch.all((k_out - k).abs() <= quant_tolerance(k))
    assert torch.all((v_out - v).abs() <= quant_tolerance(v))


def test_all_zero_tokens_do_not_produce_nan(cache):
    """scale = amax/127 is 0 for an all-zero token; the clamp guard must hold."""
    z = torch.zeros(BATCH, N_HEADS, 4, HEAD_DIM)
    cache.write(0, z, z)
    k_out, _ = cache.read(0)
    assert torch.all(k_out == 0)
    assert not torch.isnan(k_out).any()


def test_write_spanning_page_boundary(cache):
    k, v = rand_kv(6)
    cache.write(0, k, v)
    k_out, _ = cache.read(0)
    assert cache.page_table.shape == (BATCH, 2)
    assert torch.all((k_out - k).abs() <= quant_tolerance(k))


def test_sequential_writes_accumulate(cache):
    k1, v1 = rand_kv(4)
    k2, v2 = rand_kv(3)
    cache.write(0, k1, v1)
    cache.advance(4)
    cache.write(0, k2, v2)
    k_out, _ = cache.read(0)
    assert k_out.shape[2] == 7
    assert torch.all((k_out[:, :, :4, :] - k1).abs() <= quant_tolerance(k1))
    assert torch.all((k_out[:, :, 4:, :] - k2).abs() <= quant_tolerance(k2))


def test_sequences_are_independent(cache):
    k = torch.arange(1, BATCH + 1, dtype=torch.float32).view(BATCH, 1, 1, 1).expand(
        BATCH, N_HEADS, 4, HEAD_DIM).contiguous()
    cache.write(0, k, k)
    out, _ = cache.read(0)
    for b in range(BATCH):
        assert torch.allclose(out[b], k[b], atol=1e-3)


def test_memory_ratio_against_fp16(cache, fp16_paged):
    """The H3 lever. Ratio is defined against FP16 -- comparing to an FP32 pool
    silently returns a different number."""
    k, v = rand_kv(6)
    cache.write(0, k, v)
    fp16_paged.write(0, k.half(), v.half())
    expected = (1 + 2 / HEAD_DIM) / 2
    assert cache.memory_bytes() / fp16_paged.memory_bytes() == pytest.approx(expected)
    assert expected == pytest.approx(0.515625)


def test_scales_are_stored_fp16(cache):
    assert cache.k_scales.dtype == torch.float16
    assert cache.k_pages.dtype == torch.int8


def test_free_releases_pages(cache):
    cache.write(0, *rand_kv(6))
    assert cache.memory_bytes() > 0
    cache.free()
    assert cache.memory_bytes() == 0
