"""Unit tests for ContiguousKVCache under the batched interface."""
import pytest
import torch

from src.model_core.config import ModelConfig
from src.kv_cache.contiguous_cache import ContiguousKVCache

N_LAYERS, N_HEADS, D_MODEL = 2, 2, 16
MAX_SEQ, BATCH = 16, 2


@pytest.fixture
def cfg():
    return ModelConfig(n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL)


@pytest.fixture
def cache(cfg):
    return ContiguousKVCache(cfg, batch_size=BATCH, max_seq_len=MAX_SEQ,
                             device="cpu", dtype=torch.float32)


def rand_kv(T, head_dim):
    return torch.randn(BATCH, N_HEADS, T, head_dim), torch.randn(BATCH, N_HEADS, T, head_dim)


def test_write_and_read_back(cache, cfg):
    k, v = rand_kv(4, cfg.head_dim)
    cache.write(0, k, v)
    k_out, v_out = cache.read(0)
    assert k_out.shape == (BATCH, N_HEADS, 4, cfg.head_dim)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_read_sees_current_forward_before_advance(cache, cfg):
    """filled tracks write(); seq_len tracks advance(). read() must use filled,
    or a decode step cannot attend to the token it just wrote."""
    k, v = rand_kv(4, cfg.head_dim)
    cache.write(0, k, v)
    assert cache.read(0)[0].shape[2] == 4
    assert cache.seq_len == 0


def test_sequential_writes_accumulate(cache, cfg):
    k1, v1 = rand_kv(4, cfg.head_dim)
    k2, v2 = rand_kv(2, cfg.head_dim)
    cache.write(0, k1, v1)
    cache.advance(4)
    cache.write(0, k2, v2)
    k_out, _ = cache.read(0)
    assert k_out.shape[2] == 6
    assert torch.allclose(k_out[:, :, :4, :], k1)
    assert torch.allclose(k_out[:, :, 4:, :], k2)


def test_layers_are_independent(cache, cfg):
    k0, v0 = rand_kv(4, cfg.head_dim)
    k1, v1 = rand_kv(4, cfg.head_dim)
    cache.write(0, k0, v0)
    cache.write(1, k1, v1)
    assert torch.allclose(cache.read(0)[0], k0)
    assert torch.allclose(cache.read(1)[0], k1)


def test_sequences_are_independent(cache, cfg):
    k = torch.arange(BATCH, dtype=torch.float32).view(BATCH, 1, 1, 1).expand(
        BATCH, N_HEADS, 4, cfg.head_dim).contiguous()
    cache.write(0, k, k)
    out, _ = cache.read(0)
    for b in range(BATCH):
        assert torch.all(out[b] == b)


def test_read_returns_a_view_not_a_copy(cache, cfg):
    """The contiguous read path must stay copy-free -- it is the baseline the
    paged gather is measured against."""
    cache.write(0, *rand_kv(4, cfg.head_dim))
    assert cache.read(0)[0]._base is not None


def test_overflow_raises(cache, cfg):
    with pytest.raises(RuntimeError, match="max_seq_len"):
        cache.write(0, *rand_kv(MAX_SEQ + 1, cfg.head_dim))


def test_memory_bytes_is_full_reservation(cache, cfg):
    """Contiguous reserves up front, so memory does not depend on how much of
    the reservation is actually written."""
    expected = N_LAYERS * BATCH * N_HEADS * MAX_SEQ * cfg.head_dim * 4 * 2
    assert cache.memory_bytes() == expected
    cache.write(0, *rand_kv(4, cfg.head_dim))
    cache.advance(4)
    assert cache.memory_bytes() == expected


@pytest.mark.parametrize("seq_len,expected", [(0, 1.0), (8, 0.5), (16, 0.0)])
def test_fragmentation_ratio_is_waste_over_allocation(cache, cfg, seq_len, expected):
    """Values are unchanged by the switch to waste-over-allocation — the
    reservation is a single fixed allocation, so the two definitions coincide.
    The driver changed: written extent, not the advance() cursor. An unwritten
    contiguous cache is 1.0 because the reservation exists from construction."""
    if seq_len:
        cache.write(0, *rand_kv(seq_len, cfg.head_dim))
    assert cache.fragmentation_ratio() == pytest.approx(expected)
