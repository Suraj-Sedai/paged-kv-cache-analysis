import pytest
import torch

from src.model_core.config import ModelConfig
from src.kv_cache.contiguous_cache import ContiguousKVCache

N_LAYERS = 2
N_HEADS  = 2
D_MODEL  = 16
BATCH    = 1
MAX_SEQ  = 32


@pytest.fixture
def cfg():
    return ModelConfig(n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL)


@pytest.fixture
def cache(cfg):
    return ContiguousKVCache(cfg, batch_size=BATCH, max_seq_len=MAX_SEQ, device="cpu")


def rand_kv(T, head_dim):
    k = torch.randn(BATCH, N_HEADS, T, head_dim)
    v = torch.randn(BATCH, N_HEADS, T, head_dim)
    return k, v


# --- write / read ---

def test_write_and_read_back(cache, cfg):
    k, v = rand_kv(4, cfg.head_dim)
    cache.write(0, k, v, start_pos=0)
    k_out, v_out = cache.read(0, upto_pos=4)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_sequential_writes(cache, cfg):
    k1, v1 = rand_kv(4, cfg.head_dim)
    k2, v2 = rand_kv(2, cfg.head_dim)
    cache.write(0, k1, v1, start_pos=0)
    cache.write(0, k2, v2, start_pos=4)
    k_out, v_out = cache.read(0, upto_pos=6)
    assert torch.allclose(k_out[:, :, :4, :], k1)
    assert torch.allclose(k_out[:, :, 4:,  :], k2)
    assert torch.allclose(v_out[:, :, :4, :], v1)
    assert torch.allclose(v_out[:, :, 4:,  :], v2)


def test_layers_are_independent(cache, cfg):
    k0, v0 = rand_kv(4, cfg.head_dim)
    k1, v1 = rand_kv(4, cfg.head_dim)
    cache.write(0, k0, v0, start_pos=0)
    cache.write(1, k1, v1, start_pos=0)
    k_out0, _ = cache.read(0, upto_pos=4)
    k_out1, _ = cache.read(1, upto_pos=4)
    assert torch.allclose(k_out0, k0)
    assert torch.allclose(k_out1, k1)


def test_write_out_of_bounds_raises(cache, cfg):
    k, v = rand_kv(MAX_SEQ + 1, cfg.head_dim)
    with pytest.raises(IndexError):
        cache.write(0, k, v, start_pos=0)


# --- advance ---

def test_advance_increments_position(cache):
    assert cache.current_seq_len == 0
    cache.advance(5)
    assert cache.current_seq_len == 5
    cache.advance(3)
    assert cache.current_seq_len == 8


def test_advance_overflow_raises(cache):
    with pytest.raises(IndexError):
        cache.advance(MAX_SEQ + 1)


# --- reset ---

def test_reset_clears_position_and_data(cache, cfg):
    k, v = rand_kv(4, cfg.head_dim)
    cache.write(0, k, v, start_pos=0)
    cache.advance(4)
    cache.reset()
    assert cache.current_seq_len == 0
    k_out, _ = cache.read(0, upto_pos=4)
    assert torch.all(k_out == 0)


# --- memory_bytes ---

def test_memory_bytes(cache, cfg):
    # k tensor + v tensor, each: [n_layers, batch, n_heads, max_seq, head_dim] @ float32
    expected = 2 * N_LAYERS * BATCH * N_HEADS * MAX_SEQ * cfg.head_dim * 4
    assert cache.memory_bytes() == expected


# --- fragmentation_ratio ---

def test_fragmentation_ratio_empty(cache):
    assert cache.fragmentation_ratio() == pytest.approx(1.0)


def test_fragmentation_ratio_half_used(cache):
    cache.advance(MAX_SEQ // 2)
    assert cache.fragmentation_ratio() == pytest.approx(0.5)


def test_fragmentation_ratio_full(cache):
    cache.advance(MAX_SEQ)
    assert cache.fragmentation_ratio() == pytest.approx(0.0)
