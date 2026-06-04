import pytest
import torch

from src.model_core.config import ModelConfig
from src.kv_cache.contiguous_cache import ContiguousKVCache

N_LAYERS = 2
N_HEADS  = 2
D_MODEL  = 16
MAX_SEQ  = 32
SEQ_ID   = 0


@pytest.fixture
def cfg():
    return ModelConfig(n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL)


@pytest.fixture
def cache(cfg):
    # float32 so write/read round-trips compare exactly
    return ContiguousKVCache(cfg, max_seq_len=MAX_SEQ, device="cpu", dtype=torch.float32)


def rand_kv(T, head_dim):
    # per-sequence layout: [n_heads, T, head_dim] (no batch dim)
    k = torch.randn(N_HEADS, T, head_dim)
    v = torch.randn(N_HEADS, T, head_dim)
    return k, v


# --- write / read ---

def test_write_and_read_back(cache, cfg):
    k, v = rand_kv(4, cfg.head_dim)
    cache.write(0, SEQ_ID, k, v)
    k_out, v_out = cache.read(0, SEQ_ID)
    assert k_out.shape == (N_HEADS, 4, cfg.head_dim)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_read_sees_current_forward_before_advance(cache, cfg):
    # read() must reflect tokens written in the same forward, even though
    # advance() (which drives logical seq_len) is only called afterwards.
    k, v = rand_kv(4, cfg.head_dim)
    cache.write(0, SEQ_ID, k, v)
    k_out, _ = cache.read(0, SEQ_ID)
    assert k_out.shape[1] == 4   # not 0


def test_sequential_writes(cache, cfg):
    k1, v1 = rand_kv(4, cfg.head_dim)
    k2, v2 = rand_kv(2, cfg.head_dim)
    cache.write(0, SEQ_ID, k1, v1)
    cache.advance(SEQ_ID, 4)            # logical length advances between forwards
    cache.write(0, SEQ_ID, k2, v2)      # writes at position 4
    k_out, v_out = cache.read(0, SEQ_ID)
    assert k_out.shape[1] == 6
    assert torch.allclose(k_out[:, :4, :], k1)
    assert torch.allclose(k_out[:, 4:, :], k2)
    assert torch.allclose(v_out[:, :4, :], v1)
    assert torch.allclose(v_out[:, 4:, :], v2)


def test_layers_are_independent(cache, cfg):
    k0, v0 = rand_kv(4, cfg.head_dim)
    k1, v1 = rand_kv(4, cfg.head_dim)
    cache.write(0, SEQ_ID, k0, v0)
    cache.write(1, SEQ_ID, k1, v1)
    k_out0, _ = cache.read(0, SEQ_ID)
    k_out1, _ = cache.read(1, SEQ_ID)
    assert torch.allclose(k_out0, k0)
    assert torch.allclose(k_out1, k1)


def test_sequences_are_independent(cache, cfg):
    k0, v0 = rand_kv(4, cfg.head_dim)
    k1, v1 = rand_kv(4, cfg.head_dim)
    cache.write(0, 0, k0, v0)
    cache.write(0, 1, k1, v1)
    assert torch.allclose(cache.read(0, 0)[0], k0)
    assert torch.allclose(cache.read(0, 1)[0], k1)


def test_write_out_of_bounds_raises(cache, cfg):
    k, v = rand_kv(MAX_SEQ + 1, cfg.head_dim)
    with pytest.raises(RuntimeError):
        cache.write(0, SEQ_ID, k, v)


# --- advance ---

def test_advance_increments_logical_length(cache, cfg):
    cache.write(0, SEQ_ID, *rand_kv(4, cfg.head_dim))   # _ensure inits seq_len=0
    assert cache.seq_len[SEQ_ID] == 0
    cache.advance(SEQ_ID, 4)
    assert cache.seq_len[SEQ_ID] == 4
    cache.advance(SEQ_ID)                                 # default amount=1
    assert cache.seq_len[SEQ_ID] == 5


# --- free ---

def test_free_releases_sequence(cache, cfg):
    cache.write(0, SEQ_ID, *rand_kv(4, cfg.head_dim))
    cache.free(SEQ_ID)
    assert SEQ_ID not in cache.k
    assert SEQ_ID not in cache.seq_len
    with pytest.raises(KeyError):
        cache.read(0, SEQ_ID)


# --- reset ---

def test_reset_clears_all(cache, cfg):
    cache.write(0, 0, *rand_kv(4, cfg.head_dim))
    cache.write(0, 1, *rand_kv(4, cfg.head_dim))
    cache.advance(0, 4)
    cache.reset()
    assert len(cache.k) == 0
    assert len(cache.v) == 0
    assert len(cache.seq_len) == 0
    assert cache.memory_bytes() == 0


# --- memory_bytes (reserved slots, fp16 accounting) ---

def test_memory_bytes_empty(cache):
    assert cache.memory_bytes() == 0


def test_memory_bytes_counts_live_sequences(cache, cfg):
    # reserved per seq: n_layers * n_heads * max_seq_len * head_dim slots,
    # @ 2 bytes (fp16 accounting), times 2 for k + v.
    per_seq = N_LAYERS * N_HEADS * MAX_SEQ * cfg.head_dim * 2
    cache.write(0, 0, *rand_kv(4, cfg.head_dim))
    assert cache.memory_bytes() == 2 * 1 * per_seq
    cache.write(0, 1, *rand_kv(4, cfg.head_dim))
    assert cache.memory_bytes() == 2 * 2 * per_seq


# --- fragmentation_ratio (reserved-but-unused, by logical length) ---

def test_fragmentation_ratio_unknown_seq(cache):
    assert cache.fragmentation_ratio(SEQ_ID) == pytest.approx(0.0)


def test_fragmentation_ratio_empty_after_write(cache, cfg):
    cache.write(0, SEQ_ID, *rand_kv(4, cfg.head_dim))   # seq_len still 0
    assert cache.fragmentation_ratio(SEQ_ID) == pytest.approx(1.0)


def test_fragmentation_ratio_half_used(cache, cfg):
    cache.write(0, SEQ_ID, *rand_kv(4, cfg.head_dim))
    cache.advance(SEQ_ID, MAX_SEQ // 2)
    assert cache.fragmentation_ratio(SEQ_ID) == pytest.approx(0.5)


def test_fragmentation_ratio_full(cache, cfg):
    cache.write(0, SEQ_ID, *rand_kv(4, cfg.head_dim))
    cache.advance(SEQ_ID, MAX_SEQ)
    assert cache.fragmentation_ratio(SEQ_ID) == pytest.approx(0.0)
