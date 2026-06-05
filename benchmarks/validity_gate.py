"""Validity gate for the cache benchmarks.

Root problem in Exp 1: the toy 4L/256d model was overhead-bound, so per-step
decode time (TPOT) did not grow with context length. Cache-layout effects are
buried under fixed Python/launch overhead and any throughput "crossover" is noise.

This gate is the go/no-go check before trusting any full sweep:
sweep seq_len at batch=1 with a fixed decode length and confirm TPOT rises
MONOTONICALLY with seq_len. If it does, decode is attention-bound and the
comparison is meaningful. If it stays flat, scale the model further.
"""
import statistics

import torch

from src.inference.controller import InferenceController
from src.kv_cache.contiguous_cache import ContiguousKVCache
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel

# --- GPT-2 small: 12L / 12H / 768d (~124M params) ---
N_LAYERS, N_HEADS, D_MODEL, D_FF = 12, 12, 768, 3072

# push context long enough that attention's O(seq_len) term dominates the
# fixed per-step MLP/projection cost
SEQ_LENS   = [512, 1024, 2048, 4096, 8192]
BATCH      = 1
DECODE_LEN = 60
N_WARMUP   = 3
N_MEASURE  = 12


def measure_tpot(model, config, prompt_ids):
    """Return list of per-run steady-state TPOT (ms, p50 per-step) over fresh runs.

    p50 per-step (not whole-run mean) isolates steady-state decode compute from
    the sampling-only first step and one-off allocation stalls -> far lower CV.
    """
    tpots = []
    for _ in range(N_MEASURE):
        cache = ContiguousKVCache(config, config.max_seq_len, prompt_ids.device,
                                  dtype=next(model.parameters()).dtype)
        ctrl = InferenceController(model, config)
        res = ctrl.generate(prompt_ids, max_new_tokens=DECODE_LEN, cache=cache, use_cache=True)
        tpots.append(res.metrics.tpot_p50_ms)
    return tpots


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}  |  model: {N_LAYERS}L/{N_HEADS}H/{D_MODEL}d\n")

    rows = []
    for seq_len in SEQ_LENS:
        max_seq_len = seq_len + DECODE_LEN
        config = ModelConfig(max_seq_len=max_seq_len, n_layers=N_LAYERS,
                             n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF)
        model = GPTModel(config).to(device)
        prompt_ids = torch.randint(0, config.vocab_size, (BATCH, seq_len), device=device)

        for _ in range(N_WARMUP):
            cache = ContiguousKVCache(config, max_seq_len, device,
                                      dtype=next(model.parameters()).dtype)
            InferenceController(model, config).generate(
                prompt_ids, max_new_tokens=DECODE_LEN, cache=cache, use_cache=True)

        tpots = measure_tpot(model, config, prompt_ids)
        median = statistics.median(tpots)
        cv = statistics.pstdev(tpots) / median * 100 if median else 0.0
        rows.append((seq_len, median, cv))
        print(f"seq_len={seq_len:>5}  TPOT={median:7.3f} ms  CV={cv:5.2f}%")

    tpot_series = [r[1] for r in rows]
    monotonic = all(b > a for a, b in zip(tpot_series, tpot_series[1:]))
    worst_cv  = max(r[2] for r in rows)

    print("\n--- VALIDITY GATE ---")
    print(f"TPOT monotonic in seq_len : {'PASS' if monotonic else 'FAIL'}")
    print(f"worst-case CV             : {worst_cv:.2f}%  ({'ok' if worst_cv < 3 else 'raise N_MEASURE'})")
    if not monotonic:
        print(">> Still overhead-bound. Scale the model up before any full sweep.")
    else:
        print(">> Attention-bound. Safe to proceed to the full cache sweep.")


if __name__ == "__main__":
    main()
