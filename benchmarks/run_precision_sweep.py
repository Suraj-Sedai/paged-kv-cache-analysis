"""Experiment 3a — precision sweep: FP16-paged vs INT8-paged (memory frontier).

The H3 lever is memory. INT8 stores the KV cache at 0.516x the FP16 bytes (int8
data + an fp16 per-token scale tax = (1 + 2/head_dim)/2 = 1.03125/2 at head_dim 64),
so for the same workload it should:
  (1) report a deterministic 0.516x cache_memory_mb  — the GUARANTEED signal, and
  (2) survive (seq_len x batch) regimes where FP16 hits OOM — the frontier shift,
      which is VRAM-dependent and only material insofar as the KV cache is a large
      share of total memory (model + activations + cache).

This sweep is paged-only; precision is the single independent variable (same
block_size, same paging code path), so any difference isolates precision. Decode
on this GPU is launch-bound (Exp 1), so timing columns are reported with CV and
read as noise — memory and OOM status are the clean signals.

Headline = the cache-memory ratio (always valid). OOM frontier = secondary,
GPU-dependent demonstration. If the frontier barely moves on a given GPU, that is
the honest finding (cache didn't dominate memory) and the ratio still stands.
"""
import csv
import os
import statistics
import time

import torch

from src.inference.controller import InferenceController
from src.kv_cache.paged import PagedKVCache
from src.kv_cache.int8_paged import INT8KVCache
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel

N_LAYERS, N_HEADS, D_MODEL, D_FF = 8, 8, 512, 2048   # same architecture as Exp 1/2

CACHE_TYPES = ["paged_fp16", "paged_int8"]
SEQ_LENS    = [512, 1024, 2048, 4096, 8192]          # pushed out to probe the OOM frontier
BATCH_SIZES = [1, 8, 16, 32]
BLOCK_SIZE  = 16                                     # Exp 2 optimum; held constant here
DECODE_LEN  = 60
N_WARMUP    = 3
N_MEASURE   = 20
BUDGET_S    = 20.0   # one warmup slower than this -> cell thrashes under memory pressure -> 'slow'

OUTPUT_PATH = os.path.join("experiments", "results", "exp3_precision_sweep.csv")
FIELDNAMES  = ["cache_type", "seq_len", "batch", "block_size", "status",
               "ttft_ms", "tpot_ms", "throughput", "throughput_cv",
               "peak_memory_mb", "cache_memory_mb", "frag_ratio"]

def make_cache(cache_type, mc, batch, max_seq_len, device, io_dtype):
    """Paged pool sized for all `batch` seqs. FP16 vs INT8 share the paging code
    path; only storage precision differs (compute stays io_dtype on read)."""
    per_seq_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = per_seq_blocks * batch
    common = dict(n_layers=mc.n_layers, n_heads=mc.n_heads, dim_head=mc.head_dim,
                  block_size=BLOCK_SIZE, num_blocks=num_blocks,
                  batch_size=batch, device=device)
    if cache_type == "paged_fp16":
        return PagedKVCache(**common, dtype=io_dtype)
    if cache_type == "paged_int8":
        return INT8KVCache(**common, compute_dtype=io_dtype)
    raise ValueError(cache_type)


def measure_cache_memory_frag(cache_type, mc, batch, max_seq_len, final_len, device, io_dtype):
    """Fill a throwaway cache to final state and read its OWN memory_bytes() /
    fragmentation_ratio() — each cache's accounting is the source of truth, so the
    int8 vs fp16 byte ratio is exact (and includes the int8 scale tax).

    Batched API: one write for the whole batch. Writing layer 0 only is enough —
    a page spans all layers, so memory_bytes() already covers the full pool draw.
    """
    cache = make_cache(cache_type, mc, batch, max_seq_len, device, io_dtype)
    dummy = torch.zeros(batch, mc.n_heads, final_len, mc.head_dim,
                        device=device, dtype=io_dtype)
    cache.write(0, dummy, dummy)
    cache.advance(final_len)
    mem_mb = cache.memory_bytes() / (1024 * 1024)
    frag = cache.fragmentation_ratio()
    cache.free()
    del cache, dummy
    if device == "cuda":
        torch.cuda.empty_cache()
    return round(mem_mb, 2), round(frag, 4)


def run_one(model, mc, cache, prompt_ids):
    controller = InferenceController(model, mc)
    return controller.generate(prompt_ids, max_new_tokens=DECODE_LEN, cache=cache, use_cache=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}  |  model: {N_LAYERS}L/{N_HEADS}H/{D_MODEL}d  |  FP16 weights, paged FP16 vs INT8")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results = []

    for cache_type in CACHE_TYPES:
        for seq_len in SEQ_LENS:
            for batch in BATCH_SIZES:
                max_seq_len = seq_len + DECODE_LEN
                final_len = seq_len + DECODE_LEN
                mc = ModelConfig(max_seq_len=max_seq_len, n_layers=N_LAYERS,
                                 n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF)
                model = GPTModel(mc).to(device).half()   # FP16 weights; cache precision is the variable
                io_dtype = next(model.parameters()).dtype
                prompt_ids = torch.randint(0, mc.vocab_size, (batch, seq_len), device=device)

                base = {"cache_type": cache_type, "seq_len": seq_len, "batch": batch,
                        "block_size": BLOCK_SIZE}
                blank_timing = {"ttft_ms": "", "tpot_ms": "", "throughput": "",
                                "throughput_cv": "", "peak_memory_mb": ""}
                try:
                    cache_mem_mb, frag = measure_cache_memory_frag(
                        cache_type, mc, batch, max_seq_len, final_len, device, io_dtype)
                    base.update(cache_memory_mb=cache_mem_mb, frag_ratio=frag)

                    t0 = time.perf_counter()
                    run_one(model, mc,
                            make_cache(cache_type, mc, batch, max_seq_len, device, io_dtype),
                            prompt_ids)
                    if time.perf_counter() - t0 > BUDGET_S:
                        row = {**base, "status": "slow", **blank_timing}
                    else:
                        for _ in range(N_WARMUP - 1):
                            run_one(model, mc,
                                    make_cache(cache_type, mc, batch, max_seq_len, device, io_dtype),
                                    prompt_ids)
                        metrics_list = []
                        for _ in range(N_MEASURE):
                            cache = make_cache(cache_type, mc, batch, max_seq_len, device, io_dtype)
                            metrics_list.append(run_one(model, mc, cache, prompt_ids).metrics)

                        throughputs = [m.throughput_tokens_per_sec for m in metrics_list]
                        thr_median = statistics.median(throughputs)
                        thr_cv = (statistics.pstdev(throughputs) / thr_median * 100) if thr_median else 0.0
                        row = {
                            **base, "status": "ok",
                            "ttft_ms":    round(statistics.median(m.ttft_ms for m in metrics_list), 3),
                            "tpot_ms":    round(statistics.median(m.tpot_p50_ms for m in metrics_list), 3),
                            "throughput": round(thr_median, 1),
                            "throughput_cv": round(thr_cv, 2),
                            "peak_memory_mb": round(statistics.median(m.peak_memory_mb for m in metrics_list), 2),
                        }
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row = {**base, "status": "oom", **blank_timing,
                           "cache_memory_mb": base.get("cache_memory_mb", ""),
                           "frag_ratio": base.get("frag_ratio", "")}

                results.append(row)
                print(row)
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
