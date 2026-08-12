import csv
import os
import statistics
import subprocess
import time

import torch

from src.inference.controller import InferenceController
from src.kv_cache.contiguous_cache import ContiguousKVCache
from src.kv_cache.paged import PagedKVCache
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel

# moderate model (8L/8H/512d), FP16 weights, SDPA attention, batched cache.
#
# v1 NOTE (now stale, kept for provenance): the per-sequence Python loop in
# attention put decode in a launch-bound regime, so timing columns were noise
# and only memory was trustworthy. The batched interface removes B*n_layers
# cache ops per token. Re-check throughput_cv before trusting any timing claim.
N_LAYERS, N_HEADS, D_MODEL, D_FF = 8, 8, 512, 2048

SEED = 0   # weights, prompts, and sampling are all seeded; cells are comparable

SEQ_LENS    = [128, 256, 512, 1024, 2048]
BATCH_SIZES = [1, 4, 8, 16]
BLOCK_SIZE  = 16
DECODE_LEN  = 60

# GRID ARTIFACT — do not read the frag_ratio column as a finding about length.
#
# Every cell's final length is seq_len + DECODE_LEN, and all five are congruent
# to 12 mod BLOCK_SIZE:
#
#   seq   final  final%16  pages  alloc  waste  frag_ratio
#   128     188     12        12    192      4    0.020833
#   256     316     12        20    320      4    0.012500
#   512     572     12        36    576      4    0.006944
#  1024    1084     12        68   1088      4    0.003676
#  2048    2108     12       132   2112      4    0.001894
#
# Paged waste is the padding to the next page boundary, so a fixed residue pins
# the NUMERATOR at 4 slots for every row. Only the denominator moves. The smooth
# 1/L decay down the column is therefore arithmetic, not a property of paging at
# scale — it is what any constant numerator over a growing allocation looks like.
# The real ratio is a sawtooth in length (188 -> 4/192, 192 -> 0.0, 193 -> 15/208);
# this grid samples one tooth position five times.
#
# Two claims this does NOT support: that fragmentation falls with sequence
# length, and that paging wastes ~2% at short contexts. Both need a grid that
# varies final_len % BLOCK_SIZE. What IS supported: absolute waste is under one
# page per sequence at every length (tests/test_cache_equivalence.py::
# test_paged_absolute_waste_is_under_one_page) and the upper envelope of the
# ratio decays (::test_paged_waste_ratio_envelope_decays).
N_WARMUP    = 3
N_MEASURE   = 20
BUDGET_S    = 20.0   # if one warmup run exceeds this, the cell thrashes under
                     # memory pressure on this GPU -> mark 'slow', skip timing

COMMIT      = "unknown"   # set in main()
OUTPUT_PATH = os.path.join("experiments", "results", "exp1_layout_comparison.csv")
FIELDNAMES  = ["cache_type", "seq_len", "batch", "block_size", "status",
               "ttft_ms", "tpot_ms", "throughput", "throughput_cv",
               "peak_memory_mb", "cache_memory_mb", "frag_ratio", "alloc_basis",
               "dtype", "attn_impl", "commit"]


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_cache(cache_type, model_config, batch_size, max_seq_len, device, io_dtype):
    """Both caches use the model dtype so a comparison isolates layout (not
    precision). Paged gets a shared pool sized for all `batch_size` seqs.

    WITHDRAWN COMPARISON (a decision, not a TODO): max_seq_len here is the EXACT
    final length, so contiguous reservation waste is 0.0 by construction. That
    number is the experimental setup restated, not a measurement, and "contiguous
    fragments less" would be a true number attached to a vacuous comparison. The
    contiguous-vs-paged fragmentation comparison is withdrawn until the trace
    harness supplies a real reservation policy (Oracle / FixedMax / Percentile).

    The frag columns and their basis labels still ship on both sides, so the
    comparison can be reconstructed once reservation is a variable rather than an
    oracle. Nothing here is deleted; it is deferred.

    The paged series stands alone and is valid on its own: absolute waste is
    bounded by one block per sequence at every length, so the ratio is
    O(block_size / L) in the worst case, exactly 0 when block_size divides L, and
    sawtooth in between. It does NOT show fragmentation decaying with sequence
    length — the smooth decay in the exp1 grid is the residue artifact recorded
    at SEQ_LENS above.
    """
    if cache_type == "contiguous":
        return ContiguousKVCache(model_config, batch_size, max_seq_len, device, dtype=io_dtype)
    per_seq_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    return PagedKVCache(
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        dim_head=model_config.head_dim,
        block_size=BLOCK_SIZE,
        num_blocks=per_seq_blocks * batch_size,  # shared pool across all seqs
        batch_size=batch_size,
        device=device,
        dtype=io_dtype,
    )


def measure_cache_memory_frag(cache_type, model_config, batch, max_seq_len, final_len, device, io_dtype):
    """Fill a throwaway cache to the run's final state and read its OWN
    memory_bytes()/fragmentation_ratio() — exact (deterministic, uniform
    lengths) and uses each cache's accounting as the single source of truth.

    Done out-of-band because controller.generate() frees every seq at cleanup,
    so the live cache is empty by the time metrics are reported.
    """
    cache = make_cache(cache_type, model_config, batch, max_seq_len, device, io_dtype)
    dummy = torch.zeros(batch, model_config.n_heads, final_len, model_config.head_dim,
                        device=device, dtype=io_dtype)
    cache.write(0, dummy, dummy)   # layer 0 is enough: allocates full reservation
    cache.advance(final_len)
    mem_mb = cache.memory_bytes() / (1024 * 1024)
    frag = cache.fragmentation_ratio()
    cache.free()
    return round(mem_mb, 2), round(frag, 4)


def run_one(model, model_config, cache, prompt_ids):
    controller = InferenceController(model, model_config)
    return controller.generate(prompt_ids, max_new_tokens=DECODE_LEN, cache=cache, use_cache=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global COMMIT
    COMMIT = git_commit()
    print(f"Running on: {device}  |  model: {N_LAYERS}L/{N_HEADS}H/{D_MODEL}d")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results = []

    for cache_type in ["contiguous", "paged"]:
        for seq_len in SEQ_LENS:
            for batch in BATCH_SIZES:
                max_seq_len = seq_len + DECODE_LEN
                final_len = seq_len + DECODE_LEN
                model_config = ModelConfig(max_seq_len=max_seq_len, n_layers=N_LAYERS,
                                           n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF)
                torch.manual_seed(SEED)   # identical weights + prompts per cell
                model = GPTModel(model_config).to(device).half()   # FP16 weights
                io_dtype = next(model.parameters()).dtype
                prompt_ids = torch.randint(0, model_config.vocab_size, (batch, seq_len), device=device)

                # memory + frag are deterministic and cheap (no forward) -> always reported
                base = {
                    "cache_type": cache_type, "seq_len": seq_len, "batch": batch,
                    "block_size": BLOCK_SIZE if cache_type == "paged" else "N/A",
                    # frag_ratio is waste/allocation on both sides, but the
                    # DENOMINATOR differs: contiguous divides by its up-front
                    # reservation, paged by the pages it actually holds. Label the
                    # basis so the column is never read as a like-for-like
                    # compare, and so the withdrawn comparison (see make_cache)
                    # can be reconstructed once reservation is a real variable.
                    "alloc_basis": "reservation" if cache_type == "contiguous" else "pages_held",
                    "dtype": str(io_dtype).replace("torch.", ""),
                    "attn_impl": "sdpa",
                    "commit": COMMIT,
                }
                blank_timing = {"ttft_ms": "", "tpot_ms": "", "throughput": "",
                                "throughput_cv": "", "peak_memory_mb": ""}
                try:
                    cache_mem_mb, frag = measure_cache_memory_frag(
                        cache_type, model_config, batch, max_seq_len, final_len, device, io_dtype)
                    base.update(cache_memory_mb=cache_mem_mb, frag_ratio=frag)

                    # one timed warmup: if the cell thrashes, bail before the 20-run loop
                    t0 = time.perf_counter()
                    run_one(model, model_config,
                            make_cache(cache_type, model_config, batch, max_seq_len, device, io_dtype),
                            prompt_ids)
                    if time.perf_counter() - t0 > BUDGET_S:
                        row = {**base, "status": "slow", **blank_timing}
                    else:
                        for _ in range(N_WARMUP - 1):
                            run_one(model, model_config,
                                    make_cache(cache_type, model_config, batch, max_seq_len, device, io_dtype),
                                    prompt_ids)
                        metrics_list = []
                        for _ in range(N_MEASURE):
                            cache = make_cache(cache_type, model_config, batch, max_seq_len, device, io_dtype)
                            metrics_list.append(run_one(model, model_config, cache, prompt_ids).metrics)

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
