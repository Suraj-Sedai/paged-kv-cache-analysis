"""Regenerate exp1/exp2/exp3-precision CSVs as *_v2 with the metrics fix.

Writes alongside the originals (never overwrites). Each row carries BOTH:
  tpot_ms            -> FIXED metric (decode_step_times = forward steps only)
  tpot_ms_old_buggy  -> OLD metric  (first_sample injected as element 0)
computed from the SAME runs, so before/after is same-hardware and isolates the
fix from run-to-run noise and from the (slower) CPU this is regenerated on.

Reduced for CPU feasibility (documented in the report):
  N_WARMUP=1, N_MEASURE=3, and cells with seq*batch > PROD_CAP are marked 'slow'
  WITHOUT running (a single fp16 forward on the largest original cells takes
  minutes on this box). The original scripts also mark oversized cells 'slow'
  via BUDGET_S; this just pre-empts the expensive probe.
"""
import csv, os, statistics, time
import torch

from src.inference.controller import InferenceController
from src.kv_cache.contiguous_cache import ContiguousKVCache
from src.kv_cache.paged import PagedKVCache
from src.kv_cache.int8_paged import INT8KVCache
from src.model_core.config import ModelConfig
from src.model_core.model import GPTModel
from src.profiling.metrics import percentile
import src.inference.controller as C

DECODE_LEN = 60
N_WARMUP   = 1
N_MEASURE  = 3
PROD_CAP   = 9600          # seq*batch above this -> 'slow' (not run) on this CPU
RESULTS    = os.path.join("experiments", "results")
FIELDNAMES = ["cache_type", "seq_len", "batch", "block_size", "status",
              "ttft_ms", "tpot_ms", "tpot_ms_old_buggy", "throughput",
              "throughput_cv", "peak_memory_mb", "cache_memory_mb", "frag_ratio"]

# capture raw decode list + first_sample from each generate() without touching metrics.py
_CAP = {}
_orig = C.build_generation_metrics
def _spy(**k):
    _CAP["d"] = list(k["decode_step_times_ms"])
    _CAP["fs"] = k["first_sampling_latency_ms"]
    return _orig(**k)
C.build_generation_metrics = _spy


def make_cache(kind, mc, batch, max_seq_len, block_size, device, dtype):
    if kind == "contiguous":
        return ContiguousKVCache(mc, max_seq_len, device, dtype=dtype)
    pb = (max_seq_len + block_size - 1) // block_size
    common = dict(n_layers=mc.n_layers, n_heads=mc.n_heads, dim_head=mc.head_dim,
                  block_size=block_size, num_blocks=pb * batch, device=device)
    if kind in ("paged", "paged_fp16"):
        return PagedKVCache(**common, dtype=dtype)
    if kind == "paged_int8":
        return INT8KVCache(**common, compute_dtype=dtype)
    raise ValueError(kind)


def cache_mem_frag(kind, mc, batch, max_seq_len, block_size, final_len, device, dtype):
    cache = make_cache(kind, mc, batch, max_seq_len, block_size, device, dtype)
    dummy = torch.zeros(mc.n_heads, final_len, mc.head_dim, device=device, dtype=dtype)
    for b in range(batch):
        cache.write(0, b, dummy, dummy); cache.advance(b, final_len)
    mem = cache.memory_bytes() / (1024 * 1024)
    frag = cache.fragmentation_ratio(0)
    for b in range(batch):
        cache.free(b)
    return round(mem, 2), round(frag, 4)


def run_cell(kind, seq_len, batch, block_size, half):
    max_seq_len = seq_len + DECODE_LEN
    mc = ModelConfig(max_seq_len=max_seq_len, n_layers=8, n_heads=8, d_model=512, d_ff=2048)
    model = GPTModel(mc).eval()
    if half:
        model = model.half()
    dtype = next(model.parameters()).dtype
    prompt = torch.randint(0, mc.vocab_size, (batch, seq_len))
    bs = block_size if block_size != "N/A" else 16
    mem, frag = cache_mem_frag(kind, mc, batch, max_seq_len, bs, max_seq_len, "cpu", dtype)
    base = {"cache_type": kind, "seq_len": seq_len, "batch": batch,
            "block_size": block_size, "cache_memory_mb": mem, "frag_ratio": frag}

    if seq_len * batch > PROD_CAP:
        return {**base, "status": "slow", "ttft_ms": "", "tpot_ms": "",
                "tpot_ms_old_buggy": "", "throughput": "", "throughput_cv": "",
                "peak_memory_mb": ""}

    ctrl = InferenceController(model, mc)
    for _ in range(N_WARMUP):
        ctrl.generate(prompt, DECODE_LEN, cache=make_cache(kind, mc, batch, max_seq_len, bs, "cpu", dtype), use_cache=True)
    ttfts, new_p50s, old_p50s, thrs, peaks = [], [], [], [], []
    for _ in range(N_MEASURE):
        r = ctrl.generate(prompt, DECODE_LEN, cache=make_cache(kind, mc, batch, max_seq_len, bs, "cpu", dtype), use_cache=True)
        d, fs = _CAP["d"], _CAP["fs"]
        ttfts.append(r.metrics.ttft_ms)
        new_p50s.append(percentile(d, 50))
        old_p50s.append(percentile([fs] + d, 50))
        thrs.append(r.metrics.throughput_tokens_per_sec)
        peaks.append(r.metrics.peak_memory_mb)
    thr_med = statistics.median(thrs)
    cv = (statistics.pstdev(thrs) / thr_med * 100) if thr_med else 0.0
    return {**base, "status": "ok",
            "ttft_ms": round(statistics.median(ttfts), 3),
            "tpot_ms": round(statistics.median(new_p50s), 3),
            "tpot_ms_old_buggy": round(statistics.median(old_p50s), 3),
            "throughput": round(thr_med, 1),
            "throughput_cv": round(cv, 2),
            "peak_memory_mb": round(statistics.median(peaks), 2)}


def write_csv(name, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES); w.writeheader(); w.writerows(rows)
    print(f"  saved {len(rows)} rows -> {path}", flush=True)


def main():
    t0 = time.perf_counter()
    # ---- exp1: layout comparison (fp32, contiguous vs paged) ----
    print("[exp1] start", flush=True)
    rows = []
    for kind in ["contiguous", "paged"]:
        for seq in [128, 256, 512, 1024, 2048]:
            for batch in [1, 4, 8, 16]:
                bs = 16 if kind == "paged" else "N/A"
                rows.append(run_cell(kind, seq, batch, bs, half=False))
                print("   ", rows[-1], flush=True)
    write_csv("exp1_layout_comparison_v2.csv", rows)

    # ---- exp2: block-size sweep (fp16, paged) ----
    print("[exp2] start", flush=True)
    rows = []
    for bs in [8, 16, 32, 64]:
        for seq in [100, 300, 600, 1100, 2000]:
            for batch in [1, 8, 16]:
                rows.append(run_cell("paged", seq, batch, bs, half=True))
                print("   ", rows[-1], flush=True)
    write_csv("exp2_block_size_sweep_v2.csv", rows)

    # ---- exp3: precision sweep (fp16 weights, paged fp16 vs int8) ----
    print("[exp3] start", flush=True)
    rows = []
    for kind in ["paged_fp16", "paged_int8"]:
        for seq in [512, 1024, 2048, 4096, 8192]:
            for batch in [1, 8, 16, 32]:
                rows.append(run_cell(kind, seq, batch, 16, half=True))
                print("   ", rows[-1], flush=True)
    write_csv("exp3_precision_sweep_v2.csv", rows)
    print(f"DONE in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
