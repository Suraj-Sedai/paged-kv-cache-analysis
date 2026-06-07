# paged-kv-cache-analysis

A small, reproducible benchmarking study of **paged vs. contiguous KV cache** for LLM inference.
Not a new algorithm — a measurement paper on *when and why* each cache design wins.

> Suraj Sedai 

## Motivation

Modern LLM inference is bottlenecked by **memory, not compute**. The key-value cache —
which stores intermediate attention states to avoid redundant recomputation — grows with
sequence length and batch size, quickly exhausting GPU memory on any realistic workload.

Two cache designs compete to manage that memory:

- **Contiguous allocation** — one pre-allocated tensor `[layers, batch, heads, max_seq_len, head_dim]`.
- **Paged allocation** — fixed-size blocks with a lookup table, allocating memory on demand.

The vLLM paper (Kwon et al., 2023) introduced the paged KV cache and showed throughput
gains, but it didn't analyze the *regime-dependent* tradeoffs or the sensitivity to
**block size**. I started this project to fill that gap: not a new algorithm, but the first
clean, reproducible characterization of **when and why** each design wins, with `block_size`
as an explicit independent variable. The aim is a narrow, falsifiable, and useful result —
a measurement paper that tells practitioners which cache to reach for, and when.


## What I'm looking for

1. **The crossover point** — at what `seq_len` × `batch_size` does paged KV cache start beating
   contiguous on throughput and memory?
2. **Block size sensitivity** — how does `block_size` (8/16/32/64) trade fragmentation against
   per-token lookup overhead, and is there an optimum? *(the novel part)*
3. **Precision shift** — does INT8 KV quantization move the crossover, and at what cost to quality
   (perplexity)?

## Status

- ✅ Contiguous + paged caches, shared `BaseKVCache` API, equivalence tests
- ✅ Experiment 1 (layout comparison) — data + analysis notebook
- ⬜ Experiment 2 (block size sweep)
- ⬜ Experiment 3 (INT8 precision + perplexity)


## Layout

```
src/kv_cache/      cache implementations (base, contiguous, paged)
src/{model_core,inference,profiling}/   ported model + harness
benchmarks/        sweep scripts
experiments/results/   CSV outputs
analysis/notebooks/    analysis + figures
tests/             equivalence + unit tests
```

## Run

```bash
pip install -r requirements.txt
python -m benchmarks.run_cache_comparison    # Experiment 1
pytest                                       # tests
```
