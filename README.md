# paged-kv-cache-analysis

A small, reproducible benchmarking study of **paged vs. contiguous KV cache** for LLM inference.
Not a new algorithm — a measurement paper on *when and why* each cache design wins.

> Suraj Sedai 

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
