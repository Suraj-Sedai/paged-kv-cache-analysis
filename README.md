# paged-kv-cache-analysis

A reproducible measurement of **paged vs. contiguous KV cache** for transformer
inference — when each layout wins, by how much, and how block size and cache
precision move the line. This is a characterization study, not a new algorithm.

*Suraj Sedai*

## Motivation

I kept reading that the paged KV cache "just wins," and that never sat right with me.
vLLM made paging the default for serving, but the original paper measured it as a
finished system — paging plus a scheduler plus continuous batching, all at once. What
I wanted to know is narrower and, I think, more useful: strip away the scheduler, hold
the model and the harness fixed, and ask what the *layout itself* costs and buys. A
block table is not free. Every token read now goes through an indirection, and at some
point that overhead has to outweigh the memory it saves — or not. Nobody had drawn that
line.

The lever I care most about is `block_size`. It's a single knob that trades two things
against each other: small blocks waste less memory in the last partial block but add
more per-token lookups; large blocks do the reverse. vLLM fixed it at 16 and moved on.
Treating it as the independent variable — sweeping it while everything else stays still
— is the part of this work I haven't seen done cleanly, and it's where the result is.

I also wanted the discipline of a measurement paper: a result that survives a re-run,
states its baseline, and reports the cases where my own hypothesis was wrong. A clean
negative is worth more than a massaged win.

## Background

During autoregressive decoding, attention needs the keys and values of every prior
token. Recomputing them each step is quadratic, so they're cached — the **KV cache**.
It grows linearly with sequence length and batch, and for long contexts it, not the
weights, is what fills the GPU.

How that cache is laid out in memory is the question here:

- **Contiguous.** One tensor per request, sized for the maximum sequence length up
  front: `[layers, heads, max_seq_len, head_dim]`. Reads are a slice — no indirection,
  fast. The cost is that you reserve the worst case even when the actual sequence is
  short, and you can't grow past it.
- **Paged.** Memory is carved into fixed-size blocks and handed out on demand through a
  per-sequence block table, the way an OS pages virtual memory. You only allocate what
  a sequence actually uses, and different sequences share one pool. The cost is the
  indirection on every access, plus whatever is wasted in each sequence's last partial
  block (internal fragmentation).

Paging trades raw access speed for memory flexibility. The interesting question is the
exchange rate, and how `block_size` sets it.

## Questions and hypotheses

1. **Crossover.** At what `seq_len` × `batch` does paged overtake contiguous on
   throughput and memory?
   *H1 — paged wins at long sequences / high batch where its memory flexibility pays
   off, and loses at short sequences where indirection dominates.*
2. **Block size.** How does `block_size` ∈ {8, 16, 32, 64} trade fragmentation against
   lookup cost?
   *H2 — there's a workload-dependent optimum; small blocks cut fragmentation but raise
   per-token cost.*
3. **Precision.** Does INT8 KV quantization shift the crossover?
   *H3 — INT8 relieves memory pressure and pushes the crossover outward.*

Hypotheses are falsifiable, and at least one of them broke (see below). When the data
disagreed, the disagreement is the finding.

## Method

Everything runs on one GPU (RTX 5070 Ti Laptop), FP16 weights, an 8-layer / 8-head /
512-d model — small enough to sweep exhaustively, large enough that the cache behaves
realistically. Each cell is seeded, warmed up with discarded runs, and reported as the
median over 20 measurements with its coefficient of variation. Config and commit are
logged next to every CSV.

A few rules this codebase enforces, because earlier versions violated them and produced
results that looked great and meant nothing:

- **Equivalence is the gate.** Contiguous and paged must emit identical tokens for
  identical inputs (`pytest tests/test_cache_equivalence.py`) before any benchmark is
  allowed to count.
- **Memory is measured live**, not as pre-allocated pool capacity — otherwise paging's
  whole point is erased on paper.
- **Fragmentation is computed from the actual last-block waste**, never assumed, so it
  moves with `block_size` and sequence length as it should.
- **The read-path asymmetry is labeled, not hidden.** Contiguous returns a view; paged
  reassembles its blocks into a contiguous tensor on read. That copy is real and it
  taxes paged's decode latency, so timing is read against it rather than as if the paths
  were equal.

On this GPU decode is launch-bound, which I verified with a standalone gate
(`benchmarks/validity_gate.py`). So memory and quality are the trustworthy axes;
latency is reported honestly but with that caveat attached.

## What I found

- **Block size has an optimum at 8–16, and it's the headline.** Small blocks cut
  fragmentation and, on this workload, also win throughput; 32 and 64 raise TPOT and
  waste memory with nothing to show for it. At seq 1100 / batch 16, throughput falls
  from ~400 tok/s at block 8 to ~240 at block 64, while last-block waste climbs toward
  0.8 at the largest blocks. H2 holds, and the useful range sits at or below vLLM's
  default of 16.

- **INT8 KV is almost free on quality and exactly 0.516× on memory.** The cache shrinks
  to `(1 + 2/head_dim)/2` of FP16 — int8 data plus a per-token fp16 scale, dead
  consistent across every cell. Perplexity on WikiText-2 moves by under 0.01%
  (+0.002% on GPT-2, +0.006% on GPT-2-medium), with ~98.5% top-1 agreement. The
  per-token symmetric scheme holds up at least to GPT-2-medium.

- **But INT8 does not move the OOM frontier here — H3 is false on this hardware.** It
  fails at exactly the same `seq_len` × `batch` cells as FP16. The reason is simple once
  you look: the cache is only ~10% of peak memory (≈1 GB of cache against ≈10 GB peak at
  seq 4096 / batch 16), and the rest is prefill activations, which quantizing the cache
  does nothing for. Precision helps when the cache is the bottleneck. On this setup it
  isn't, and the honest result is to say so.

Full numbers, per-cell, with provenance, are in `STATUS.md` and `experiments/results/`.

## Prior work

- **Kwon et al., 2023 (vLLM / PagedAttention)** introduced the paged KV cache and showed
  it wins as a system, but did not sweep `block_size` or map the regime-dependent
  tradeoff. That's the gap this fills.
- **Dao et al. (FlashAttention)** for the IO-aware, memory-bound framing of attention.
- **KIVI / SmoothQuant** for KV-cache and activation quantization. My INT8 is
  per-token-scale, KV-cache-only; weights stay FP16.

## Limitations

These are real and stated up front. The model is randomly initialized — fine for
memory and latency, which is why quality (perplexity) is measured separately on real
GPT-2 weights rather than this model. Sequences within a batched forward advance in
lockstep, so per-sequence memory wins can't appear inside one batch and would need a
staggered harness to surface. The paged read path copies, which taxes its latency. And
everything is one consumer laptop GPU; the numbers are a characterization on this
hardware, not a universal claim.

## Layout

```
src/kv_cache/        base, contiguous, paged, int8_paged
src/{model_core,inference,profiling}/   ported model + harness (frozen)
benchmarks/          sweep scripts + validity gate
experiments/         results/ (CSVs), data/ (eval corpus)
tests/               equivalence + per-cache unit tests
```

`InferenceController` is cache-agnostic — swapping contiguous ↔ paged ↔ INT8 is a
one-line change, which is what keeps the comparison fair.

## Run

```bash
pip install -r requirements.txt
pytest                                          # equivalence is the gate

python benchmarks/run_cache_comparison.py       # Exp 1 — layout crossover
python benchmarks/run_block_size_sweep.py       # Exp 2 — block size
python benchmarks/run_precision_sweep.py        # Exp 3a — INT8 memory
python benchmarks/run_perplexity_eval.py --text experiments/data/wikitext2_test.txt
                                                # Exp 3b — INT8 quality
```
