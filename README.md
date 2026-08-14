# paged-kv-cache-analysis

I measured how the KV cache uses memory during transformer inference on one
consumer GPU. Two comparisons: a contiguous cache layout against a paged one, and
FP16 cache storage against INT8. Everything that survived checking is about memory,
which is where the interesting result turned out to be.

Decode latency is out of scope. At this model size the per-step cost is mostly
kernel launch overhead, not attention, so the timing columns describe my harness
rather than the cache. I report them but draw no conclusions from them.

The model is 8 layers, 8 heads, d_model 512, head_dim 64, FP16 weights, random
initialization. Runs use block size 16 and 60 decode steps, throw away 3 warmups,
and take the median of 20 measured runs at seed 0. Paged and contiguous produce
identical tokens for identical inputs. That gate is `tests/test_cache_equivalence.py`
and the suite passes 56 tests.

## The main finding

Peak memory depends on `batch × seq_len` and nothing else. Halve the batch and
double the context and you land in the same place:

| cells | dtype | peak (MB) | difference |
|---|---|---|---|
| 8192×32 vs 16384×16 | fp16 | 7086.5 vs 7090.6 | 0.058% |
| 8192×32 vs 16384×16 | int8 | 5246.6 vs 5255.5 | 0.170% |
| 8192×64 vs 16384×32 | int8 | 10328.9 vs 10332.8 | 0.038% |

FP16 runs out of memory at 8192×64 and 16384×32 under every budget I tried, so the
last pair exists only for INT8. Numbers from `experiments/results/oom_frontier.json`.

Fitting peak against `batch × seq_len` over the 20 cells of the precision sweep
gives two straight lines:

```
fp16:  peak_MB = 27.15 KB/token * (batch * seq_len) + 168.0 MB    R^2 = 0.99997
int8:  peak_MB = 19.93 KB/token * (batch * seq_len) + 164.0 MB    R^2 = 0.99999
```

Held out, those fits predict the five frontier cells to within +0.18% to +0.46%.
The frontier cells sit outside the fit grid and go to twice its longest context.
All the errors lean the same way, which makes sense: the frontier runs decode 8
tokens where the sweep decodes 60, so they hold slightly less cache than the fit
expects. Fitting the frontier cells alone gives 27.02 and 19.84 KB/token, so the
slope holds across both sets.

The slope splits in two. The cache part comes from the shape of the cache and
nothing else: `2 * n_layers * n_heads * head_dim * bytes_per_element` is 16 KB per
token for FP16 and 8.25 KB for INT8 with its per-token FP16 scale. The measured
cache column is 16.125 KB/token at seq 8192, rising to 18 at seq 512, because every
allocation also covers 60 decode tokens and rounds up to a whole block. What is left
of the peak slope is activations: 11.0 KB/token for FP16, 11.6 for INT8. Nearly the
same, as it should be when only the cache changes precision.

In capacity terms, under an 11.5 GB budget the FP16 fit runs out at about 440,000
cached tokens and the INT8 fit at about 599,000. The 36% gap is just the ratio of
the slopes. Both are ceilings, not promises: the budget caps reserved memory while
peak reports allocated memory, so the real limit sits a little lower.

<!-- TODO(suraj): why is a linear memory model the headline? Is it that capacity
     becomes predictable without running the workload, or that the activation term
     turns out not to care about cache dtype? Pick one, two sentences. -->

## Measurement pitfalls

Four harness bugs each gave me a confident wrong answer first.

| pitfall | what I concluded | corrected |
|---|---|---|
| The LM head ran over every prefill position, building a `[batch, seq_len, 50257]` logits tensor for positions never sampled from | The cache is a small share of memory, 10.5% of peak at 4096×16, so cache precision cannot matter | Sliced to the last position: peak at 4096×16 drops 9872.54 → 1905.43 MB, cache share 10.5% → 54.6%. At 2048×16, 3958.11 → 1038.9 MB and 13.3% → 50.8%. 8192×16 and 8192×32 go from OOM to ok |
| Attention looped over the batch in Python, so decode cost `B × n_layers` cache calls per token instead of `n_layers` | Paged decode is slow by nature, and block sizes can be ranked by throughput | Batched interface: paged TPOT at 2048×16 drops 22.374 → 11.024 ms, throughput 425.8 → 1206.2 tok/s. The same change swapped in SDPA and redefined fragmentation as waste over total allocation, which the old paged number overstated by n_pages: 12× at seq 128, 132× at seq 2048 |
| Contiguous reserved exactly `seq_len + decode_len`, the true final length | Contiguous fragments less than paged, 0.0 against 0.0019–0.0208 | Under an oracle reservation, 0.0 restates the setup instead of measuring. Comparison withdrawn, see below. The layouts also land within 1.88 MB on peak in all 20 cells, under 0.9% |
| The Windows driver spilled oversized allocations to host memory instead of failing | No OOM is reachable here: 14009 MB came back as a successful run on an 11.9 GB card | `set_per_process_memory_fraction` makes it fail in process. Under enforced budgets, six cells separate FP16 from INT8 |

The first three are in the CSVs before and after each fix, in git history. The
fourth is in `benchmarks/oom_frontier.py` and commit `e414dee`, not in a CSV.

<!-- TODO(suraj): argue why this table is a contribution and not an apology. What
     does a reader do differently after reading it? -->

## Results

INT8 moves the OOM frontier, which is what H3 predicted. Under enforced budgets,
FP16 fails where INT8 finishes:

| budget | cell | int8 peak (MB) |
|---|---|---|
| 9 GB | 8192×48 | 7790.6 |
| 10 GB | 8192×48 | 7790.6 |
| 11 GB, 11.5 GB | 8192×64 | 10328.9 |
| 11 GB, 11.5 GB | 16384×32 | 10332.8 |

At 8 GB nothing separates them: both finish at 8192×32 and both fail above it. So
the frontier is three configurations across four budgets.

How much of peak the cache accounts for, from `exp3_precision_sweep.csv`:

| seq × batch | cache/peak fp16 | cache/peak int8 | int8 peak vs fp16 |
|---|---|---|---|
| 512×1 | 5.2% | 2.8% | 2.7% less |
| 2048×16 | 50.8% | 33.9% | 22.8% less |
| 4096×32 | 57.0% | 39.5% | 25.6% less |
| 8192×32 | 58.0% | 40.5% | 26.0% less |

About 92.5% of the cache savings reach peak memory. Comparing the two dtypes at the
same cell, the drop in peak over the drop in cache bytes is 92.41%, 92.46% and
92.55% on the three frontier cells, and the median over the 20 sweep cells is
92.54%. Most of that follows from the fits: activations do not care about cache
dtype, so only the cache part of peak can move. The other 7.5% is a steady
0.58 KB/token that INT8 hands back. I have not worked out which allocation it is.
The FP32 and FP16 temporaries in `INT8KVCache._quantize` and `read` are the obvious
suspects, but that is a guess, not a measurement.

The INT8 cache is 0.515639 times the FP16 cache over the sweep and 0.515627 over
the frontier runs, against a predicted `(1 + 2/head_dim)/2 = 0.515625`. Four digits.
It is also a unit test, `test_int8_memory_ratio_is_0516`.

Quality is measured separately on real GPT-2 weights, over 284,614 tokens of
WikiText-2 test, because the study model is randomly initialized:

| model | Δppl | increase | top-1 agreement |
|---|---|---|---|
| gpt2 | +0.0007 | 0.002% | 0.9854 |
| gpt2-medium | +0.0016 | 0.006% | 0.9883 |

## What I withdrew

H1 said paged overtakes contiguous somewhere in the grid. It does not. Paged
throughput is 0.32× to 0.77× contiguous across all 20 cells of
`exp1_layout_comparison.csv`, with no trend toward 1.0 in any direction I swept, and
peak memory is equal within 0.9% everywhere. Two things make a crossover impossible
here: paged `read()` copies where contiguous returns a view, and every sequence in a
batch advances together, so paged never gets to allocate less than contiguous
reserved.

The fragmentation comparison between layouts goes with it. Contiguous reserves the
exact final length, so its fragmentation is 0.0 by construction. "Paged fragments
more" is a true number attached to a comparison that could not have come out any
other way. It needs a reservation policy that is a real variable, a fixed maximum or
a percentile of observed lengths, before it means anything. Both caches still write
the fragmentation columns with an `alloc_basis` label, so it can be rebuilt.

The block-size throughput ranking is withdrawn too. Two runs of the same script
disagreed: block 32 at 1100×16 gave 226.6 tok/s once and 495.3 the next time. No
block size wins on throughput at that noise level.

<!-- TODO(suraj): H1 is withdrawn, not answered. What would it take to test it
     honestly, the staggered harness or a fused kernel, and is that this paper or
     the next one? -->

## Limitations

One laptop GPU, an RTX 5070 Ti with 11.9 GB. One architecture, one head_dim, and
block size fixed at 16 outside the block-size sweep. The weights are random, which
is fine for memory and is why quality runs on GPT-2 instead.

Paged `read()` builds the whole history into a dense tensor. A real PagedAttention
kernel reads the block table inside attention and never builds it. So latency
measured against this implementation is the cost of paging without a fused kernel,
not a property of the layout.

Decode latency is not measurable here. Paged TPOT at batch 16 moves from 10.62 to
11.02 ms while context grows from 128 to 2048: sixteen times the context for 4% more
time per token. That is launch overhead, not attention. Throughput CV reaches 42.59%
in the layout sweep and 46.79% in the precision sweep, and the clocks are not locked.

Nothing is compared against a production serving system, so none of this says what a
deployed engine would do.

<!-- TODO(suraj): which of these does a reviewer reject the paper over, and what is
     your answer to it? Rank them, don't list them. -->

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest -q                 # equivalence gate, 56 tests
```

One file per command:

```bash
python -m benchmarks.run_cache_comparison    # experiments/results/exp1_layout_comparison.csv
python -m benchmarks.run_block_size_sweep    # experiments/results/exp2_block_size_sweep.csv
python -m benchmarks.run_precision_sweep     # experiments/results/exp3_precision_sweep.csv
python -m benchmarks.oom_frontier            # experiments/results/oom_frontier.json
python -m benchmarks.run_perplexity_eval --model gpt2 \
    --text experiments/data/wikitext2_test.txt   # experiments/results/exp3_perplexity.csv
```

`oom_frontier` needs CUDA and sets its own budgets, so its results do not depend on
the card's VRAM. `run_perplexity_eval` pulls GPT-2 from Hugging Face; the second row
of that table is the same command with `--model gpt2-medium`. `STATUS.md` says which
CSVs are current and which predate the harness fixes.

## Related work

vLLM (Kwon et al., SOSP 2023) introduced the paged KV cache and showed that paging
plus a scheduler plus continuous batching beats a contiguous serving system. It
fixed block size at 16 and measured the system end to end. This study is narrower:
one process, no scheduler, block size and cache precision as the variables, memory
as the outcome.

vAttention (Prabhu et al., ASPLOS 2025) argues that paging in user space costs
software overhead a fused attention kernel would avoid, and uses CUDA virtual memory
to get on-demand allocation without a block table. The caveat in
`src/kv_cache/paged.py` is the same point from the other side: my `read()` rebuilds
the history because there is no fused kernel here, so my latency numbers are an
instance of the overhead vAttention describes, not evidence about layouts.
