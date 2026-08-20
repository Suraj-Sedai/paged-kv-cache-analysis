# paged-kv-cache-analysis

I measured how the KV cache uses memory during transformer inference on one consumer
GPU. Two comparisons: a contiguous cache layout against a paged one, and FP16 cache
storage against INT8. What survived checking is a memory model that predicts peak
memory on an architecture it has never seen, to about 1%.

Latency is out of scope. At this model size the per-step cost is mostly kernel launch
overhead, not attention, so the timing columns describe my harness rather than the
cache. I report them and draw no conclusions from them.

The model is 8 layers, 8 heads, d_model 512, head_dim 64, FP16 weights, random
initialization. Runs use block size 16 and 60 decode steps, discard 3 warmups, and
take the median of 20 measured runs at seed 0. Paged and contiguous produce identical
tokens for identical inputs — that gate is `tests/test_cache_equivalence.py`, and the
suite passes 56 tests.

## The memory model

Peak memory depends on `batch × seq_len` and nothing else. Halve the batch and double
the context and you land in the same place, within 0.04% to 0.17%. Fitting peak
against `batch × seq_len` gives one straight line per dtype:

```
fp16:  peak_MB = 27.15 KB/token * (batch * seq_len) + 168.0 MB    R^2 = 0.99997
int8:  peak_MB = 19.93 KB/token * (batch * seq_len) + 164.0 MB    R^2 = 0.99999
```

Held out, those fits predict five frontier cells to within +0.18% to +0.46%, at twice
the longest context they were fitted on.

The slope splits in two. The cache part is derivable from the architecture before
running anything — `2 * n_layers * d_model * bytes`, which is 16 KB/token at FP16 and
8.25 at INT8 with its per-token scale. The rest is activations, about 11 KB/token,
and it does not care what dtype the cache is.

That split is what makes this a model instead of a curve fit, so I tested it at six
architectures, moving one axis at a time (`benchmarks/arch_sweep.py`). Activations
stay flat when depth changes — 3.6% across a 4× change in `n_layers`, while the cache
term moves 4× — and rise with width, +29% for double `d_model` and +71% for double
`d_ff`. Fitting the activation term on three shapes and holding out GPT-2 small
predicts its slope within 0.37% and its full peak memory within 1.1%, with no
parameter fitted on that shape.

In capacity terms, under an 11.5 GB budget the FP16 fit runs out at about 440,000
cached tokens and the INT8 fit at about 599,000. That 36% gap is just the ratio of
the slopes.

Numbers from `experiments/results/oom_frontier.json` and `arch_sweep.json`, worked
through in notebooks 04 and 05.

<!-- TODO(suraj): why is a linear memory model the headline? Is it that capacity
     becomes predictable without running the workload, or that the activation term
     turns out not to care about cache dtype? Pick one, two sentences. -->

## Precision

INT8 moves the OOM frontier, which is what H3 predicted. Under enforced budgets FP16
fails where INT8 finishes: at 8192×48 under 9 and 10 GB, and at 8192×64 and 16384×32
under 11 and 11.5 GB. At 8 GB nothing separates them. So the frontier is three
configurations across four budgets — real, and smaller than the halved cache suggests.

The reason it is smaller: activations do not change with cache dtype, so only the
cache part of peak can move. About 92.5% of the cache saving reaches peak. The
remaining 0.58 KB/token is steady across cells and I have not identified it; the FP32
and FP16 temporaries in `INT8KVCache._quantize` are the obvious suspects, but that is
a guess.

The INT8 cache measures 0.5156× the FP16 cache against a predicted
`(1 + 2/head_dim)/2 = 0.515625`. Quality is measured separately on real GPT-2 weights
over 284,614 tokens of WikiText-2, because the study model is randomly initialized:

| model | Δppl | increase | top-1 agreement |
|---|---|---|---|
| gpt2 | +0.0007 | 0.002% | 0.9854 |
| gpt2-medium | +0.0016 | 0.006% | 0.9883 |

## What I withdrew

H1 said paged overtakes contiguous somewhere in the grid. It does not. Paged
throughput is 0.32× to 0.77× contiguous across all 20 cells, with no trend toward
1.0, and peak memory is equal within 0.9% everywhere. Two things make a crossover
impossible here: paged `read()` copies where contiguous returns a view, and the pool
is allocated at full final size up front, so paged never gets to hold less than
contiguous reserved.

The fragmentation comparison goes with it. Contiguous reserves the exact final
length, so its fragmentation is 0.0 by construction — a true number attached to a
comparison that could not have come out any other way. The block-size sweep has the
same problem one level down: with uniform sequence lengths, fragmentation is fixed by
`(seq_len + 60) mod block_size` and reproduces exactly from arithmetic. Both need
variable-length sequences and a real reservation policy before they mean anything.

The block-size throughput ranking is withdrawn too. Two runs of the same script gave
226.6 and 495.3 tok/s at the same cell.

<!-- TODO(suraj): H1 is withdrawn, not answered. What would it take to test it
     honestly, the staggered harness or a fused kernel, and is that this paper or
     the next one? -->

## Measurement pitfalls

Four harness bugs each gave me a confident wrong answer first. They are in the CSVs
before and after each fix, in git history.

| bug | what I wrongly concluded |
|---|---|
| LM head ran over every prefill position, building a `[batch, seq_len, 50257]` logits tensor | the cache is only 10.5% of peak, so cache precision cannot matter (it is 50–58%) |
| attention looped over the batch in Python, costing `B × n_layers` cache calls per token | paged decode is slow by nature, and block sizes can be ranked by throughput |
| contiguous reserved exactly the true final length | contiguous fragments less than paged |
| the Windows driver spilled oversized allocations to host memory instead of failing | no OOM is reachable on this machine |

<!-- TODO(suraj): argue why this table is a contribution and not an apology. What
     does a reader do differently after reading it? -->

## Limitations

One laptop GPU, an RTX 5070 Ti with 11.9 GB. Block size is fixed at 16 outside the
block-size sweep, and the weights are random, which is fine for memory and is why
quality runs on GPT-2 instead.

The architecture sweep moves `n_layers`, `d_model` and `d_ff`, but `head_dim` is
pinned at 64 and `n_heads` moves with `d_model`. So the cache term is confirmed only
as a product, and the model as written would not transfer to GQA or MQA, where the KV
head count is decoupled from width. Each width axis moves by a single doubling, and
four points per shape rule out curvature that would matter without giving a
confidence interval on a slope.

Paged `read()` builds the whole history into a dense tensor. A real PagedAttention
kernel reads the block table inside attention and never builds it, so latency measured
here is the cost of paging without a fused kernel, not a property of the layout.

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
python -m benchmarks.run_cache_comparison    # exp1_layout_comparison.csv
python -m benchmarks.run_block_size_sweep    # exp2_block_size_sweep.csv
python -m benchmarks.run_precision_sweep     # exp3_precision_sweep.csv
python -m benchmarks.oom_frontier            # oom_frontier.json
python -m benchmarks.arch_sweep              # arch_sweep.json
python -m benchmarks.run_perplexity_eval --model gpt2 \
    --text experiments/data/wikitext2_test.txt   # exp3_perplexity.csv
```

`oom_frontier` and `arch_sweep` need CUDA and set their own budgets, so their results
do not depend on the card's VRAM. `run_perplexity_eval` pulls GPT-2 from Hugging Face.
`STATUS.md` says which outputs are current and which predate the harness fixes.

## Related work

vLLM (Kwon et al., SOSP 2023) introduced the paged KV cache and measured a full
serving system end to end, with block size fixed at 16. This study is narrower: one
process, no scheduler, block size and cache precision as the variables, memory as the
outcome.

vAttention (Prabhu et al., ASPLOS 2025) argues that paging in user space costs
software overhead a fused kernel would avoid. My `read()` rebuilds the history because
there is no fused kernel here, so my latency numbers are an instance of that overhead,
not evidence about layouts.
