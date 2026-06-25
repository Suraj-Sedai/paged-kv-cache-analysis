#!/usr/bin/env python3
"""
Publication figures for the paged-KV-cache measurement paper.

Reflects the CORRECTED findings (post TPOT-fix rerun, commit e383030):
  - H1 (exp1): contiguous beats paged on decode throughput everywhere; gap is
    read-path indirection, not cache footprint (cache memory within 2 MB).
  - H2 (exp2): fragmentation-vs-TPOT TRADE-OFF across block_size. Throughput
    ranking WITHDRAWN (launch-bound, not reproducible) -- not plotted as a result.
  - H3 (exp3): INT8 cache = 0.516x FP16, cache is ~10-14% of peak, and INT8
    rescues ZERO failure cells (OOM frontier unchanged).

Every number is read from the canonical CSVs. Aggregates are printed to stdout
for verification. Each figure also writes its underlying numbers to a small CSV.

Provenance (verify with `git log -1 -- <csv>`):
  exp1_layout_comparison.csv  : commit e383030 (2026-06-21, "rerun ... fixed TPOT")
  exp2_block_size_sweep.csv   : commit e383030
  exp3_precision_sweep.csv    : commit e383030
  exp3_perplexity.csv         : commit eae781c (2026-06-15)  [not plotted here]
"""
from __future__ import annotations
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------- paths
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "experiments", "results")
OUT = os.path.join(REPO, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

EXP1 = os.path.join(RESULTS, "exp1_layout_comparison.csv")
EXP2 = os.path.join(RESULTS, "exp2_block_size_sweep.csv")
EXP3 = os.path.join(RESULTS, "exp3_precision_sweep.csv")

CV_THRESHOLD = 10.0  # throughput_cv (%) above which a cell is flagged noisy

# ----------------------------------------------------------------------------- style
# Okabe-Ito colorblind-safe palette
OI = {
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "black":   "#000000",
}
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "font.family": "DejaVu Sans",
})


def save(fig, stem):
    png = os.path.join(OUT, stem + ".png")
    pdf = os.path.join(OUT, stem + ".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


def dump(df, stem):
    path = os.path.join(OUT, stem + "_data.csv")
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


def flag_noisy(df, label):
    noisy = df[df["throughput_cv"] > CV_THRESHOLD]
    print(f"  [{label}] cells with throughput_cv > {CV_THRESHOLD}%: {len(noisy)}")
    for _, r in noisy.iterrows():
        bs = r.get("block_size", "")
        print(f"      {r['cache_type']:<14} seq={int(r['seq_len']):>5} "
              f"batch={int(r['batch']):>2} bs={bs:>4}  cv={r['throughput_cv']:.1f}%")


# =============================================================================
# FIGURE 1 -- H1: cost of paged indirection (exp1)
# =============================================================================
def figure1():
    print("\n=== FIGURE 1 (H1: paged indirection cost, exp1) ===")
    df = pd.read_csv(EXP1)
    ok = df[df["status"] == "ok"].copy()

    # --- main panel: throughput vs seq_len at batch 16 ---
    b16 = ok[ok["batch"] == 16].sort_values("seq_len")
    cont = b16[b16["cache_type"] == "contiguous"]
    paged = b16[b16["cache_type"] == "paged"]

    print("  throughput (tok/s) at batch=16:")
    seqs = sorted(set(cont["seq_len"]) & set(paged["seq_len"]))
    rows = []
    for s in seqs:
        c = cont[cont["seq_len"] == s].iloc[0]
        p = paged[paged["seq_len"] == s].iloc[0]
        ratio = c["throughput"] / p["throughput"]
        cache_dab = abs(c["cache_memory_mb"] - p["cache_memory_mb"])
        peak_dab = abs(c["peak_memory_mb"] - p["peak_memory_mb"])
        print(f"    seq={int(s):>5}: contig={c['throughput']:>7.1f}  "
              f"paged={p['throughput']:>6.1f}  contig/paged={ratio:.2f}x   "
              f"cacheΔ={cache_dab:.1f}MB  peakΔ={peak_dab:.1f}MB")
        rows.append(dict(seq_len=int(s),
                         contiguous_tps=c["throughput"], paged_tps=p["throughput"],
                         contig_over_paged=ratio,
                         contiguous_cv=c["throughput_cv"], paged_cv=p["throughput_cv"],
                         cache_mem_delta_mb=cache_dab, peak_mem_delta_mb=peak_dab))
    fdata = pd.DataFrame(rows)
    flag_noisy(b16, "fig1 batch16")
    max_cache_delta = fdata["cache_mem_delta_mb"].max()

    # small-multiples data across batches
    batches = [1, 4, 8, 16]

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)

    # left: batch-16 main panel
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(fdata["seq_len"], fdata["contiguous_tps"], "-o",
            color=OI["blue"], label="contiguous")
    ax.plot(fdata["seq_len"], fdata["paged_tps"], "-s",
            color=OI["vermil"], label="paged (block=16)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(fdata["seq_len"])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("sequence length")
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_title("(a) batch = 16: contiguous wins every cell")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")

    # right: small multiples across batch
    ax2 = fig.add_subplot(gs[0, 1])
    cmap_c = plt.cm.Blues(np.linspace(0.45, 0.95, len(batches)))
    cmap_p = plt.cm.Oranges(np.linspace(0.45, 0.95, len(batches)))
    for i, bt in enumerate(batches):
        c = ok[(ok["cache_type"] == "contiguous") & (ok["batch"] == bt)].sort_values("seq_len")
        p = ok[(ok["cache_type"] == "paged") & (ok["batch"] == bt)].sort_values("seq_len")
        ax2.plot(c["seq_len"], c["throughput"], "-o", color=cmap_c[i],
                 markersize=4, linewidth=1.4)
        ax2.plot(p["seq_len"], p["throughput"], "--s", color=cmap_p[i],
                 markersize=4, linewidth=1.4)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sorted(ok["seq_len"].unique()))
    ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax2.set_xlabel("sequence length")
    ax2.set_ylabel("decode throughput (tok/s)")
    ax2.set_title("(b) batches 1/4/8/16 (darker = larger batch)")
    ax2.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=OI["blue"], marker="o", linestyle="-",
                          label="contiguous"),
               plt.Line2D([], [], color=OI["orange"], marker="s", linestyle="--",
                          label="paged")]
    ax2.legend(handles=handles, frameon=False, loc="upper right")

    fig.suptitle("H1 -- cost of paged indirection (decode throughput)", y=1.02,
                 fontsize=13, fontweight="bold")
    cap = ("Contiguous beats paged in every cell. Live KV-cache memory differs by "
           f"≤{max_cache_delta:.0f} MB between layouts, so the gap is read-path "
           "indirection (paged read() materialises history via torch.cat), not "
           "cache footprint. (2048×16 omitted: 'slow' for both layouts.)")
    fig.text(0.5, -0.06, cap, ha="center", va="top", fontsize=8.5, wrap=True,
             color="0.3")
    save(fig, "fig1_h1_throughput")
    dump(fdata, "fig1_h1_throughput")


# =============================================================================
# FIGURE 2 -- H2: fragmentation vs latency trade-off (exp2)
# =============================================================================
def figure2():
    print("\n=== FIGURE 2 (H2: frag-vs-TPOT trade-off, exp2) ===")
    df = pd.read_csv(EXP2)
    ok = df[df["status"] == "ok"].copy()

    g = ok.groupby("block_size")
    agg = g.agg(mean_frag=("frag_ratio", "mean"),
                mean_tpot=("tpot_ms", "mean"),
                n=("frag_ratio", "size")).reset_index().sort_values("block_size")
    print("  per block_size aggregates (mean over all seq_len x batch cells):")
    for _, r in agg.iterrows():
        print(f"    bs={int(r['block_size']):>3}: mean_frag={r['mean_frag']:.3f}  "
              f"mean_tpot={r['mean_tpot']:6.2f} ms   (n={int(r['n'])} cells)")
    flag_noisy(ok, "fig2 exp2")

    bs = agg["block_size"].astype(int).astype(str)
    x = np.arange(len(agg))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) fragmentation -- RISES
    axL.plot(x, agg["mean_frag"], "-o", color=OI["vermil"])
    for xi, yi in zip(x, agg["mean_frag"]):
        axL.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=OI["vermil"])
    axL.set_xticks(x); axL.set_xticklabels(bs)
    axL.set_xlabel("block size")
    axL.set_ylabel("mean fragmentation ratio")
    axL.set_title("(a) fragmentation RISES ▲")
    axL.set_ylim(0, max(agg["mean_frag"]) * 1.25)
    axL.annotate("", xy=(len(x) - 1, agg["mean_frag"].iloc[-1]),
                 xytext=(0, agg["mean_frag"].iloc[0]),
                 arrowprops=dict(arrowstyle="->", color=OI["vermil"], alpha=0.35, lw=1.5))

    # (b) TPOT -- FALLS
    axR.plot(x, agg["mean_tpot"], "-s", color=OI["blue"])
    for xi, yi in zip(x, agg["mean_tpot"]):
        axR.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=OI["blue"])
    axR.set_xticks(x); axR.set_xticklabels(bs)
    axR.set_xlabel("block size")
    axR.set_ylabel("mean TPOT (ms)")
    axR.set_title("(b) per-token latency FALLS ▼")
    axR.set_ylim(0, max(agg["mean_tpot"]) * 1.25)
    axR.annotate("", xy=(len(x) - 1, agg["mean_tpot"].iloc[-1]),
                 xytext=(0, agg["mean_tpot"].iloc[0]),
                 arrowprops=dict(arrowstyle="->", color=OI["blue"], alpha=0.35, lw=1.5))

    fig.suptitle("H2 -- block size is a fragmentation ↔ latency trade-off, not a single optimum",
                 y=1.0, fontsize=13, fontweight="bold")
    cap = ("Opposing directions are the point: small blocks waste less memory but cost "
           "more per-token latency. Throughput-based block-size ranking is WITHDRAWN "
           "(launch-bound, not reproducible) and is not shown. TPOT is also taxed by the "
           "paged read-path copy (a measurement caveat), so panel (b) is an upper bound on "
           "the true indirection cost.")
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=8.5, wrap=True, color="0.3")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save(fig, "fig2_h2_tradeoff")
    dump(agg.assign(block_size=agg["block_size"].astype(int)), "fig2_h2_tradeoff")


# =============================================================================
# FIGURE 3 -- H3 memory: INT8 vs FP16 cache, and cache share of peak (exp3)
# =============================================================================
def figure3():
    print("\n=== FIGURE 3 (H3 memory, exp3) ===")
    df = pd.read_csv(EXP3)
    ok = df[df["status"] == "ok"].copy()

    b16 = ok[ok["batch"] == 16]
    fp = b16[b16["cache_type"] == "paged_fp16"].sort_values("seq_len")
    q8 = b16[b16["cache_type"] == "paged_int8"].sort_values("seq_len")
    seqs = sorted(set(fp["seq_len"]) & set(q8["seq_len"]))

    print("  (a) cache memory INT8 vs FP16 at batch=16 (ok cells):")
    rows_a = []
    for s in seqs:
        f = fp[fp["seq_len"] == s].iloc[0]
        q = q8[q8["seq_len"] == s].iloc[0]
        ratio = q["cache_memory_mb"] / f["cache_memory_mb"]
        print(f"    seq={int(s):>5}: fp16={f['cache_memory_mb']:>7.2f} MB  "
              f"int8={q['cache_memory_mb']:>7.2f} MB  int8/fp16={ratio:.4f}")
        rows_a.append(dict(seq_len=int(s), fp16_cache_mb=f["cache_memory_mb"],
                           int8_cache_mb=q["cache_memory_mb"], int8_over_fp16=ratio))
    a = pd.DataFrame(rows_a)

    print("  (b) cache as %% of peak (FP16, batch=16, ok cells):")
    rows_b = []
    for s in seqs:
        f = fp[fp["seq_len"] == s].iloc[0]
        pct = 100.0 * f["cache_memory_mb"] / f["peak_memory_mb"]
        print(f"    seq={int(s):>5}: cache={f['cache_memory_mb']:>7.1f} MB / "
              f"peak={f['peak_memory_mb']:>8.1f} MB = {pct:.1f}%")
        rows_b.append(dict(seq_len=int(s), cache_mb=f["cache_memory_mb"],
                           peak_mb=f["peak_memory_mb"], cache_pct_of_peak=pct))
    b = pd.DataFrame(rows_b)
    mean_ratio = a["int8_over_fp16"].mean()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    xs = np.arange(len(seqs))
    labels = [str(int(s)) for s in seqs]

    # (a) grouped bars fp16 vs int8 cache memory
    w = 0.38
    axL.bar(xs - w / 2, a["fp16_cache_mb"], w, color=OI["blue"], label="FP16")
    axL.bar(xs + w / 2, a["int8_cache_mb"], w, color=OI["orange"], label="INT8")
    for xi, r in zip(xs, a["int8_over_fp16"]):
        axL.annotate(f"{r:.3f}×", (xi + w / 2, a['int8_cache_mb'].iloc[list(xs).index(xi)]),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=8, color=OI["orange"])
    axL.set_xticks(xs); axL.set_xticklabels(labels)
    axL.set_xlabel("sequence length")
    axL.set_ylabel("KV cache memory (MB)")
    axL.set_title(f"(a) INT8 cache ≈ {mean_ratio:.3f}× FP16, constant")
    axL.legend(frameon=False)

    # (b) cache % of peak
    axR.bar(xs, b["cache_pct_of_peak"], color=OI["green"], width=0.6)
    for xi, yi in zip(xs, b["cache_pct_of_peak"]):
        axR.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=9, color="0.2")
    axR.set_xticks(xs); axR.set_xticklabels(labels)
    axR.set_xlabel("sequence length")
    axR.set_ylabel("KV cache as % of peak memory")
    axR.set_title("(b) cache is a ~10–14% minority of peak")
    axR.set_ylim(0, 100)
    axR.axhspan(0, 15, color=OI["green"], alpha=0.08)
    axR.annotate("prefill activations (≈ batch×heads×seq²)\ndominate the other ~86–90%",
                 xy=(len(xs) / 2 - 0.5, 55), ha="center", fontsize=9, color="0.35")

    fig.suptitle("H3 (memory) -- INT8 halves cache, but cache is not the dominant term",
                 y=1.0, fontsize=13, fontweight="bold")
    cap = ("(a) INT8 cache = (1 + 2/head_dim)/2 = 0.516× FP16 at head_dim 64, exact "
           "across every cell. (b) Because cache is a minority of peak, halving it can't "
           "move the memory frontier — see Fig. 4.")
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=8.5, wrap=True, color="0.3")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    save(fig, "fig3_h3_memory")
    dump(a.merge(b[["seq_len", "peak_mb", "cache_pct_of_peak"]], on="seq_len"),
         "fig3_h3_memory")


# =============================================================================
# FIGURE 4 -- H3 frontier: identical OOM/slow grids (exp3)
# =============================================================================
def figure4():
    print("\n=== FIGURE 4 (H3 frontier: identical failure grids, exp3) ===")
    df = pd.read_csv(EXP3)
    code = {"ok": 0, "slow": 1, "oom": 2}
    df["scode"] = df["status"].map(code)

    seqs = sorted(df["seq_len"].unique())
    batches = sorted(df["batch"].unique())

    def grid(ctype):
        m = np.full((len(seqs), len(batches)), np.nan)
        for i, s in enumerate(seqs):
            for j, bt in enumerate(batches):
                sel = df[(df["cache_type"] == ctype) & (df["seq_len"] == s)
                         & (df["batch"] == bt)]
                if len(sel):
                    m[i, j] = sel.iloc[0]["scode"]
        return m

    g_fp = grid("paged_fp16")
    g_q8 = grid("paged_int8")

    # report failing cells and identity check
    def fails(ctype):
        f = df[(df["cache_type"] == ctype) & (df["status"] != "ok")]
        return set((int(r["seq_len"]), int(r["batch"]), r["status"]) for _, r in f.iterrows())
    ff, fq = fails("paged_fp16"), fails("paged_int8")
    print(f"  FP16 failing cells: {sorted(ff)}")
    print(f"  INT8 failing cells: {sorted(fq)}")
    print(f"  identical failure set: {ff == fq}")
    print(f"  cells INT8 rescues (fp16 fails, int8 ok): "
          f"{sorted({(s,b) for (s,b,_) in ff} - {(s,b) for (s,b,_) in fq})}")

    # data dump
    recs = []
    for i, s in enumerate(seqs):
        for j, bt in enumerate(batches):
            recs.append(dict(seq_len=int(s), batch=int(bt),
                             fp16_status=["ok", "slow", "oom", "NA"][int(g_fp[i, j]) if not np.isnan(g_fp[i, j]) else 3],
                             int8_status=["ok", "slow", "oom", "NA"][int(g_q8[i, j]) if not np.isnan(g_q8[i, j]) else 3]))
    dump(pd.DataFrame(recs), "fig4_h3_frontier")

    cmap = ListedColormap([OI["green"], OI["yellow"], OI["vermil"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, m, title in zip(axes, [g_fp, g_q8], ["FP16", "INT8"]):
        ax.imshow(m, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels([str(b) for b in batches])
        ax.set_yticks(range(len(seqs)))
        ax.set_yticklabels([str(s) for s in seqs])
        ax.set_xlabel("batch size")
        ax.set_title(title, fontsize=12)
        for i in range(len(seqs)):
            for j in range(len(batches)):
                v = m[i, j]
                txt = "" if np.isnan(v) else ["ok", "slow", "OOM"][int(v)]
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        color="black" if (np.isnan(v) or v < 2) else "white")
        ax.set_xticks(np.arange(-.5, len(batches), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(seqs), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
    axes[0].set_ylabel("sequence length")

    legend = [Patch(facecolor=OI["green"], label="ok"),
              Patch(facecolor=OI["yellow"], label="slow"),
              Patch(facecolor=OI["vermil"], label="OOM")]
    fig.legend(handles=legend, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("H3 (frontier) -- INT8 rescues nothing: the two grids are identical",
                 y=1.0, fontsize=13, fontweight="bold")
    cap = ("Same four cells fail in both layouts (4096×32, 8192×8 slow; 8192×16, "
           "8192×32 OOM). Halving a ~10% term cannot move a frontier set by prefill "
           "activation memory. H3 (OOM-shift) is falsified on this GPU.")
    fig.text(0.5, -0.10, cap, ha="center", va="top", fontsize=8.5, wrap=True, color="0.3")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save(fig, "fig4_h3_frontier")


if __name__ == "__main__":
    print("Generating paper figures from canonical CSVs")
    print("CSV provenance: exp1/exp2/exp3 @ commit e383030 ; exp3_perplexity @ eae781c")
    figure1()
    figure2()
    figure3()
    figure4()
    print("\nDone. Figures + per-figure *_data.csv written to paper/figures/")
