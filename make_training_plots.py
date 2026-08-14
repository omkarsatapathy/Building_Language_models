"""
make_training_plots.py
======================
Turn `logs/train_micro_log.csv` into a set of publication-quality, story-driven
figures for a GPT-MoE (Mixture-of-Experts) training run on TinyStories.

Run from the project root:
    python make_training_plots.py

All images land in ./output_images/ at 200 DPI (crisp for LinkedIn / slides).

Design notes
------------
* Colors come from a colorblind-validated categorical palette (blue / aqua /
  yellow / green) plus reserved status colors (good green, critical red).
* One y-axis per chart (no dual-axis lies), thin marks, recessive grid,
  direct labels instead of legend boxes where it reads cleaner.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# --------------------------------------------------------------------------- #
# 0. Palette + global style
# --------------------------------------------------------------------------- #
SURFACE   = "#fcfcfb"   # chart surface
PLANE     = "#f9f9f7"   # page plane
INK       = "#0b0b0b"   # primary text
INK2      = "#52514e"   # secondary text
MUTED     = "#898781"   # axis / minor labels
GRID      = "#e1e0d9"   # hairline gridlines
AXIS      = "#c3c2b7"   # baseline / axis

# categorical hues -> the four experts (fixed order, never cycled)
EXPERT_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]  # blue aqua yellow green
GOOD     = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING  = "#fab219"
ACCENT   = "#2a78d6"    # default single-series blue
ACCENT_D = "#184f95"    # dark blue for emphasis

plt.rcParams.update({
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         11,
    "axes.edgecolor":    AXIS,
    "axes.linewidth":    1.0,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.8,
    "axes.axisbelow":    True,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        INK,
    "axes.labelcolor":   INK2,
    "axes.titlecolor":   INK,
    "figure.dpi":        110,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

OUT = Path("output_images")
OUT.mkdir(exist_ok=True)


def style_axes(ax):
    """Recessive chrome: drop top/right spines, soften the rest."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)


def kfmt(x, _pos=None):
    """Format big numbers as e.g. 470K / 1.2M."""
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def ema(series, alpha=0.02):
    """Exponential moving average for a smooth trend line."""
    return series.ewm(alpha=alpha).mean()


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# 1. Load + derive
# --------------------------------------------------------------------------- #
CSV = Path("logs/train_micro_log.csv")
df = pd.read_csv(CSV)

# one micro per step in this run, but collapse to step-level to be safe
step = df.groupby("step", as_index=False).mean(numeric_only=True)
step = step.sort_values("step").reset_index(drop=True)

expert_cols = [c for c in step.columns if c.endswith("_frac") and c.startswith("expert")]
n_experts   = len(expert_cols)
top_k       = 2                          # from MoeConfig
balanced    = top_k / n_experts          # per-expert target routing fraction (0.5)

step["perplexity"]   = np.exp(step["ce_loss"].clip(upper=20))
step["ema_ce"]       = ema(step["ce_loss"])
step["route_imbal"]  = step[expert_cols].std(axis=1)       # 0 == perfectly balanced
step["wall_min"]     = step["wall_time_s"] / 60.0

# headline numbers used across figures + the summary card
ce0, ce1   = step["ce_loss"].iloc[0], step["ce_loss"].iloc[-1]
ppl0, ppl1 = step["perplexity"].iloc[0], step["perplexity"].iloc[-1]
n_steps    = int(step["step"].max()) + 1
wall_min   = step["wall_min"].iloc[-1]
tokens_tot = step["tokens"].sum()
# steady-state throughput ignores the first (compile/warmup) step
steady = step["tok_per_sec"].iloc[1:]
tps_med = steady.median()

print(f"Loaded {len(step)} steps | CE {ce0:.2f}->{ce1:.2f} | PPL {ppl0:,.0f}->{ppl1:.1f}")


# --------------------------------------------------------------------------- #
# 2. HERO — cross-entropy loss with EMA trend + annotations
# --------------------------------------------------------------------------- #
def fig_loss_curve():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    style_axes(ax)

    ax.plot(step["step"], step["ce_loss"], color=ACCENT, lw=0.9, alpha=0.35,
            label="CE loss (per step)")
    ax.plot(step["step"], step["ema_ce"], color=ACCENT_D, lw=2.4,
            label="EMA trend")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss  (log scale)")
    ax.set_title("Learning to tell stories: the loss falls ~7.6×",
                 fontsize=15, fontweight="bold", pad=14)

    # start / end callouts
    ax.scatter([0], [ce0], color=CRITICAL, zorder=5, s=45)
    ax.annotate(f"start  {ce0:.2f}\nPPL {ppl0:,.0f}",
                (0, ce0), xytext=(28, -6), textcoords="offset points",
                color=INK2, fontsize=10, va="top")
    ax.scatter([n_steps - 1], [ce1], color=GOOD, zorder=5, s=45)
    ax.annotate(f"final  {ce1:.2f}\nPPL {ppl1:.1f}",
                (n_steps - 1, ce1), xytext=(-12, 34), textcoords="offset points",
                ha="right", color=INK2, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.text(0.5, -0.02,
             f"{n_steps:,} steps · {wall_min:.1f} min wall-clock · {kfmt(tokens_tot)} tokens seen",
             ha="center", color=MUTED, fontsize=9.5)
    save(fig, "01_loss_curve.png")


# --------------------------------------------------------------------------- #
# 3. Perplexity — the intuitive twin of loss
# --------------------------------------------------------------------------- #
def fig_perplexity():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    style_axes(ax)
    ax.plot(step["step"], step["perplexity"], color=ACCENT, lw=2.0)
    ax.fill_between(step["step"], step["perplexity"], step["perplexity"].min(),
                    color=ACCENT, alpha=0.08)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.set_xlabel("training step")
    ax.set_ylabel("perplexity  (log scale)")
    ax.set_title("Perplexity: from ~53,000 guesses to ~4 per token",
                 fontsize=15, fontweight="bold", pad=14)
    ax.annotate(f"{ppl1:.1f}", (n_steps - 1, ppl1), xytext=(-10, 12),
                textcoords="offset points", ha="right",
                color=GOOD, fontweight="bold", fontsize=12)
    save(fig, "02_perplexity.png")


# --------------------------------------------------------------------------- #
# 4. THE MoE MONEY SHOT — expert routing balances out over training
# --------------------------------------------------------------------------- #
def fig_expert_balance():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    style_axes(ax)

    for i, col in enumerate(expert_cols):
        sm = ema(step[col], alpha=0.02)
        ax.plot(step["step"], step[col], color=EXPERT_COLORS[i], lw=0.7, alpha=0.18)
        ax.plot(step["step"], sm, color=EXPERT_COLORS[i], lw=2.2,
                label=f"expert {i}")

    ax.axhline(balanced, color=INK2, lw=1.2, ls=(0, (5, 4)))
    ax.annotate(f"perfect balance = top_k / n_experts = {balanced:.2f}",
                (0, balanced), xytext=(6, 8), textcoords="offset points",
                color=INK2, fontsize=9.5)

    # lines all pile onto 0.5 at the right, so a legend beats direct labels here
    ax.legend(frameon=False, loc="upper right", ncol=4, fontsize=10,
              handlelength=1.2, columnspacing=1.2)
    ax.set_xlabel("training step")
    ax.set_ylabel("fraction of tokens routed to expert")
    ax.set_title("Load balancing works: 4 experts converge to an even split",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlim(0, n_steps)
    save(fig, "03_expert_balance.png")


# --------------------------------------------------------------------------- #
# 5. Routing imbalance decays toward zero (single clean signal)
# --------------------------------------------------------------------------- #
def fig_imbalance():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    style_axes(ax)
    ax.plot(step["step"], step["route_imbal"], color=MUTED, lw=0.7, alpha=0.4)
    ax.plot(step["step"], ema(step["route_imbal"], 0.02), color=CRITICAL, lw=2.4)
    ax.axhline(0, color=GOOD, lw=1.2, ls=(0, (5, 4)))
    ax.set_xlabel("training step")
    ax.set_ylabel("std-dev across experts")
    ax.set_title("Routing imbalance collapses as the aux loss does its job",
                 fontsize=15, fontweight="bold", pad=14)
    i0 = step["route_imbal"].iloc[:20].mean()
    i1 = step["route_imbal"].iloc[-20:].mean()
    ax.annotate(f"early ≈ {i0:.3f}", (10, i0), xytext=(20, 10),
                textcoords="offset points", color=INK2, fontsize=10)
    ax.annotate(f"late ≈ {i1:.3f}", (n_steps - 1, i1), xytext=(-10, 18),
                textcoords="offset points", ha="right", color=GOOD,
                fontweight="bold", fontsize=10)
    save(fig, "04_routing_imbalance.png")


# --------------------------------------------------------------------------- #
# 6. Final expert utilization bars vs. the ideal
# --------------------------------------------------------------------------- #
def fig_final_utilization():
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    style_axes(ax)
    finals = [step[c].iloc[-50:].mean() for c in expert_cols]
    xs = np.arange(n_experts)
    bars = ax.bar(xs, finals, width=0.62, color=EXPERT_COLORS,
                  edgecolor=SURFACE, linewidth=2, zorder=3)
    for x, v in zip(xs, finals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", color=INK,
                fontweight="bold", fontsize=11)
    ax.axhline(balanced, color=INK2, lw=1.4, ls=(0, (5, 4)), zorder=4)
    ax.text(n_experts - 0.5, balanced + 0.01, f"ideal {balanced:.2f}",
            ha="right", color=INK2, fontsize=10)
    ax.set_xticks(xs, [f"expert {i}" for i in range(n_experts)])
    ax.set_ylabel("mean routing fraction (last 50 steps)")
    ax.set_ylim(0, max(finals) * 1.22)
    ax.set_title("Every expert pulls its weight at convergence",
                 fontsize=15, fontweight="bold", pad=14)
    save(fig, "05_final_utilization.png")


# --------------------------------------------------------------------------- #
# 7. Routing heatmap: experts x time (shows the drift then lock-in)
# --------------------------------------------------------------------------- #
def fig_routing_heatmap():
    n_bins = 60
    step["bin"] = pd.cut(step["step"], bins=n_bins, labels=False)
    mat = step.groupby("bin")[expert_cols].mean().T.values  # experts x bins
    dev = mat - balanced                                     # signed deviation from 0.5
    vlim = np.abs(dev).max()                                 # symmetric diverging scale
    fig, ax = plt.subplots(figsize=(11, 3.6))
    # diverging blue<->red centered on gray: cool = under-used, warm = over-used
    im = ax.imshow(dev, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   interpolation="nearest")
    ax.set_yticks(range(n_experts), [f"expert {i}" for i in range(n_experts)])
    xt = np.linspace(0, n_bins - 1, 6)
    ax.set_xticks(xt, [f"{int(t/(n_bins-1)*n_steps):,}" for t in xt])
    ax.set_xlabel("training step")
    ax.set_title("Deviation from an even split — early imbalance, then a flat wall of balance",
                 fontsize=14, fontweight="bold", pad=12)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cb.set_label("over / under-used  (fraction − 0.50)", color=INK2, fontsize=9)
    cb.outline.set_visible(False)
    save(fig, "06_routing_heatmap.png")


# --------------------------------------------------------------------------- #
# 8. Learning-rate schedule (warmup -> cosine decay)
# --------------------------------------------------------------------------- #
def fig_lr():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    style_axes(ax)
    ax.plot(step["step"], step["lr"], color=ACCENT, lw=2.4)
    ax.fill_between(step["step"], step["lr"], color=ACCENT, alpha=0.08)
    warm = int((step["lr"].diff() < 0).idxmax())  # first step where LR starts to fall
    ax.axvline(warm, color=WARNING, lw=1.4, ls=(0, (4, 4)))
    ax.annotate("warmup ends →\ncosine decay begins", (warm, step["lr"].max()),
                xytext=(14, -8), textcoords="offset points",
                color=INK2, fontsize=10, va="top")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*1e4:.1f}e-4"))
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate: linear warmup, then GPT-3-style cosine decay",
                 fontsize=15, fontweight="bold", pad=14)
    save(fig, "07_lr_schedule.png")


# --------------------------------------------------------------------------- #
# 9. Gradient norm stability (with the 1.0 clip line)
# --------------------------------------------------------------------------- #
def fig_grad_norm():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    style_axes(ax)
    ax.plot(step["step"], step["grad_norm"], color=MUTED, lw=0.6, alpha=0.5)
    ax.plot(step["step"], ema(step["grad_norm"], 0.02), color=ACCENT_D, lw=2.2)
    ax.axhline(1.0, color=CRITICAL, lw=1.4, ls=(0, (5, 4)))
    ax.annotate("grad-clip = 1.0", (n_steps - 1, 1.0), xytext=(-10, 8),
                textcoords="offset points", ha="right", color=CRITICAL, fontsize=10)
    gmax = step["grad_norm"].max()
    ax.scatter([step["grad_norm"].idxmax()], [gmax], color=CRITICAL, s=40, zorder=5)
    ax.annotate(f"init spike {gmax:.2f}", (step["grad_norm"].idxmax(), gmax),
                xytext=(16, -2), textcoords="offset points", color=INK2, fontsize=10)
    ax.set_xlabel("training step")
    ax.set_ylabel("global gradient norm")
    ax.set_title("Gradients stay tame: one init spike, then a calm ride down",
                 fontsize=15, fontweight="bold", pad=14)
    save(fig, "08_grad_norm.png")


# --------------------------------------------------------------------------- #
# 10. Throughput distribution (steady-state tokens/sec)
# --------------------------------------------------------------------------- #
def fig_throughput():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    for ax in (axl, axr):
        style_axes(ax)

    axl.plot(step["step"].iloc[1:], steady, color=ACCENT, lw=0.8, alpha=0.55)
    axl.plot(step["step"].iloc[1:], ema(steady, 0.02), color=ACCENT_D, lw=2.2)
    axl.axhline(tps_med, color=GOOD, lw=1.3, ls=(0, (5, 4)))
    axl.annotate(f"median {kfmt(tps_med)} tok/s", (n_steps - 1, tps_med),
                 xytext=(-10, 10), textcoords="offset points", ha="right",
                 color=GOOD, fontweight="bold", fontsize=10)
    axl.yaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    axl.set_xlabel("training step")
    axl.set_ylabel("tokens / second")
    axl.set_title("Throughput after warmup", fontsize=13, fontweight="bold", pad=10)

    axr.hist(steady, bins=40, color=ACCENT, alpha=0.85, edgecolor=SURFACE)
    axr.axvline(tps_med, color=GOOD, lw=1.6)
    axr.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    axr.set_xlabel("tokens / second")
    axr.set_ylabel("steps")
    axr.set_title("Distribution", fontsize=13, fontweight="bold", pad=10)

    fig.suptitle("How fast did it train?", fontsize=15, fontweight="bold", y=1.02)
    save(fig, "09_throughput.png")


# --------------------------------------------------------------------------- #
# 11. CE vs Aux loss — who drives the total? (two stacked panels, one y each)
# --------------------------------------------------------------------------- #
def fig_ce_vs_aux():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    for ax in (a1, a2):
        style_axes(ax)

    a1.plot(step["step"], step["ce_loss"], color=ACCENT, lw=0.8, alpha=0.35)
    a1.plot(step["step"], ema(step["ce_loss"]), color=ACCENT_D, lw=2.2)
    a1.set_ylabel("cross-entropy")
    a1.set_title("Cross-entropy does all the learning; the aux loss just referees",
                 fontsize=15, fontweight="bold", pad=14)

    a2.plot(step["step"], step["aux_loss"], color="#eda100", lw=0.8, alpha=0.4)
    a2.plot(step["step"], ema(step["aux_loss"]), color="#c98500", lw=2.2)
    a2.set_ylabel("aux (load-bal)")
    a2.set_xlabel("training step")
    a2.annotate("nearly flat — balance is maintained, not fought for",
                (n_steps * 0.5, step["aux_loss"].mean()),
                xytext=(0, 14), textcoords="offset points", ha="center",
                color=INK2, fontsize=9.5)
    save(fig, "10_ce_vs_aux.png")


# --------------------------------------------------------------------------- #
# 12. Summary scorecard — the LinkedIn hero tile
# --------------------------------------------------------------------------- #
def fig_scorecard():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axis("off")
    fig.patch.set_facecolor(PLANE)
    ax.set_facecolor(PLANE)

    ax.text(0.02, 0.92, "GPT-MoE on TinyStories — training in one glance",
            fontsize=19, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.02, 0.83,
            f"4-layer decoder · {n_experts} experts (top-{top_k}) · RoPE + QK-norm · SwiGLU · "
            f"{n_steps:,} steps in {wall_min:.1f} min",
            fontsize=11.5, color=INK2, transform=ax.transAxes)

    tiles = [
        ("Final CE loss",   f"{ce1:.2f}",        f"from {ce0:.2f}",         GOOD),
        ("Final perplexity",f"{ppl1:.1f}",       f"from {ppl0:,.0f}",       GOOD),
        ("Loss reduction",  f"{ce0/ce1:.1f}×",   "lower cross-entropy",     ACCENT_D),
        ("Throughput",      f"{kfmt(tps_med)}",  "tokens / sec (median)",   ACCENT_D),
        ("Expert balance",  f"±{step['route_imbal'].iloc[-20:].mean():.3f}",
                                                 f"target {balanced:.2f} each", GOOD),
        ("Tokens seen",     f"{kfmt(tokens_tot)}", "at 65,536 / step",       ACCENT_D),
    ]
    x0, y0, w, h, gx, gy = 0.02, 0.42, 0.30, 0.26, 0.335, 0.34
    for i, (label, val, sub, col) in enumerate(tiles):
        cx = x0 + (i % 3) * gx
        cy = y0 - (i // 3) * gy
        box = FancyBboxPatch((cx, cy), w, h, transform=ax.transAxes,
                             boxstyle="round,pad=0.008,rounding_size=0.02",
                             facecolor=SURFACE, edgecolor=GRID, linewidth=1.2)
        ax.add_patch(box)
        ax.text(cx + 0.02, cy + h - 0.05, label.upper(), fontsize=9.5,
                color=MUTED, fontweight="bold", transform=ax.transAxes)
        ax.text(cx + 0.02, cy + 0.085, val, fontsize=25, fontweight="bold",
                color=col, transform=ax.transAxes)
        ax.text(cx + 0.02, cy + 0.03, sub, fontsize=9.5, color=INK2,
                transform=ax.transAxes)
    save(fig, "00_scorecard.png")


# --------------------------------------------------------------------------- #
# 13. Loss vs wall-clock time (the "was it worth the minutes?" view)
# --------------------------------------------------------------------------- #
def fig_loss_vs_time():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    style_axes(ax)
    ax.plot(step["wall_min"], step["ce_loss"], color=ACCENT, lw=0.8, alpha=0.3)
    ax.plot(step["wall_min"], ema(step["ce_loss"]), color=ACCENT_D, lw=2.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("wall-clock minutes")
    ax.set_ylabel("cross-entropy loss (log)")
    ax.set_title("Return on time: most of the gain lands in the first few minutes",
                 fontsize=15, fontweight="bold", pad=14)
    # mark the 90%-of-total-drop crossing
    target = ce0 - 0.9 * (ce0 - ce1)
    cross = step.loc[step["ce_loss"] <= target, "wall_min"]
    if len(cross):
        t90 = cross.iloc[0]
        ax.axvline(t90, color=GOOD, lw=1.4, ls=(0, (5, 4)))
        ax.annotate(f"90% of the drop by {t90:.1f} min",
                    (t90, target), xytext=(12, 30), textcoords="offset points",
                    color=GOOD, fontweight="bold", fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    save(fig, "11_loss_vs_time.png")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sns.set_style("white")
    print("Rendering figures ->", OUT.resolve())
    fig_scorecard()
    fig_loss_curve()
    fig_perplexity()
    fig_expert_balance()
    fig_imbalance()
    fig_final_utilization()
    fig_routing_heatmap()
    fig_lr()
    fig_grad_norm()
    fig_throughput()
    fig_ce_vs_aux()
    fig_loss_vs_time()
    print(f"\nDone. {len(list(OUT.glob('*.png')))} images in {OUT}/")
