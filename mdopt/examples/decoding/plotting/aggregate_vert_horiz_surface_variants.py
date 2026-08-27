"""Three figure variants for the surface-code vert/horiz comparison.

Variant 1: chi=32 only, 2 panels (ablation + LER).
Variant 2: speedup factor vs chi, one curve per p, 3 ablation paths (vert /
           horiz / both) as separate panels (log-log).
Variant 3: absolute decode time vs p, one panel per chi (2x4 grid for full chi
           sweep).

All variants share the thesis figure style:
    cmap = viridis_r,  linewidth=1.5,  markersize=4,  capsize=2,
    fmt = "marker--",  grid ls=":" lw=0.6,
    rcParams as in gen_failure_rate_5qubit_erasure.py.

All variants use paired per-decode timings for tight bootstrap error bars.
"""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

DATA_PKL = "data/cache/vert_horiz_surface_L5_data.pkl"
HIGHCHI_PKLS = [f"vert_horiz_surface_L5_chi{c}_data.pkl" for c in (64, 128, 256, 512)]
N_BOOT = 10000
PLOT_CHI = 32
ALL_CHIS = [4, 8, 16, 32, 64, 128, 256, 512]

BASELINE = ("Natural", "Naive")
VERT = ("Natural", "Optimised")
HORIZ = ("Optimised", "Naive")
BOTH = ("Optimised", "Optimised")

# Thesis-style plotting primitives
CMAP = colormaps["viridis_r"]
LW = 1.5
MS = 4
CAPSIZE = 2

# Markers per strategy (so panels with multiple strategies stay distinguishable
# beyond colour alone — useful in monochrome printing).
M_BASE = "o"
M_VERT = "s"
M_HORIZ = "^"
M_BOTH = "D"


def _strat_colors(n):
    """n evenly-spaced colours from viridis_r."""
    if n == 1:
        return [CMAP(0.5)]
    return [CMAP(i / (n - 1)) for i in range(n)]


def paired_speedup(res, chi, p, target, rng):
    tb = res[(chi, p, *BASELINE)]["per_decode_t"].astype(np.float64)
    tt = res[(chi, p, *target)]["per_decode_t"].astype(np.float64)
    n = tb.size
    point = 1.0 - tt.sum() / tb.sum()
    boot = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 256):
        stop = min(start + 256, N_BOOT)
        idx = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = 1.0 - tt[idx].sum(axis=1) / tb[idx].sum(axis=1)
    return 100.0 * point, 100.0 * boot.std(ddof=1)


def paired_factor(res, chi, p, target, rng):
    """Returns (t_opt / t_baseline, bootstrap SEM).  <1 means opt is faster."""
    tb = res[(chi, p, *BASELINE)]["per_decode_t"].astype(np.float64)
    tt = res[(chi, p, *target)]["per_decode_t"].astype(np.float64)
    n = tb.size
    point = tt.sum() / tb.sum()
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = tt[idx].sum(axis=1) / tb[idx].sum(axis=1)
    return float(point), float(boot.std(ddof=1))


def paired_time(res, chi, p, strat, rng):
    t = res[(chi, p, *strat)]["per_decode_t"].astype(np.float64)
    n = t.size
    point = t.mean()
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = t[idx].mean(axis=1)
    return point, boot.std(ddof=1)


def _load_merged():
    with open(DATA_PKL, "rb") as fh:
        d = pickle.load(fh)
    res = dict(d["results"])
    ps = d["p_list"]
    for pkl in HIGHCHI_PKLS:
        if not os.path.exists(pkl):
            print(f"WARN: missing {pkl}")
            continue
        with open(pkl, "rb") as fh:
            d2 = pickle.load(fh)
        res.update(d2["results"])
    return res, ps


def main():
    res, ps = _load_merged()
    chis = [
        c
        for c in ALL_CHIS
        if all(
            (c, p, o, v) in res
            for p in ps
            for o in ("Natural", "Optimised")
            for v in ("Naive", "Optimised")
        )
    ]
    print(f"chis loaded: {chis}")
    rng = np.random.default_rng(0)

    # ── Variant 1: chi=32 only, ablation + LER ────────────────────────────────
    fig1, (ax_ab, ax_ler) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), constrained_layout=True
    )
    # Ablation panel — 3 strategies on viridis_r.
    abl_colors = _strat_colors(3)
    ablation = [
        (VERT, "vertical only", abl_colors[0], M_VERT),
        (HORIZ, "horizontal only", abl_colors[1], M_HORIZ),
        (BOTH, "both opts", abl_colors[2], M_BOTH),
    ]
    for tgt, lbl, color, marker in ablation:
        means = np.array([paired_speedup(res, PLOT_CHI, p, tgt, rng)[0] for p in ps])
        sems = np.array([paired_speedup(res, PLOT_CHI, p, tgt, rng)[1] for p in ps])
        ax_ab.errorbar(
            ps,
            means,
            yerr=sems,
            fmt=marker + "--",
            color=color,
            label=lbl,
            markersize=MS,
            capsize=CAPSIZE,
            linewidth=LW,
        )
    ax_ab.axhline(0, color="k", lw=0.5, ls=":")
    ax_ab.set_xscale("log")
    ax_ab.set_xlabel(r"Physical error rate $p$")
    ax_ab.set_ylabel(r"Decoding-time reduction (\%)")
    ax_ab.set_title(rf"$\chi_{{\max}} = {PLOT_CHI}$, ablation", fontsize=9)
    ax_ab.grid(True, ls=":", linewidth=0.6)
    ax_ab.legend(loc="lower right", framealpha=0.9)

    # LER subplot: Natural vs RCM (Naive vert).
    Z2 = 1.96**2

    def ler_or_ub(cell):
        if cell["ler"] > 0:
            return cell["ler"], cell["ler_sem"], False
        return Z2 / (cell["n_trials"] + Z2), 0.0, True

    ord_colors = _strat_colors(2)
    for (ordering, ls, marker), color in zip(
        [("Natural", "--", "o"), ("Optimised", "--", "s")], ord_colors
    ):
        ys, sems, ubs, xs = [], [], [], []
        for p in ps:
            v, s, is_ub = ler_or_ub(res[(PLOT_CHI, p, ordering, "Naive")])
            ys.append(v)
            sems.append(s)
            ubs.append(is_ub)
            xs.append(p)
        ys = np.asarray(ys)
        sems = np.asarray(sems)
        ubs = np.asarray(ubs)
        xs = np.asarray(xs)
        real = ~ubs
        if real.any():
            ax_ler.errorbar(
                xs[real],
                ys[real],
                yerr=sems[real],
                fmt=marker,
                ls="None",
                color=color,
                markersize=MS,
                capsize=CAPSIZE,
            )
        if ubs.any():
            ax_ler.scatter(
                xs[ubs],
                ys[ubs],
                marker="v",
                facecolors="none",
                edgecolors=color,
                s=18,
                linewidths=0.9,
            )
        ax_ler.plot(
            xs,
            ys,
            ls=ls,
            color=color,
            linewidth=LW,
            label=("Natural" if ordering == "Natural" else "RCM"),
        )
    ax_ler.set_xscale("log")
    ax_ler.set_yscale("log")
    ax_ler.set_xlabel(r"Physical error rate $p$")
    ax_ler.set_ylabel("Logical error rate")
    ax_ler.set_title(rf"$\chi_{{\max}} = {PLOT_CHI}$, LER (Naive vert)", fontsize=9)
    ax_ler.grid(True, ls=":", linewidth=0.6)
    ax_ler.legend(loc="lower right", framealpha=0.9)
    fig1.savefig(_fig("surface-vert-horiz-ablation.pdf"), dpi=300, bbox_inches="tight")
    print("Saved surface-vert-horiz-ablation.pdf")

    # ── Variant 2: speedup factor vs chi, 3-panel log-log ─────────────────────
    fig2, (ax2_v, ax2_h, ax2_b) = plt.subplots(
        1, 3, figsize=(10.5, 3.0), sharey=True, constrained_layout=True
    )
    p_colors = _strat_colors(len(ps))
    panels = [
        (ax2_v, VERT, "Vertical only", M_VERT),
        (ax2_h, HORIZ, "Horizontal only", M_HORIZ),
        (ax2_b, BOTH, "Both opts", M_BOTH),
    ]
    for ax, target, title, marker in panels:
        for j, p in enumerate(ps):
            means = np.array([paired_factor(res, c, p, target, rng)[0] for c in chis])
            sems = np.array([paired_factor(res, c, p, target, rng)[1] for c in chis])
            ax.errorbar(
                chis,
                means,
                yerr=sems,
                fmt=marker + "--",
                color=p_colors[j],
                label=rf"$p={p:.0e}$",
                markersize=MS,
                capsize=CAPSIZE,
                linewidth=LW,
            )
        ax.axhline(1, color="k", lw=0.5, ls=":")
        ax.set_xlabel(r"Bond dimension $\chi_{\max}$")
        ax.set_title(title, fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(chis)
        ax.set_xticklabels([str(c) for c in chis])
        ax.grid(True, ls=":", linewidth=0.6)
    ax2_v.set_ylabel(r"Time ratio $t_{\mathrm{opt}}\,/\,t_{\mathrm{baseline}}$")
    ax2_h.legend(loc="lower left", framealpha=0.9)
    fig2.savefig(_fig("qubit-ordering-comparison.pdf"), dpi=300, bbox_inches="tight")
    print("Saved qubit-ordering-comparison.pdf")

    # ── Variant 3: absolute decode time vs p, one panel per chi (2x4) ─────────
    strat_colors = _strat_colors(4)
    strats = [
        (BASELINE, "baseline (no opt)", strat_colors[0], M_BASE),
        (VERT, r"$+$ vertical only", strat_colors[1], M_VERT),
        (HORIZ, r"$+$ horizontal only", strat_colors[2], M_HORIZ),
        (BOTH, "both opts", strat_colors[3], M_BOTH),
    ]
    n_chi = len(chis)
    n_cols = 4
    n_rows = (n_chi + n_cols - 1) // n_cols
    fig3, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.6 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes_flat, chis):
        for strat, lbl, color, marker in strats:
            means = np.array([paired_time(res, c, p, strat, rng)[0] for p in ps])
            sems = np.array([paired_time(res, c, p, strat, rng)[1] for p in ps])
            ax.errorbar(
                ps,
                means,
                yerr=sems,
                fmt=marker + "--",
                color=color,
                label=lbl if c == chis[0] else None,
                markersize=MS,
                capsize=CAPSIZE,
                linewidth=LW,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$\chi_{{\max}} = {c}$", fontsize=9)
        ax.grid(True, ls=":", linewidth=0.6)
    for ax in list(axes_flat)[n_chi:]:
        ax.set_visible(False)
    for ax in (axes[-1] if n_rows > 1 else axes):
        ax.set_xlabel(r"Physical error rate $p$")
    for ax in (axes[:, 0] if n_rows > 1 else [axes[0]]):
        ax.set_ylabel(r"Avg.\ decoding time per shot (s)")
    (axes[0, 0] if n_rows > 1 else axes[0]).legend(
        loc="lower right", framealpha=0.9, fontsize=7
    )
    fig3.savefig(_fig("qubit-ordering-speedup.pdf"), dpi=300, bbox_inches="tight")
    print("Saved qubit-ordering-speedup.pdf")


if __name__ == "__main__":
    main()
