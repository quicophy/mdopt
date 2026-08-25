"""Surface-code 2x2 strategy comparison with paired per-error speedups.

The same N errors are decoded under all 4 strategies, so per-error timing
fluctuations (OS scheduling, cache state, etc.) are correlated across
strategies and cancel in the paired difference. SEMs come from a bootstrap
over per-error pairs (10000 resamples).

Outputs:
  vert_horiz_surface.pdf  — two-panel figure
                            (combined speedup vs p; ablation at chi=PLOT_CHI)
"""

# --- repo-relative paths (this script lives in examples/decoding/plotting/) ---
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DECODING = _os.path.dirname(_HERE)
_EXAMPLES = _os.path.dirname(_DECODING)
_ROOT = _os.path.dirname(_EXAMPLES)
for _p in (_ROOT, _EXAMPLES, _DECODING):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# Relative data dirs / output files below resolve against this directory.
_os.chdir(_DECODING)
_FIGDIR = _os.path.join(_DECODING, "figures")
_os.makedirs(_FIGDIR, exist_ok=True)


def _fig(_name):
    """Absolute path inside examples/decoding/figures/ (cwd-independent)."""
    return _os.path.join(_FIGDIR, _os.path.basename(_name))


# -----------------------------------------------------------------------------

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize

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

DATA_PKL = "vert_horiz_surface_L5_data.pkl"
PLOT_OUT = _fig("vert_horiz_surface.pdf")
PLOT_CHI = 32
N_BOOT = 10000
BOOT_SEED = 0

BASELINE = ("Natural", "Naive")
VERT = ("Natural", "Optimised")
HORIZ = ("Optimised", "Naive")
BOTH = ("Optimised", "Optimised")


def paired_speedup(res, chi, p, target, rng):
    """Paired per-error speedup with bootstrap SEM.

    Returns (mean_pct, sem_pct).  Uses (sum t_b - sum t_t) / sum t_b within
    each bootstrap resample, which equals 1 - mean(t_t)/mean(t_b) but lets
    cancellations from common per-error overhead happen naturally.
    """
    tb = res[(chi, p, *BASELINE)]["per_decode_t"].astype(np.float64)
    tt = res[(chi, p, *target)]["per_decode_t"].astype(np.float64)
    assert tb.shape == tt.shape, f"shape mismatch at ({chi},{p}) {target}"
    n = tb.size
    point = 1.0 - tt.sum() / tb.sum()
    boot = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 256):
        stop = min(start + 256, N_BOOT)
        idx = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = 1.0 - tt[idx].sum(axis=1) / tb[idx].sum(axis=1)
    return 100.0 * point, 100.0 * boot.std(ddof=1)


def main():
    with open(DATA_PKL, "rb") as fh:
        d = pickle.load(fh)
    res = d["results"]
    chis = d["chi_max_list"]
    ps = d["p_list"]
    L = d["lattice_size"]
    n = d["n_qubits"]
    bw_n = d["bandwidth_natural"]
    bw_o = d["bandwidth_optimised"]

    print(f"Surface L={L}: [[{n}, 1, {L}]]")
    print(f"  bandwidth  Natural : max_span={bw_n[0]}, total_span={bw_n[1]}")
    print(f"             Optimised: max_span={bw_o[0]}, total_span={bw_o[1]}")
    print(
        f"  reduction  max {100*(bw_n[0]-bw_o[0])/bw_n[0]:.1f}%, "
        f"total {100*(bw_n[1]-bw_o[1])/bw_n[1]:.1f}%"
    )

    rng = np.random.default_rng(BOOT_SEED)
    sp = {}
    for c in chis:
        for p in ps:
            for label, tgt in [("vert", VERT), ("horiz", HORIZ), ("both", BOTH)]:
                sp[(c, p, label)] = paired_speedup(res, c, p, tgt, rng)

    print(f"\nDecoding-time reduction (%) — paired, bootstrap SEM (N_boot={N_BOOT}):")
    print(
        f"{'chi':>4} {'p':>6} | {'vert only':>14} | {'horiz only':>14} | "
        f"{'both':>14}"
    )
    for c in chis:
        for p in ps:
            v_m, v_s = sp[(c, p, "vert")]
            h_m, h_s = sp[(c, p, "horiz")]
            b_m, b_s = sp[(c, p, "both")]
            print(
                f"{c:>4} {p:>6.0e} | "
                f"{v_m:>7.2f}±{v_s:>5.2f} | "
                f"{h_m:>7.2f}±{h_s:>5.2f} | "
                f"{b_m:>7.2f}±{b_s:>5.2f}"
            )

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax_l, ax_r, ax_ler) = plt.subplots(
        1, 3, figsize=(11.0, 3.0), constrained_layout=True
    )

    # Left: combined-opt speedup vs p, viridis per chi.
    cmap = colormaps["viridis_r"]
    norm = Normalize(vmin=0, vmax=len(chis) - 1)
    for idx, c in enumerate(chis):
        color = cmap(norm(idx))
        means = np.array([sp[(c, p, "both")][0] for p in ps])
        sems = np.array([sp[(c, p, "both")][1] for p in ps])
        ax_l.errorbar(
            ps,
            means,
            yerr=sems,
            fmt="o--",
            color=color,
            label=rf"$\chi_{{\max}}={c}$",
            markersize=4,
            capsize=2,
            linewidth=1.5,
        )
    ax_l.axhline(0, color="k", lw=0.5, ls=":")
    ax_l.set_xscale("log")
    ax_l.set_xlabel(r"Physical error rate $p$")
    ax_l.set_ylabel(r"Combined-opt time reduction (\%)")
    ax_l.set_title(r"Combined: matrex $+$ RCM", fontsize=9)
    ax_l.grid(True, ls=":", linewidth=0.6)
    ax_l.legend(loc="lower right", framealpha=0.9)

    # Right: per-axis ablation at chi=PLOT_CHI.
    ablation = [
        ("vert", r"vertical (matrex) only", "#1f77b4", "s"),
        ("horiz", r"horizontal (RCM) only", "#d62728", "^"),
        ("both", r"both opts", "#2ca02c", "D"),
    ]
    for lbl, name, color, marker in ablation:
        means = np.array([sp[(PLOT_CHI, p, lbl)][0] for p in ps])
        sems = np.array([sp[(PLOT_CHI, p, lbl)][1] for p in ps])
        ax_r.errorbar(
            ps,
            means,
            yerr=sems,
            fmt=marker + "-",
            color=color,
            label=name,
            markersize=4,
            capsize=2,
            linewidth=1.5,
        )
    ax_r.axhline(0, color="k", lw=0.5, ls=":")
    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"Physical error rate $p$")
    ax_r.set_ylabel(r"Time reduction vs baseline (\%)")
    ax_r.set_title(rf"Per-axis ablation, $\chi_{{\max}} = {PLOT_CHI}$", fontsize=9)
    ax_r.grid(True, ls=":", linewidth=0.6)
    ax_r.legend(loc="lower right", framealpha=0.9)

    # Panel C: LER vs p at all chi, Natural (solid) vs RCM (dashed).
    # Use Wilson 95% upper bound for cells with 0 observed failures so all chi
    # show meaningful data even when LER < 1/N.
    Z2 = 1.96**2

    def ler_or_ub(cell):
        ler, N = cell["ler"], cell["n_trials"]
        if cell["ler"] > 0:
            return ler, cell["ler_sem"], False  # value, sem, is_upper_bound
        # Wilson upper bound for 0/N:
        ub = Z2 / (N + Z2)
        return ub, 0.0, True

    for idx, c in enumerate(chis):
        color = cmap(norm(idx))
        for ordering, ls, marker in [("Natural", "-", "o"), ("Optimised", "--", "s")]:
            ys, sems, ubs = [], [], []
            for p in ps:
                v, s, is_ub = ler_or_ub(res[(c, p, ordering, "Naive")])
                ys.append(v)
                sems.append(s)
                ubs.append(is_ub)
            ys = np.asarray(ys)
            sems = np.asarray(sems)
            ubs = np.asarray(ubs)
            # Real points
            real = ~ubs
            if real.any():
                ax_ler.errorbar(
                    np.asarray(ps)[real],
                    ys[real],
                    yerr=sems[real],
                    fmt=marker,
                    ls=ls,
                    color=color,
                    markersize=4,
                    capsize=2,
                    linewidth=1.5,
                )
            # Upper-bound points: downward triangles to indicate "≤ ub"
            if ubs.any():
                ax_ler.scatter(
                    np.asarray(ps)[ubs],
                    ys[ubs],
                    marker="v",
                    facecolors="none",
                    edgecolors=color,
                    s=18,
                    linewidths=0.9,
                )
    # No second connecting pass: the errorbar calls above already join the real
    # points in the intended colour and style. Redrawing here dashed every
    # ordering, hiding the solid Natural curve, and joined the Wilson upper
    # bounds as though they were measured values.

    ax_ler.set_xscale("log")
    ax_ler.set_yscale("log")
    ax_ler.set_xlabel(r"Physical error rate $p$")
    ax_ler.set_ylabel("Logical error rate")
    ax_ler.set_title(r"LER (Naive vert; Natural/RCM)", fontsize=9)
    ax_ler.grid(True, ls=":", linewidth=0.6)
    ler_handles = [
        plt.Line2D(
            [], [], color="k", ls="-", marker="o", markersize=4, label="Natural"
        ),
        plt.Line2D([], [], color="k", ls="--", marker="s", markersize=4, label="RCM"),
        plt.Line2D(
            [],
            [],
            color="grey",
            ls="None",
            marker="v",
            markersize=5,
            markerfacecolor="none",
            label="Wilson 95\\% UB (no fails)",
        ),
    ]
    ax_ler.legend(handles=ler_handles, loc="lower right", framealpha=0.9)

    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")
    print(f"\nSaved {PLOT_OUT}")


if __name__ == "__main__":
    main()
