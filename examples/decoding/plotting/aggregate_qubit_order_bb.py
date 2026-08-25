"""Aggregate per-cell qubit-ordering pickles and produce Figure 8.14.

Produces a two-panel figure:
  - Left:  logical error rate vs p, one curve per chi_max, Natural (solid) /
           RCM (dashed) overlap to within MC error — RCM preserves LER.
  - Right: relative wall-time speedup (1 - t_RCM / t_Natural) vs p, one curve
           per chi_max, with per-seed SEM error bars — RCM is 10-18% faster.
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

import os
import pickle
from collections import defaultdict

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

DATA_DIR = "data-quantum-bivariate-bicycle/qubit_order_data"
PICKLE_OUT = "qubit_order_bb_data.pkl"
PLOT_OUT = _fig("qubit_order_bb.pdf")
CHI_LIST = [8, 16, 24, 32, 40]
P_LIST = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
ORDERINGS = ["Natural", "Optimised"]


def load_all():
    """Returns dict[(chi, p, ordering)] -> list of per-seed run dicts."""
    by_cell = defaultdict(list)
    for fn in os.listdir(DATA_DIR):
        if not fn.endswith(".pkl"):
            continue
        with open(os.path.join(DATA_DIR, fn), "rb") as fh:
            d = pickle.load(fh)
        if d.get("contraction_strategy", "Naive") != "Naive":
            continue
        by_cell[(d["chi"], d["p"], d["ordering"])].append(d)
    return by_cell


def aggregate(by_cell):
    """Pool seeds: pooled LER + per-seed avg_time list (for SEM)."""
    out = {}
    for key, runs in by_cell.items():
        runs_sorted = sorted(runs, key=lambda r: r["seed"])
        n_fail = sum(r["n_fail"] for r in runs_sorted)
        n_trials = sum(r["n_trials"] for r in runs_sorted)
        ler = n_fail / n_trials
        ler_sem = float(np.sqrt(ler * (1 - ler) / n_trials))
        seed_times = np.array([r["avg_time_s"] for r in runs_sorted])
        seeds = np.array([r["seed"] for r in runs_sorted])
        out[key] = {
            "chi": key[0],
            "p": key[1],
            "ordering": key[2],
            "ler": ler,
            "ler_sem": ler_sem,
            "avg_time_s": float(seed_times.mean()),
            "avg_time_sem": float(seed_times.std(ddof=1) / np.sqrt(len(seed_times))),
            "n_fail": n_fail,
            "n_trials": n_trials,
            "n_seeds": len(runs_sorted),
            "seed_times": seed_times,
            "seeds": seeds,
        }
    return out


def main():
    by_cell = load_all()
    aggregated = aggregate(by_cell)

    expected = [(c, p, o) for c in CHI_LIST for p in P_LIST for o in ORDERINGS]
    missing = [k for k in expected if k not in aggregated]
    if missing:
        print(f"WARNING: {len(missing)} cells missing: {missing[:3]}…")

    # Speedup per (chi, p): pair Natural & RCM by seed → per-seed speedup → mean ± SEM.
    speedup = {}
    for chi in CHI_LIST:
        for p in P_LIST:
            n = aggregated[chi, p, "Natural"]
            o = aggregated[chi, p, "Optimised"]
            assert (n["seeds"] == o["seeds"]).all(), f"seed mismatch at ({chi},{p})"
            per_seed = 1.0 - o["seed_times"] / n["seed_times"]
            speedup[chi, p] = {
                "mean": float(per_seed.mean()),
                "sem": float(per_seed.std(ddof=1) / np.sqrt(len(per_seed))),
                "per_seed": per_seed,
            }

    # Save aggregated pickle (drop big arrays for cleanliness)
    save = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("seed_times", "seeds")}
        for k, v in aggregated.items()
    }
    save_speedup = {k: {"mean": v["mean"], "sem": v["sem"]} for k, v in speedup.items()}
    with open(PICKLE_OUT, "wb") as fh:
        pickle.dump(
            {
                "aggregated": save,
                "speedup": save_speedup,
                "chi_list": CHI_LIST,
                "p_list": P_LIST,
                "orderings": ORDERINGS,
            },
            fh,
        )
    n_trials_per_cell = sorted({v["n_trials"] for v in aggregated.values()})
    print(
        f"Wrote {PICKLE_OUT}: {len(aggregated)}/{len(expected)} cells, "
        f"per-cell n_trials in {n_trials_per_cell}."
    )

    # ── Console summary ────────────────────────────────────────────────────────
    print(
        f"\n{'chi':>4} {'p':>7} | "
        f"{'Nat LER':>15} | {'RCM LER':>15} | "
        f"{'Nat time (s)':>15} | {'RCM time (s)':>15} | "
        f"{'Speedup %':>11}"
    )
    for c in CHI_LIST:
        for p in P_LIST:
            n = aggregated[c, p, "Natural"]
            o = aggregated[c, p, "Optimised"]
            s = speedup[c, p]
            print(
                f"{c:>4} {p:>7.0e} | "
                f"{n['ler']:.5f}±{n['ler_sem']:.5f} | "
                f"{o['ler']:.5f}±{o['ler_sem']:.5f} | "
                f"{n['avg_time_s']:>7.3f}±{n['avg_time_sem']:>5.3f} | "
                f"{o['avg_time_s']:>7.3f}±{o['avg_time_sem']:>5.3f} | "
                f"{100*s['mean']:>6.2f}±{100*s['sem']:.2f}"
            )

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    cmap = colormaps["viridis_r"]
    norm = Normalize(vmin=0, vmax=len(CHI_LIST) - 1)
    # Natural solid, RCM dashed, as the module docstring above promises.
    style = {"Natural": ("-", "o"), "Optimised": ("--", "s")}

    # Left panel: LER vs p, both orderings (overlap demonstrates LER preservation).
    for idx, chi in enumerate(CHI_LIST):
        color = cmap(norm(idx))
        for ordering in ORDERINGS:
            ls, marker = style[ordering]
            lers = [aggregated[chi, p, ordering]["ler"] for p in P_LIST]
            sems = [aggregated[chi, p, ordering]["ler_sem"] for p in P_LIST]
            ax_l.errorbar(
                P_LIST,
                lers,
                yerr=sems,
                fmt=marker,
                ls=ls,
                color=color,
                markersize=4,
                capsize=2,
                linewidth=1.5,
            )
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel(r"Physical error rate $p$")
    ax_l.set_ylabel("Logical error rate")
    ax_l.grid(True, ls=":", linewidth=0.6)

    # Right panel: relative wall-time speedup vs p, one curve per chi.
    for idx, chi in enumerate(CHI_LIST):
        color = cmap(norm(idx))
        means = np.array([100 * speedup[chi, p]["mean"] for p in P_LIST])
        sems = np.array([100 * speedup[chi, p]["sem"] for p in P_LIST])
        ax_r.errorbar(
            P_LIST,
            means,
            yerr=sems,
            fmt="o--",
            color=color,
            markersize=4,
            capsize=2,
            linewidth=1.5,
        )
    ax_r.axhline(0, color="k", lw=0.5, ls=":")
    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"Physical error rate $p$")
    ax_r.set_ylabel(
        r"Decoding-time reduction $1 - t_{\mathrm{RCM}}/t_{\mathrm{Natural}}$ (\%)"
    )
    ax_r.grid(True, ls=":", linewidth=0.6)

    # Combined legend on left panel: chi colours + ordering linestyles.
    chi_handles = [
        plt.Line2D(
            [],
            [],
            color=cmap(norm(i)),
            ls="--",
            marker="o",
            markersize=4,
            label=rf"$\chi_{{\max}}={chi}$",
        )
        for i, chi in enumerate(CHI_LIST)
    ]
    ord_handles = [
        plt.Line2D(
            [], [], color="k", ls="-", marker="o", markersize=4, label="Natural"
        ),
        plt.Line2D([], [], color="k", ls="--", marker="s", markersize=4, label="RCM"),
    ]
    leg1 = ax_l.legend(handles=chi_handles, loc="lower right", framealpha=0.9)
    ax_l.add_artist(leg1)
    ax_l.legend(handles=ord_handles, loc="upper left", framealpha=0.9)
    ax_r.legend(handles=chi_handles, loc="lower right", framealpha=0.9)

    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")
    print(f"\nSaved {PLOT_OUT}")


if __name__ == "__main__":
    main()
