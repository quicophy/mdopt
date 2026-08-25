"""Aggregate the 2x2 (vertical x horizontal) strategy grid for the
[[72,12,6]] BB code and produce the joint comparison figure + summary.

Loads pickles from
  examples/decoding/data-quantum-bivariate-bicycle/qubit_order_data/
where filenames carrying "vert-Optimised" use contraction_strategy="Optimised"
and all other pickles default to contraction_strategy="Naive".

Outputs:
  - vert_horiz_bb_data.pkl  : aggregated dict + per-seed speedups
  - vert_horiz_bb.pdf       : 2-panel figure at chi=40
                              (decode time vs p; speedup vs p)
  - prints a chi x ablation speedup table
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
PICKLE_OUT = "vert_horiz_bb_data.pkl"
PLOT_OUT = _fig("vert_horiz_bb.pdf")
CHI_LIST = [8, 16, 24, 32, 40]
P_LIST = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
ORDERINGS = ["Natural", "Optimised"]
VERTICALS = ["Naive", "Optimised"]
PLOT_CHI = 40  # focal chi for the 2-panel figure


def load_all():
    by_cell = defaultdict(list)
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith(".pkl"):
            continue
        with open(os.path.join(DATA_DIR, fn), "rb") as fh:
            d = pickle.load(fh)
        vert = d.get("contraction_strategy", "Naive")
        by_cell[(d["chi"], d["p"], d["ordering"], vert)].append(d)
    return by_cell


def aggregate(by_cell):
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
            "vertical": key[3],
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


def per_seed_speedup(agg, chi, p, baseline, target):
    """Per-seed (1 - t_target / t_baseline). Returns mean ± SEM and array."""
    b = agg[(chi, p, *baseline)]
    t = agg[(chi, p, *target)]
    assert (
        b["seeds"] == t["seeds"]
    ).all(), f"seed mismatch baseline={baseline} target={target} at ({chi},{p})"
    per = 1.0 - t["seed_times"] / b["seed_times"]
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(len(per))), per


def main():
    by_cell = load_all()
    agg = aggregate(by_cell)

    expected = [
        (c, p, o, v)
        for c in CHI_LIST
        for p in P_LIST
        for o in ORDERINGS
        for v in VERTICALS
    ]
    missing = [k for k in expected if k not in agg]
    if missing:
        print(f"WARNING: {len(missing)} cells missing: {missing[:5]}…")
    else:
        print(f"All {len(expected)} cells of the 2x2 strategy grid present.")

    # Three ablation paths (baseline always = Naive vert + Natural horiz):
    BASELINE = ("Natural", "Naive")  # no opt
    VERT = ("Natural", "Optimised")  # vertical alone
    HORIZ = ("Optimised", "Naive")  # horizontal alone
    BOTH = ("Optimised", "Optimised")  # both

    speedups = {}  # (chi, p, label) -> (mean, sem)
    for chi in CHI_LIST:
        for p in P_LIST:
            for label, target in [("vert", VERT), ("horiz", HORIZ), ("both", BOTH)]:
                m, s, _ = per_seed_speedup(agg, chi, p, BASELINE, target)
                speedups[(chi, p, label)] = (m, s)

    # ── Console summary ────────────────────────────────────────────────────────
    print(
        f"\nSpeedup % vs baseline (Naive vert + Natural horiz), mean across "
        f"p values (± std):"
    )
    print(f"{'chi':>4} | {'vert only':>14} | {'horiz only':>14} | " f"{'both':>14}")
    for chi in CHI_LIST:
        row = [f"{chi:>4}"]
        for label in ("vert", "horiz", "both"):
            vals = np.array([100 * speedups[chi, p, label][0] for p in P_LIST])
            row.append(f"{vals.mean():>7.2f} ±{vals.std():>5.2f}")
        print(" | ".join(row))

    print(f"\nDecode time at chi={PLOT_CHI} (s):")
    print(
        f"{'p':>7} | {'baseline':>10} | {'+vert':>10} | {'+horiz':>10} | "
        f"{'+both':>10}"
    )
    for p in P_LIST:
        b = agg[(PLOT_CHI, p, *BASELINE)]["avg_time_s"]
        v = agg[(PLOT_CHI, p, *VERT)]["avg_time_s"]
        h = agg[(PLOT_CHI, p, *HORIZ)]["avg_time_s"]
        bo = agg[(PLOT_CHI, p, *BOTH)]["avg_time_s"]
        print(f"{p:>7.0e} | {b:>9.3f}s | {v:>9.3f}s | {h:>9.3f}s | {bo:>9.3f}s")

    # ── Save aggregated pickle ─────────────────────────────────────────────────
    save = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("seed_times", "seeds")}
        for k, v in agg.items()
    }
    with open(PICKLE_OUT, "wb") as fh:
        pickle.dump(
            {
                "aggregated": save,
                "speedups": speedups,
                "chi_list": CHI_LIST,
                "p_list": P_LIST,
                "orderings": ORDERINGS,
                "verticals": VERTICALS,
                "baseline_key": BASELINE,
            },
            fh,
        )
    print(f"\nWrote {PICKLE_OUT}")

    # ── Figure: 2 panels, both relative-speedup ────────────────────────────────
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)

    # Left panel: combined opt's speedup vs p, one curve per chi (viridis).
    cmap = colormaps["viridis_r"]
    norm = Normalize(vmin=0, vmax=len(CHI_LIST) - 1)
    for idx, chi in enumerate(CHI_LIST):
        color = cmap(norm(idx))
        means = np.array([100 * speedups[chi, p, "both"][0] for p in P_LIST])
        sems = np.array([100 * speedups[chi, p, "both"][1] for p in P_LIST])
        ax_l.errorbar(
            P_LIST,
            means,
            yerr=sems,
            fmt="o--",
            color=color,
            label=rf"$\chi_{{\max}}={chi}$",
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

    # Right panel: per-axis ablation at chi=PLOT_CHI.
    ablation_label = {
        "vert": r"vertical (matrex) only",
        "horiz": r"horizontal (RCM) only",
        "both": r"both opts",
    }
    ablation_color = {"vert": "#1f77b4", "horiz": "#d62728", "both": "#2ca02c"}
    ablation_marker = {"vert": "s", "horiz": "^", "both": "D"}
    for lbl in ("vert", "horiz", "both"):
        means = np.array([100 * speedups[PLOT_CHI, p, lbl][0] for p in P_LIST])
        sems = np.array([100 * speedups[PLOT_CHI, p, lbl][1] for p in P_LIST])
        ax_r.errorbar(
            P_LIST,
            means,
            yerr=sems,
            fmt=ablation_marker[lbl] + "-",
            color=ablation_color[lbl],
            label=ablation_label[lbl],
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

    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")
    print(f"Saved {PLOT_OUT}")


if __name__ == "__main__":
    main()
