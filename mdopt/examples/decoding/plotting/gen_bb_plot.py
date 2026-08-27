"""
Generate bb-tn-failure-rate.pdf — MPS-MPO logical error rate for IBM bivariate-bicycle
codes at chi=400.  Canonical thesis style (viridis_r, lw=1.5, fmt="o--").

Also prints a data-quality table: failures observed per (N, p), pooled LER and
SEM, and how many more experiments would be needed to reach 10% relative error.
"""

# --- asset paths (the code lives in the package; the data does not) ---
import os as _os

from mdopt.examples.paths import decoding_assets as _decoding_assets, figure as _fig

# Relative data dirs below resolve against the repo-level examples/decoding/.
_DECODING = str(_decoding_assets())
_os.chdir(_DECODING)
# -----------------------------------------------------------------------------

import sys


import os
import glob
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

mpl.style.use("default")
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)

DATA_DIR = "data/quantum-bivariate-bicycle"
CHI = 400
ERS = [1e-4, 1e-3, 1e-2]
TARGET_REL_ERR = 0.10  # for the "more data needed" estimate

# Map filename-N → [[n_phys, k, d]] label (verified from polynomials in pickles)
CODE_LABEL = {
    36: r"$[\![72, 12, 6]\!]$",
    45: r"$[\![90, 8, 10]\!]$",
    54: r"$[\![108, 8, 10]\!]$",
    72: r"$[\![144, 12, 12]\!]$",
}
N_PHYS = {36: 72, 45: 90, 54: 108, 72: 144}
LATTICES = [36, 45, 54, 72]


def pool(N, p):
    files = sorted(
        glob.glob(f"{DATA_DIR}/latticesize{N}_bonddim{CHI}_errorrate{p}_*.pkl")
    )
    if not files:
        return None
    all_fail = []
    for f in files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        all_fail.extend(d["failures"])
    a = np.array(all_fail, dtype=float)
    a = a[~np.isnan(a)]
    return a, len(files)


def needed_for_rel_err(n_fail, p_ler, target_rel_err=TARGET_REL_ERR):
    """Trials needed at p_ler so that SEM/LER == target_rel_err.

    SEM = sqrt(p(1-p)/n).  Rel err = SEM/p = sqrt((1-p)/(n p)).
    Target rel err r → n = (1-p) / (p r^2).
    """
    if p_ler <= 0:
        return float("inf")
    return (1 - p_ler) / (p_ler * target_rel_err**2)


# ── Compute everything ─────────────────────────────────────────────────────────
rows = []  # for table
ler_dict = {}
err_dict = {}
for N in LATTICES:
    for p in ERS:
        res = pool(N, p)
        if res is None:
            rows.append(
                {
                    "N": N,
                    "p": p,
                    "seeds": 0,
                    "n_exp": 0,
                    "n_fail": 0,
                    "ler": None,
                    "sem": None,
                    "rel_err": None,
                    "n_needed": None,
                }
            )
            continue
        a, n_seeds = res
        n_exp = len(a)
        n_fail = int(a.sum())
        ler = a.mean() if n_exp else 0.0
        sem = np.sqrt(ler * (1 - ler) / n_exp) if n_exp and ler > 0 else 0.0
        rel = sem / ler if ler > 0 else float("inf")
        n_need = needed_for_rel_err(n_fail, ler)
        n_more = max(0, n_need - n_exp) if np.isfinite(n_need) else float("inf")
        rows.append(
            {
                "N": N,
                "p": p,
                "seeds": n_seeds,
                "n_exp": n_exp,
                "n_fail": n_fail,
                "ler": ler,
                "sem": sem,
                "rel_err": rel,
                "n_needed": n_need,
                "n_more": n_more,
            }
        )
        if ler > 0:
            ler_dict[(N, p)] = ler
            err_dict[(N, p)] = sem

# ── Print data-quality table ───────────────────────────────────────────────────
print(
    f"{'code':>16s} {'p':>7s} {'seeds':>6s} {'exp':>10s} {'fails':>6s} "
    f"{'LER':>10s} {'rel.err':>8s} {'need (10%)':>12s} {'add':>12s}"
)
print("-" * 105)
for r in rows:
    label = f"[[{N_PHYS[r['N']]}]]"
    if r["ler"] is None:
        print(
            f"{label:>16s} {r['p']:>7.0e} {'--':>6s} {'--':>10s} {'--':>6s} "
            f"{'--':>10s} {'--':>8s} {'--':>12s} {'--':>12s}   MISSING"
        )
        continue
    rel = f"{r['rel_err']:.0%}" if np.isfinite(r["rel_err"]) else "—"
    need = f"{r['n_needed']:.1e}" if np.isfinite(r["n_needed"]) else "∞"
    add = f"{r['n_more']:.1e}" if np.isfinite(r["n_more"]) else "∞"
    flag = ""
    if r["n_fail"] == 0:
        flag = "   ZERO FAILURES — only an UPPER BOUND"
    elif r["rel_err"] > 0.20:
        flag = "   high rel err"
    print(
        f"{label:>16s} {r['p']:>7.0e} {r['seeds']:>6d} {r['n_exp']:>10d} "
        f"{r['n_fail']:>6d} {r['ler']:>10.2e} {rel:>8s} {need:>12s} "
        f"{add:>12s}{flag}"
    )

# ── Plot ───────────────────────────────────────────────────────────────────────
cmap = mpl.colormaps["viridis_r"]
norm = Normalize(vmin=0, vmax=len(LATTICES) - 1)

fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

for i, N in enumerate(LATTICES):
    pts = [(p, ler_dict[(N, p)], err_dict[(N, p)]) for p in ERS if (N, p) in ler_dict]
    if not pts:
        continue
    ps = np.array([t[0] for t in pts])
    ls = np.array([t[1] for t in pts])
    es = np.array([t[2] for t in pts])
    ax.errorbar(
        ps,
        ls,
        yerr=es,
        fmt="o--",
        color=cmap(norm(i)),
        linewidth=1.5,
        markersize=4,
        capsize=2,
        label=CODE_LABEL[N],
    )

# Pseudo-threshold p_L = p reference line.
p_ref = np.array(sorted(set(ERS)))
ax.plot(
    p_ref,
    p_ref,
    "--",
    color="#1f77b4",
    linewidth=1.5,
    label="Pseudo-threshold equation",
)

ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel("Logical error rate (avg over seeds)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-8, 1e1)
ax.grid(True, ls=":", linewidth=0.6)
ax.legend(fontsize=6)

fig.savefig(_fig("bb-tn-failure-rate.pdf"), dpi=300, bbox_inches="tight")
print("\nSaved bb-tn-failure-rate.pdf")
