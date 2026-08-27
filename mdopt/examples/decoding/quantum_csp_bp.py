"""
Run BP-OSD decoding of quantum CSP codes and average over codes.

Saves per-code results to bp_results.pkl for fast re-plotting.
Data format (v3):
    {
        'version': 3,
        'base_samples': BASE_SAMPLES,       # samples used for p >= 0.001
        'n_samples_per_p': {p: int, ...},   # effective samples per error rate
        'ps': PS,
        'ns': NS,
        'batch': BATCH,
        N: {
            code_id: np.ndarray shape (len(PS),)   # n_failures per error rate
        },
        ...
    }

At p=1e-4 the ler_func is called REPS_LOW_P times and failures are accumulated,
giving n_samples_per_p[1e-4] = REPS_LOW_P * BASE_SAMPLES (default 500 000).
LER for code c at error rate p_i is n_failures[i] / n_samples_per_p[p_i].

Requires: pip install qldpc
"""

import os
import re
import json
import pickle
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from tqdm import tqdm

# Always run relative to the example assets so that relative paths
# (CACHE_FILE, TN_DIRECTORY, data-csp-codes) resolve correctly regardless
# of where the script is invoked from. The script itself now lives in the
# package, where none of that data is, so this can no longer key off __file__.
from mdopt.examples.paths import decoding_assets, figure as _figure

os.chdir(str(decoding_assets()))

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

CACHE_FILE = "bp_results.pkl"
BASE_SAMPLES = 50_000  # samples per ler_func call
PS = [0.0001, 0.001, 0.01, 0.1]
NS = [30, 40, 50, 60, 70, 80, 90]
BATCH = 9
TN_DIRECTORY = "data-quantum-csp-batch-9"  # source of truth for code IDs

# Repetitions of ler_func at p=1e-4 per lattice size.
# Chosen so that expected failures per code >= 10, using TN LER as upper bound.
# Effective samples = REPS_LOW_P[N] * BASE_SAMPLES.
REPS_LOW_P: dict[int, int] = {
    30: 3,  #  150 000 samples  (TN LER ~7.8e-5 → ~12 expected failures/code)
    40: 11,  #  550 000 samples  (TN LER ~2.0e-5 → ~11 expected failures/code)
    50: 19,  #  950 000 samples  (TN LER ~1.1e-5 → ~10 expected failures/code)
    60: 57,  # 2 850 000 samples (TN LER ~3.5e-6 → ~10 expected failures/code)
    70: 12,  #  600 000 samples  (TN LER ~1.8e-5 → ~11 expected failures/code)
    80: 24,  # 1 200 000 samples (TN LER ~8.4e-6 → ~10 expected failures/code)
    90: 170,  # 8 500 000 samples (TN LER ~1.2e-6 → ~10 expected failures/code)
}


# Effective samples per (N, error_rate) — varies by N at p=1e-4
def n_samples_for(N: int, p: float) -> int:
    return REPS_LOW_P[N] * BASE_SAMPLES if p == 0.0001 else BASE_SAMPLES


def load_csp_code(num_qubits: int, batch: int, code_id: int):
    for prefix in ["data-csp-codes", "examples/decoding/data-csp-codes"]:
        path = os.path.join(
            prefix,
            f"batch_{batch}",
            "codes",
            f"qubits_{num_qubits}",
            f"code_{code_id}.json",
        )
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            return data["num_qubits"], data["x_stabs"], data["z_stabs"]
    raise FileNotFoundError(
        f"Code not found: N={num_qubits}, batch={batch}, id={code_id}"
    )


def list_tn_code_ids(num_qubits: int, batch: int, tn_directory: str) -> list[int]:
    """
    Return the sorted list of code IDs present in the TN decoder data directory
    for the given (num_qubits, batch), so BP-OSD runs on identical code instances.
    """
    pat = re.compile(
        rf"^latticesize{num_qubits}_bonddim\d+_errorrate[0-9.]+(?:e[+-]?\d+)?_"
        r"errormodelBitflip_bias_prob[0-9.]+(?:e[+-]?\d+)?_numexperiments\d+_"
        r"tolerance[0-9.]+(?:e[+-]?\d+)?_cut[0-9.]+(?:e[+-]?\d+)?_"
        rf"batch{batch}_codeid(\d+)_seed\d+\.pkl$"
    )
    ids: set[int] = set()
    if os.path.isdir(tn_directory):
        for fname in os.listdir(tn_directory):
            m = pat.match(fname)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def list_to_parity_matrix(stabs, num_qubits):
    H = np.zeros((len(stabs), num_qubits), dtype=int)
    for row, stab in enumerate(stabs):
        H[row, stab] = 1
    return H % 2


def run_or_load() -> dict:
    """Run BP+OSD (or load from cache) and return v3 results dict."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached results from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        if data.get("version") == 3:
            return data
        print("Cache is not v3 — regenerating.")

    from qldpc.codes import CSSCode

    results: dict = {
        "version": 3,
        "base_samples": BASE_SAMPLES,
        "reps_low_p": REPS_LOW_P,
        "ps": PS,
        "ns": NS,
        "batch": BATCH,
    }

    for N in NS:
        results[N] = {}
        code_ids = list_tn_code_ids(N, BATCH, TN_DIRECTORY)
        reps = REPS_LOW_P[N]
        print(
            f"N={N}: {len(code_ids)} TN-matched codes, "
            f"p=1e-4 uses {reps * BASE_SAMPLES:,} samples "
            f"({reps} reps × {BASE_SAMPLES:,})"
        )
        for code_id in tqdm(code_ids, desc=f"N={N}"):
            try:
                _, x_stabs, z_stabs = load_csp_code(N, BATCH, code_id)
            except FileNotFoundError:
                continue

            Hx = list_to_parity_matrix(x_stabs, N)
            Hz = list_to_parity_matrix(z_stabs, N)
            code = CSSCode(Hx, Hz)

            # One ler_func with BASE_SAMPLES; call it reps times for p=1e-4.
            ler_func = code.get_logical_error_rate_func(
                num_samples=BASE_SAMPLES,
                max_error_rate=max(PS),
                pauli_bias=[1, 0, 0],
            )

            n_failures = np.zeros(len(PS), dtype=int)
            for i, p in enumerate(PS):
                n_reps = reps if p == 0.0001 else 1
                for _ in range(n_reps):
                    ler = ler_func(p)[0]
                    n_failures[i] += round(ler * BASE_SAMPLES)

            results[N][code_id] = n_failures

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved cache to {CACHE_FILE}")
    return results


def summarise(results: dict):
    """
    Two-level averaging: one LER per code, then mean-of-means + SEM.
    Handles per-N sample counts at p=1e-4 from reps_low_p.
    """
    from scipy.stats import sem as scipy_sem

    ps = results.get("ps", PS)
    base = results.get("base_samples", BASE_SAMPLES)
    reps_low_p = results.get("reps_low_p", {N: 1 for N in NS})

    summary = {}
    for N in results.get("ns", NS):
        code_dict = results.get(N, {})
        if not code_dict:
            continue
        reps = reps_low_p.get(N, 1)
        n_samp = np.array(
            [reps * base if p == 0.0001 else base for p in ps], dtype=float
        )
        ler_mat = np.array(
            [np.array(arr, dtype=float) / n_samp for arr in code_dict.values()],
            dtype=float,
        )
        avg = ler_mat.mean(axis=0)
        se = scipy_sem(ler_mat, axis=0) if len(ler_mat) > 1 else np.zeros_like(avg)
        summary[N] = (avg, se)
    return ps, summary


def main():
    results = run_or_load()
    ps, summary = summarise(results)

    cmap = mpl.colormaps["viridis_r"]
    ns = results.get("ns", NS)
    norm = Normalize(vmin=0, vmax=len(ns) - 1)

    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    for idx, N in enumerate(ns):
        if N not in summary:
            print(f"No data for N={N}, skipping.")
            continue
        avg, se = summary[N]
        n_codes = len(results.get(N, {}))

        ax.errorbar(
            ps,
            avg,
            yerr=se,
            fmt="o--",
            label=rf"$N={N}$ (num\_codes$={n_codes}$)",
            linewidth=1.5,
            markersize=4,
            capsize=2,
            color=cmap(norm(idx)),
        )

    p_ref = np.array(ps)
    ax.plot(
        p_ref,
        p_ref,
        "--",
        marker=None,
        color="#1f77b4",
        linewidth=1.5,
        label=r"$p_L = p$",
    )

    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel("Logical error rate (avg over codes)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e1)
    ax.grid(True, which="both", ls=":", linewidth=0.6)
    ax.legend(fontsize=6)

    fig.savefig(str(_figure("csp-bp-osd-average.pdf")), dpi=300, bbox_inches="tight")
    print("Saved csp-bp-osd-average.pdf")


if __name__ == "__main__":
    main()
