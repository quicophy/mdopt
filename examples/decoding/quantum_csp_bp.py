"""This script runs BP-OSD decoding of quantum CSP codes."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from qldpc.codes import CSSCode


def load_csp_code(num_qubits: int, batch: int, code_id: int):
    """
    Load the specified JSON CSP code for (num_qubits, batch, code_id).
    Looks in:
      - examples/decoding/data-csp-codes/batch_{batch}/codes/qubits_{num_qubits}/code_{code_id}.json
      - data-csp-codes/batch_{batch}/codes/qubits_{num_qubits}/code_{code_id}.json
    """
    try:
        code_dir = (
            f"examples/decoding/data-csp-codes/batch_{batch}/codes/qubits_{num_qubits}"
        )
        filename = f"code_{code_id}.json"
        path = os.path.join(code_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find {filename} in {code_dir}")
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        code_dir = f"data-csp-codes/batch_{batch}/codes/qubits_{num_qubits}"
        filename = f"code_{code_id}.json"
        path = os.path.join(code_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find {filename} in {code_dir}") from exc
        with open(path, "r") as f:
            data = json.load(f)

    return data["num_qubits"], data["x_stabs"], data["z_stabs"]


def list_to_parity_matrix(stabs, num_qubits=None):
    """
    Convert a list of stabilizers (each a list of qubit indices) into
    an (m x n) binary matrix over GF(2), where m = #stabilizers and
    n = #qubits.
    """
    # infer number of qubits if not given
    if num_qubits is None:
        max_index = max((idx for stab in stabs for idx in stab), default=-1)
        num_qubits = max_index + 1

    H = np.zeros((len(stabs), num_qubits), dtype=int)
    for row, stab in enumerate(stabs):
        H[row, stab] = 1
    return H % 2


def main():
    # Physical error rates to sweep
    ps = [0.0001, 0.001, 0.002, 0.004, 0.008, 0.01, 0.1]

    plt.figure(figsize=(7, 5))

    # Sweep these code sizes
    for N in [30, 40, 50, 60, 70, 80, 90, 100]:
        per_code_rates = []  # will be list of arrays shape (len(ps),) per code_id
        per_code_stderrs = (
            []
        )  # optional: keep if you later want to combine uncertainties

        # Try code_id = 0..99 (inclusive); skip those not found
        found_ids = []
        for code_id in tqdm(range(100), desc=f"code ids for {N} qubits"):
            try:
                _, x_stabs, z_stabs = load_csp_code(
                    num_qubits=N, batch=1, code_id=code_id
                )
            except FileNotFoundError:
                continue  # skip missing codes

            Hx = list_to_parity_matrix(x_stabs, num_qubits=N)
            Hz = list_to_parity_matrix(z_stabs, num_qubits=N)
            code = CSSCode(Hx, Hz)

            # Build the logical-error-rate function for this specific code
            ler_func = code.get_logical_error_rate_func(
                num_samples=50000,
                max_error_rate=0.1,
                pauli_bias=[1, 0, 0],  # Bit-flip channel (the probs are for X,Y,Z)
            )

            # Evaluate at each physical error rate
            results = [ler_func(p) for p in ps]  # list of (logical_rate, stderr)
            logical_rates, stderrs = map(np.array, zip(*results))

            per_code_rates.append(logical_rates)
            per_code_stderrs.append(stderrs)
            found_ids.append(code_id)

        n_found = len(per_code_rates)
        if n_found == 0:
            print(f"[warn] No codes found for N={N}. Skipping plot for this N.")
            continue

        per_code_rates = np.vstack(per_code_rates)  # shape: (n_found, len(ps))

        # Average logical error across found code IDs
        avg_logical = per_code_rates.mean(axis=0)

        # Error bars: standard error of the mean across code IDs
        if n_found > 1:
            sem_across_codes = per_code_rates.std(axis=0, ddof=1) / np.sqrt(n_found)
        else:
            sem_across_codes = np.zeros_like(avg_logical)

        plt.errorbar(
            ps,
            avg_logical,
            yerr=sem_across_codes,
            marker="o",
            linestyle="-",
            label=f"N={N} (codes={n_found})",
        )

    plt.xlabel("Physical error rate $p$")
    plt.ylabel("Logical error rate (mean over code IDs)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which="both", ls=":")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig("csp_bp_osd.pdf", dpi=300)


if __name__ == "__main__":
    main()
