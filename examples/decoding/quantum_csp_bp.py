"""Run BP-OSD decoding of quantum CSP codes and average over batches 1..14."""

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
        per_batch_means = []  # list of arrays, shape (len(ps),) per batch
        per_batch_counts = []  # number of codes found in each batch

        for batch in tqdm(range(1, 15), desc=f"N={N}: batches", leave=False):
            per_code_rates = []  # list of arrays shape (len(ps),) per code_id

            for code_id in tqdm(
                range(100), desc=f"  batch={batch} code_ids", leave=False
            ):
                try:
                    _, x_stabs, z_stabs = load_csp_code(
                        num_qubits=N, batch=batch, code_id=code_id
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
                    pauli_bias=[1, 0, 0],
                )

                # Evaluate at each physical error rate
                results = [ler_func(p) for p in ps]  # list of (logical_rate, stderr)
                logical_rates, _stderrs = map(np.array, zip(*results))
                per_code_rates.append(logical_rates)

            # Mean over codes within this batch
            if per_code_rates:
                per_code_rates = np.vstack(
                    per_code_rates
                )  # shape: (n_codes_in_batch, len(ps))
                batch_mean = per_code_rates.mean(axis=0)
                per_batch_means.append(batch_mean)
                per_batch_counts.append(per_code_rates.shape[0])

        n_batches = len(per_batch_means)
        if n_batches == 0:
            print(f"No codes found across batches for N={N}. Skipping plot for this N.")
            continue

        # Equal-weight average over batches
        batch_means = np.vstack(per_batch_means)  # (n_batches, len(ps))
        avg_over_batches = batch_means.mean(axis=0)

        # Error bars: standard error of the mean across batches
        if n_batches > 1:
            sem_over_batches = batch_means.std(axis=0, ddof=1) / np.sqrt(n_batches)
        else:
            sem_over_batches = np.zeros_like(avg_over_batches)

        total_codes = sum(per_batch_counts) if per_batch_counts else 0
        plt.errorbar(
            ps,
            avg_over_batches,
            yerr=sem_over_batches,
            marker="o",
            linestyle="-",
            label=f"N={N} (B={n_batches}, codes={total_codes})",
        )

    plt.xlabel("Physical error rate $p$")
    plt.ylabel("Logical error rate (avg over batches)")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which="both", ls=":")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig("csp_bp_osd_avg.pdf", dpi=300)


if __name__ == "__main__":
    main()
