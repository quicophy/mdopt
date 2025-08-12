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
    except FileNotFoundError:
        code_dir = f"data-csp-codes/batch_{batch}/codes/qubits_{num_qubits}"
        filename = f"code_{code_id}.json"
        path = os.path.join(code_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find {filename} in {code_dir}")
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
    # Sweep these code sizes
    for N in tqdm([30, 40, 50, 60, 70, 80, 90, 100]):

        # Load CSP code data (uses local JSON loader)
        _, x_stabs, z_stabs = load_csp_code(num_qubits=N, batch=1, code_id=4)

        # Build CSS code
        Hx = list_to_parity_matrix(x_stabs, num_qubits=N)
        Hz = list_to_parity_matrix(z_stabs, num_qubits=N)
        code = CSSCode(Hx, Hz)

        # 1) Build the logical-error-rate function
        ler_func = code.get_logical_error_rate_func(
            num_samples=10000,
            max_error_rate=0.2,
            pauli_bias=[1, 0, 0],  # Bitflip channel
        )

        # 2) Sweep physical error rates
        ps = [0.0001, 0.001, 0.002, 0.004, 0.008, 0.01, 0.1]
        results = [ler_func(p) for p in ps]
        logical_rates, stderrs = map(np.array, zip(*results))

        # 3) Plot
        plt.errorbar(
            ps,
            logical_rates,
            yerr=stderrs,
            marker="o",
            linestyle="-",
            label=f"Logical error for {N} qubits",
        )
        plt.xlabel("Physical error rate $p$")
        plt.ylabel("Logical error rate")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True)

    plt.legend(fontsize="small")
    plt.grid(True)
    plt.savefig("csp_bp_osd.pdf", dpi=300)


if __name__ == "__main__":
    main()
