import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack, vstack, eye

from ldpc import BpDecoder
from ldpc.monte_carlo_simulation import MonteCarloBscSimulation
from ldpc.code_util import compute_code_parameters


# ----------------------------
# Gallager (j,k)-regular LDPC constructor
# ----------------------------
def gallager_ldpc(n: int, j: int = 3, k: int = 4, seed: int | None = 0) -> csr_matrix:
    """
    Construct a (j,k)-regular Gallager LDPC parity-check matrix H (m x n).
    H is a vertical stack of j submatrices H_t of shape (n/k) x n.
    Each H_t has exactly one '1' per column and exactly k ones per row.
    H_1 = [I  I  ...  I] (k copies), and H_t for t>=2 are random column permutations of H_1.
    """
    if n % k != 0:
        raise ValueError(f"n must be a multiple of k={k}. Got n={n}")
    r = n // k  # rows per submatrix
    rng = np.random.default_rng(seed)

    identity = eye(r, dtype=np.uint8, format="csr")
    h1 = hstack([identity] * k, format="csr")

    blocks = [h1]
    for _ in range(j - 1):
        perm = rng.permutation(n)
        blocks.append(h1[:, perm])

    return vstack(blocks, format="csr")


def main():
    # ----------------------------
    # Build a (3,4)-regular Gallager code
    # ----------------------------
    n = 40  # code length (must be multiple of 4)
    code_seed = 42
    h = gallager_ldpc(n, j=3, k=4, seed=code_seed)
    m, n = h.shape
    n_, k_est, d_est = compute_code_parameters(h)
    rate_lb = 1.0 - 3 / 4

    print(f"H shape: {m} x {n}")
    print(
        f"Estimated code params: n={n_}, k≈{k_est} (R≈{k_est/n_:.3f}, lower bound {rate_lb:.2f}), d_est≈{d_est}"
    )

    col_w = np.asarray(h.sum(axis=0)).ravel()
    row_w = np.asarray(h.sum(axis=1)).ravel()
    assert np.all(col_w == 3) and np.all(row_w == 4), "H is not (3,4)-regular!"

    # ----------------------------
    # Sweep BSC probabilities and run Monte Carlo
    # ----------------------------
    p_grid = np.array([1e-4, 1e-3, 1e-2, 1e-1])

    n_low, n_med, n_high = 150_000, 40_000, 8_000
    max_iters = 80
    bp_method = "product_sum"

    ler, ler_err, run_counts, fail_counts = [], [], [], []
    for p in p_grid:
        p = float(p)
        target = n_low if p <= 2e-3 else (n_med if p <= 2e-2 else n_high)

        dec = BpDecoder(
            h,
            error_rate=p,
            max_iter=max_iters,
            bp_method=bp_method,
            schedule="parallel",
            omp_thread_count=1,
        )

        sim = MonteCarloBscSimulation(
            h, error_rate=p, Decoder=dec, target_run_count=target, tqdm_disable=True
        )
        stats = sim.run()
        ler.append(stats["logical_error_rate"])
        ler_err.append(stats.get("logical_error_rate_eb", 0.0))
        run_counts.append(stats["run_count"])
        fail_counts.append(stats["fail_count"])
        print(
            f"p={p:.4g}  LER={ler[-1]:.3e} ±{ler_err[-1]:.1e}"
            f"  (runs={stats['run_count']}, fails={stats['fail_count']})"
        )

    ler = np.array(ler)
    ler_err = np.array(ler_err)

    # ----------------------------
    # Save results
    # ----------------------------
    os.makedirs("data-classical-ldpc", exist_ok=True)
    save_path = f"data-classical-ldpc/bp_numbits{n}_seed{code_seed}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "n": n,
                "m": m,
                "k": k_est,
                "p_grid": p_grid,
                "ler": ler,
                "ler_err": ler_err,
                "run_counts": run_counts,
                "fail_counts": fail_counts,
                "max_iter": max_iters,
                "bp_method": bp_method,
            },
            f,
        )
    print(f"Saved BP results to {save_path}")

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure()
    plt.errorbar(p_grid, ler, yerr=ler_err, fmt="o-", capsize=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Physical bit-flip rate $p$ (BSC)")
    plt.ylabel("Logical error rate")
    plt.title(
        f"Gallager (3,4) LDPC — n={n_}, k≈{k_est} (R≈{k_est/n_:.2f}),"
        f" max_iter={max_iters}, {bp_method}"
    )
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("data-classical-ldpc/bp_plot.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
