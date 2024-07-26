"""This script launches quantum decoding on Compute Canada clusters."""

import os
import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
import qecstruct as qec

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Append paths using environment variables or hardcoded fallbacks
project_root = os.getenv("MDOPT_PATH", "/home/bereza/projects/def-ko1/bereza/mdopt")
examples_path = os.getenv(
    "MDOPT_EXAMPLES_PATH", "/home/bereza/projects/def-ko1/bereza/mdopt/examples"
)

sys.path.append(project_root)
sys.path.append(examples_path)

try:
    from mdopt.mps.utils import create_custom_product_state, marginalise
    from mdopt.contractor.contractor import mps_mpo_contract
    from mdopt.optimiser.utils import SWAP, COPY_LEFT, XOR_BULK, XOR_LEFT, XOR_RIGHT
    from examples.decoding.decoding import apply_constraints, apply_bitflip_bias
    from examples.decoding.decoding import (
        decode_css,
        pauli_to_mps,
        css_code_checks,
        css_code_logicals,
        css_code_logicals_sites,
        css_code_constraint_sites,
        generate_pauli_error_string,
    )
except ImportError as e:
    logging.error(
        "Failed to import required modules. Ensure paths are correct.", exc_info=True
    )
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Surface code decoding script.")
    parser.add_argument(
        "--lattice_size",
        type=int,
        required=True,
        help="Lattice size for the surface code.",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        required=True,
        help="Number of experiments to run.",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Seed for the random number generator."
    )
    parser.add_argument(
        "--max_bond_dims",
        nargs="+",
        type=int,
        required=True,
        help="List of maximum bond dimensions.",
    )
    parser.add_argument(
        "--error_rates",
        nargs="+",
        type=float,
        required=True,
        help="List of error rates.",
    )
    return parser.parse_args()


def run_experiment(lattice_size, num_experiments, seed, max_bond_dims, error_rates):
    seed_seq = np.random.SeedSequence(seed)
    failures_statistics = {}

    for chi_max in max_bond_dims:
        logging.info(f"CHI_MAX = {chi_max}")
        for error_rate in tqdm(error_rates):
            failures = []

            for _ in range(num_experiments):
                new_seed = seed_seq.spawn(1)[0]
                rng = np.random.default_rng(new_seed)
                random_integer = rng.integers(1, 10**8 + 1)
                experiment_seed = random_integer

                try:
                    failure = run_single_experiment(
                        lattice_size, chi_max, error_rate, experiment_seed
                    )
                    failures.append(failure)
                except Exception as e:
                    logging.error(
                        f"Experiment failed with error: {str(e)}", exc_info=True
                    )
                    failures.append(1)

            failures_statistics[lattice_size, chi_max, error_rate] = failures

    return failures_statistics


def run_single_experiment(lattice_size, chi_max, error_rate, seed):
    rep_code = qec.repetition_code(lattice_size)
    surface_code = qec.hypergraph_product(rep_code, rep_code)

    error = generate_pauli_error_string(len(surface_code), error_rate, seed=seed)
    error_mps = pauli_to_mps(error)

    _, success = decode_css(
        code=surface_code,
        error=error_mps,
        chi_max=chi_max,
        bias_type="Depolarizing",
        bias_prob=error_rate,
        renormalise=True,
        silent=True,
        contraction_strategy="Optimised",
    )

    return 1 - success


def save_failures_statistics(failures_statistics):
    for key, failures in failures_statistics.items():
        lattice_size, chi_max, error_rate = key
        failures_key = (
            f"latticesize{lattice_size}_bonddim{chi_max}_errorrate{error_rate}"
        )
        np.save(f"{failures_key}.npy", failures)
        logging.info(
            f"Completed experiments for {failures_key} with {np.mean(failures)*100:.2f}% failure rate."
        )


def main():
    args = parse_arguments()
    failures_statistics = run_experiment(
        args.lattice_size,
        args.num_experiments,
        args.seed,
        args.max_bond_dims,
        args.error_rates,
    )
    save_failures_statistics(failures_statistics)


if __name__ == "__main__":
    main()
