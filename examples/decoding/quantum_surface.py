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
    from examples.decoding.decoding import (
        decode_css,
        pauli_to_mps,
        generate_pauli_error_string,
    )
except ImportError as e:
    logging.error(
        "Failed to import required modules. Ensure paths are correct.", exc_info=True
    )
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Launch quantum surface code calculations on Compute Canada clusters."
    )
    parser.add_argument(
        "--lattice_size",
        type=int,
        required=True,
        help="Lattice size for the surface code.",
    )
    parser.add_argument(
        "--bond_dim",
        type=int,
        required=True,
        help="Maximum bond dimension to keep during contraction.",
    )
    parser.add_argument(
        "--error_prob", type=float, required=True, help="The error probability."
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        required=True,
        help="The number of experiments to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="The seed for the random number generator.",
    )
    return parser.parse_args()


def run_experiment(lattice_size, chi_max, prob_error, num_experiments, seed):
    logging.info(
        f"Starting {num_experiments} experiments for LATTICE_SIZE={lattice_size}, CHI_MAX={chi_max}, PROB_ERROR={prob_error}, SEED={seed}"
    )

    seed_seq = np.random.SeedSequence(seed)
    failures = []

    for l in tqdm(range(num_experiments)):
        new_seed = seed_seq.spawn(1)[0]
        rng = np.random.default_rng(new_seed)
        random_integer = rng.integers(1, 10**8 + 1)
        experiment_seed = random_integer

        try:
            failure = run_single_experiment(
                lattice_size, chi_max, prob_error, experiment_seed
            )
            failures.append(failure)
        except Exception as e:
            logging.error(f"Experiment {l} failed with error: {str(e)}", exc_info=True)
            failures.append(1)

        logging.info(
            f"Finished experiment {l} for LATTICE_SIZE={lattice_size}, CHI_MAX={chi_max}, PROB_ERROR={prob_error}, SEED={seed}"
        )

    return failures


def run_single_experiment(lattice_size, chi_max, prob_error, seed):
    rep_code = qec.repetition_code(lattice_size)
    surface_code = qec.hypergraph_product(rep_code, rep_code)

    prob_bias = prob_error
    error = generate_pauli_error_string(len(surface_code), prob_error, seed=seed)
    error_mps = pauli_to_mps(error)

    _, success = decode_css(
        code=surface_code,
        error=error_mps,
        chi_max=chi_max,
        bias_type="Depolarizing",
        bias_prob=prob_bias,
        renormalise=True,
        silent=True,
        contraction_strategy="Optimised",
    )

    if success == 1:
        logging.info("Decoding successful.")
        return 0
    else:
        logging.info("Decoding failed.")
        return 1


def save_failures_statistics(failures, lattice_size, chi_max, prob_error, seed):
    failures_statistics = {}
    failures_statistics[(lattice_size, chi_max, prob_error)] = failures
    failures_key = (
        f"latticesize{lattice_size}_bonddim{chi_max}_errorprob{prob_error}_seed{seed}"
    )
    np.save(f"{failures_key}.npy", failures)
    logging.info(
        f"Completed experiments for {failures_key} with {np.mean(failures)*100:.2f}% failure rate."
    )


def main():
    args = parse_arguments()
    failures = run_experiment(
        args.lattice_size,
        args.bond_dim,
        args.error_prob,
        args.num_experiments,
        args.seed,
    )
    save_failures_statistics(
        failures, args.lattice_size, args.bond_dim, args.error_prob, args.seed
    )


if __name__ == "__main__":
    main()
