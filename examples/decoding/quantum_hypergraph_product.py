"""This script launches quantum decoding on Compute Canada clusters."""

import os
import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
import qecstruct as qc

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Append paths using environment variables or hardcoded fallbacks
project_root_beluga = os.getenv(
    "MDOPT_PATH", "/home/bereza/projects/def-ko1/bereza/mdopt"
)
project_root_cedar = os.getenv(
    "MDOPT_PATH", "/home/bereza/projects/def-ko1/bereza/project-mdopt/mdopt"
)
project_root_graham = os.getenv("MDOPT_PATH", "/home/bereza/mdopt")
project_root_narval = os.getenv(
    "MDOPT_PATH", "/home/bereza/projects/def-ko1/bereza/mdopt"
)

examples_path_beluga = os.getenv(
    "MDOPT_EXAMPLES_PATH", "/home/bereza/projects/def-ko1/bereza/mdopt/examples"
)
examples_path_cedar = os.getenv(
    "MDOPT_EXAMPLES_PATH",
    "/home/bereza/projects/def-ko1/bereza/project-mdopt/mdopt/examples",
)
examples_path_graham = os.getenv("MDOPT_EXAMPLES_PATH", "/home/bereza/mdopt/examples")
examples_path_narval = os.getenv(
    "MDOPT_EXAMPLES_PATH", "/home/bereza/projects/def-ko1/bereza/mdopt/examples"
)

sys.path.append(project_root_graham)
sys.path.append(examples_path_graham)

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
        description="Launch quantum LDPC code decoding on Compute Canada clusters."
    )
    parser.add_argument(
        "--system_size",
        type=int,
        required=True,
        help="System size as the number of bits in the underlying classical code.",
    )
    parser.add_argument(
        "--bond_dim",
        type=int,
        required=True,
        help="Maximum bond dimension to keep during contraction.",
    )
    parser.add_argument(
        "--error_rate", type=float, required=True, help="The error rate."
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


def run_experiment(num_bits, chi_max, error_rate, num_experiments, seed):
    logging.info(
        f"Starting {num_experiments} experiments for NUM_BITS={num_bits}, CHI_MAX={chi_max}, ERROR_RATE={error_rate}, SEED={seed}"
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
                num_bits, chi_max, error_rate, experiment_seed
            )
            failures.append(failure)
        except Exception as e:
            logging.error(f"Experiment {l} failed with error: {str(e)}", exc_info=True)
            failures.append(1)

        logging.info(
            f"Finished experiment {l} for NUM_BITS={num_bits}, CHI_MAX={chi_max}, ERROR_RATE={error_rate}, SEED={seed}"
        )

    return failures


def run_single_experiment(num_bits, chi_max, error_rate, seed):
    CHECK_DEGREE, BIT_DEGREE = 4, 3
    NUM_CHECKS = int(BIT_DEGREE * num_bits / CHECK_DEGREE)
    if num_bits / NUM_CHECKS != CHECK_DEGREE / BIT_DEGREE:
        raise ValueError("The Tanner graph of the code must be bipartite.")

    regular_code = qc.random_regular_code(
        num_bits, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qc.Rng(seed)
    )
    hgp_code = qc.hypergraph_product(regular_code, regular_code)

    error = generate_pauli_error_string(
        len(hgp_code), error_rate, seed=seed, error_model="Depolarizing"
    )
    error = pauli_to_mps(error)

    _, success = decode_css(
        code=hgp_code,
        error=error,
        chi_max=chi_max,
        bias_type="Depolarizing",
        bias_prob=error_rate,
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


def save_failures_statistics(failures, num_bits, chi_max, error_rate, seed):
    failures_statistics = {}
    failures_statistics[(num_bits, chi_max, error_rate)] = failures
    failures_key = (
        f"numbits{num_bits}_bonddim{chi_max}_errorrate{error_rate}_seed{seed}"
    )
    np.save(f"{failures_key}.npy", failures)
    logging.info(
        f"Completed experiments for {failures_key} with {np.mean(failures)*100:.2f}% failure rate."
    )


def main():
    args = parse_arguments()
    failures = run_experiment(
        args.system_size,
        args.bond_dim,
        args.error_rate,
        args.num_experiments,
        args.seed,
    )
    save_failures_statistics(
        failures, args.system_size, args.bond_dim, args.error_rate, args.seed
    )


if __name__ == "__main__":
    main()
