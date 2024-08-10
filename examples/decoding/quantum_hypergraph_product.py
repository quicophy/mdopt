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
    from mdopt.mps.utils import create_custom_product_state
    from mdopt.optimiser.utils import SWAP, XOR_BULK, XOR_LEFT, XOR_RIGHT
    from examples.decoding.decoding import (
        linear_code_constraint_sites,
        linear_code_prepare_message,
    )
    from examples.decoding.decoding import (
        apply_bitflip_bias,
        apply_constraints,
        decode_css,
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
        help="System size as the number of bits.",
    )
    parser.add_argument(
        "--bond_dim",
        type=int,
        required=True,
        help="Maximum bond dimension to keep during contraction.",
    )
    parser.add_argument(
        "--error_rate", type=float, required=True, help="The error probability."
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
        f"Starting {num_experiments} experiments for NUM_BITS={num_bits}, CHI_MAX={chi_max}, error_rate={error_rate}, SEED={seed}"
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
            f"Finished experiment {l} for NUM_BITS={num_bits}, CHI_MAX={chi_max}, error_rate={error_rate}, SEED={seed}"
        )

    return failures


def run_single_experiment(num_bits, chi_max_contractor, error_rate, seed):
    CHECK_DEGREE, BIT_DEGREE = 4, 3
    NUM_CHECKS = int(BIT_DEGREE * num_bits / CHECK_DEGREE)
    if num_bits / NUM_CHECKS != CHECK_DEGREE / BIT_DEGREE:
        raise ValueError("The Tanner graph of the code must be bipartite.")

    prob_bias = error_rate
    chi_max_dmrg = chi_max_contractor
    cut = 1e-16
    num_dmrg_runs = 1

    regular_code = qc.random_regular_code(
        NUM_BITS, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qc.Rng(SEED)
    )
    hgp_code = qc.hypergraph_product(regular_code, regular_code)

    error = generate_pauli_error_string(
        len(hgp_code), ERROR_RATE, seed=SEED, error_model="Depolarizing"
    )
    error = pauli_to_mps(error)

    _, success = decode_css(
        code=hgp_code,
        error=error,
        chi_max=CHI_MAX,
        bias_type="Depolarizing",
        bias_prob=ERROR_RATE,
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
        f"numbits{num_bits}_bonddim{chi_max}_errorprob{error_rate}_seed{seed}"
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
