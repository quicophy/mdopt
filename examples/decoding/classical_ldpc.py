"""This script launches classical decoding on Compute Canada clusters."""

import os
import sys
import pickle
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

sys.path.append(project_root_beluga)
sys.path.append(examples_path_beluga)

try:
    from mdopt.mps.utils import create_custom_product_state
    from mdopt.optimiser.utils import SWAP, XOR_BULK, XOR_LEFT, XOR_RIGHT
    from examples.decoding.decoding import (
        linear_code_parity_matrix_dense,
        linear_code_constraint_sites,
        linear_code_prepare_message,
    )
    from examples.decoding.decoding import (
        apply_bitflip_bias,
        apply_constraints,
        decode_message,
    )
except ImportError as e:
    logging.error(
        "Failed to import required modules. Ensure paths are correct.", exc_info=True
    )
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch classical LDPC code decoding on Compute Canada clusters."
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
        "--error_rate", type=float, required=True, help="The error rate."
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        required=True,
        help="Number of experiments to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed for the random number generator.",
    )
    return parser.parse_args()


def run_single_experiment(
    chi_max,
    error_rate,
    initial_codeword,
    perturbed_codeword,
    code_constraint_sites,
):
    """Run a single experiment."""
    tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]

    initial_codeword_state = create_custom_product_state(
        initial_codeword, form="Right-canonical"
    )
    perturbed_codeword_state = create_custom_product_state(
        perturbed_codeword, form="Right-canonical"
    )

    logging.info("Applying bitflip bias to the perturbed codeword state.")
    perturbed_codeword_state = apply_bitflip_bias(
        mps=perturbed_codeword_state,
        sites_to_bias="All",
        prob_bias_list=error_rate,
        renormalise=True,
    )

    logging.info("Applying constraints to the perturbed codeword state.")
    perturbed_codeword_state = apply_constraints(
        mps=perturbed_codeword_state,
        strings=code_constraint_sites,
        logical_tensors=tensors,
        chi_max=chi_max,
        renormalise=True,
        result_to_explicit=False,
        strategy="Optimised",
        silent=True,
    )

    logging.info("Decoding the perturbed codeword state using DMRG.")
    _, success = decode_message(
        message=perturbed_codeword_state,
        codeword=initial_codeword_state,
        num_runs=50,
        chi_max_dmrg=chi_max,
        cut=1e-12,
        silent=False,
    )

    if success == 1:
        logging.info("Decoding successful.")
        return 0
    else:
        logging.info("Decoding failed.")
        return 1


def run_experiment(num_bits, chi_max, error_rate, num_experiments, seed):
    """Run the experiment consisting of multiple single experiments."""
    seed_seq = np.random.SeedSequence(seed)

    initial_codewords = []
    perturbed_codewords = []
    codes = []
    failures = []

    logging.info(
        f"Starting {num_experiments} experiments for NUM_BITS={num_bits},"
        f"CHI_MAX={chi_max}, ERROR_RATE={error_rate}, SEED={seed}"
    )

    for exp in tqdm(range(num_experiments)):
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        experiment_seed = rng.integers(1, 10**8 + 1)

        check_degree, bit_degree = 4, 3
        num_checks = int(bit_degree * num_bits / check_degree)
        if num_bits / num_checks != check_degree / bit_degree:
            raise ValueError("The Tanner graph of the code must be bipartite.")

        code = qec.random_regular_code(
            num_bits, num_checks, bit_degree, check_degree, qec.Rng(experiment_seed)
        )
        initial_codeword, perturbed_codeword = linear_code_prepare_message(
            code=code,
            error_rate=error_rate,
            error_model=qec.BinarySymmetricChannel,
            seed=experiment_seed,
        )

        initial_codewords.append(initial_codeword)
        perturbed_codewords.append(perturbed_codeword)
        codes.append(linear_code_parity_matrix_dense(code))

        try:
            code_constraint_sites = linear_code_constraint_sites(code)
            failure = run_single_experiment(
                chi_max,
                error_rate,
                initial_codeword,
                perturbed_codeword,
                code_constraint_sites,
            )
            failures.append(failure)
        except Exception as e:
            logging.error(
                f"Experiment {exp} failed with error: {str(e)}", exc_info=True
            )
            failures.append(1)

    return {
        "failures": failures,
        "initial_codewords": initial_codewords,
        "perturbed_codewords": perturbed_codewords,
        "codes": codes,
    }


def save_experiment_data(data, num_bits, chi_max, error_rate, seed):
    """Save the experiment data."""
    file_key = f"numbits{num_bits}_bonddim{chi_max}_errorrate{error_rate}_seed{seed}"
    with open(file_key, "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    logging.info(
        f"Saved experiment data for {file_key} with "
        f"{np.mean(data['failures'])*100:.2f}% failure rate."
    )


def main():
    """Main entry point."""
    args = parse_arguments()
    experiment_data = run_experiment(
        args.system_size,
        args.bond_dim,
        args.error_rate,
        args.num_experiments,
        args.seed,
    )
    save_experiment_data(
        experiment_data, args.system_size, args.bond_dim, args.error_rate, args.seed
    )


if __name__ == "__main__":
    main()
