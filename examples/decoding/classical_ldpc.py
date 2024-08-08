"""This script launches classical decoding on Compute Canada clusters."""

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
    from mdopt.mps.utils import create_custom_product_state
    from mdopt.optimiser.utils import SWAP, XOR_BULK, XOR_LEFT, XOR_RIGHT
    from examples.decoding.decoding import (
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
    parser = argparse.ArgumentParser(
        description="Launch calculations on Compute Canada clusters."
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


def run_experiment(num_bits, chi_max_contractor, prob_error, num_experiments, seed):
    logging.info(
        f"Starting {num_experiments} experiments for NUM_BITS={num_bits}, CHI_MAX={chi_max_contractor}, PROB_ERROR={prob_error}, SEED={seed}"
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
                num_bits, chi_max_contractor, prob_error, experiment_seed
            )
            failures.append(failure)
        except Exception as e:
            logging.error(f"Experiment {l} failed with error: {str(e)}", exc_info=True)
            failures.append(1)

        logging.info(
            f"Finished experiment {l} for NUM_BITS={num_bits}, CHI_MAX={chi_max_contractor}, PROB_ERROR={prob_error}, SEED={seed}"
        )

    return failures


def run_single_experiment(num_bits, chi_max_contractor, prob_error, seed):
    CHECK_DEGREE, BIT_DEGREE = 4, 3
    NUM_CHECKS = int(BIT_DEGREE * num_bits / CHECK_DEGREE)
    if num_bits / NUM_CHECKS != CHECK_DEGREE / BIT_DEGREE:
        raise ValueError("The Tanner graph of the code must be bipartite.")

    prob_bias = prob_error
    chi_max_dmrg = chi_max_contractor
    cut = 1e-16
    num_dmrg_runs = 1

    code = qec.random_regular_code(
        num_bits, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qec.Rng(seed)
    )
    code_constraint_sites = linear_code_constraint_sites(code)

    initial_codeword, perturbed_codeword = linear_code_prepare_message(
        code, prob_error, error_model=qec.BinarySymmetricChannel, seed=seed
    )
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
        prob_bias_list=prob_bias,
        renormalise=True,
    )

    logging.info("Applying constraints to the perturbed codeword state.")
    perturbed_codeword_state = apply_constraints(
        perturbed_codeword_state,
        code_constraint_sites,
        tensors,
        chi_max=chi_max_contractor,
        renormalise=True,
        result_to_explicit=False,
        strategy="Optimised",
        silent=False,
    )

    logging.info("Decoding the perturbed codeword state using DMRG.")
    dmrg_container, success = decode_message(
        message=perturbed_codeword_state,
        codeword=initial_codeword_state,
        num_runs=num_dmrg_runs,
        chi_max_dmrg=chi_max_dmrg,
        cut=cut,
        silent=True,
    )

    if success == 1:
        logging.info("Decoding successful.")
        return 0
    else:
        logging.info("Decoding failed.")
        return 1


def save_failures_statistics(failures, num_bits, chi_max_contractor, prob_error, seed):
    failures_statistics = {}
    failures_statistics[(num_bits, chi_max_contractor, prob_error)] = failures
    failures_key = f"numbits{num_bits}_bonddim{chi_max_contractor}_errorprob{prob_error}_seed{seed}"
    np.save(f"{failures_key}.npy", failures)
    logging.info(
        f"Completed experiments for {failures_key} with {np.mean(failures)*100:.2f}% failure rate."
    )


def main():
    args = parse_arguments()
    failures = run_experiment(
        args.system_size,
        args.bond_dim,
        args.error_prob,
        args.num_experiments,
        args.seed,
    )
    save_failures_statistics(
        failures, args.system_size, args.bond_dim, args.error_prob, args.seed
    )


if __name__ == "__main__":
    main()
