"""This script launches quantum decoding on Compute Canada clusters."""

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

sys.path.append(project_root_narval)
sys.path.append(examples_path_narval)

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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch quantum surface code decoding on Compute Canada clusters."
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
        "--error_rate", type=float, required=True, help="The error rate."
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        required=True,
        help="The number of experiments to run.",
    )
    parser.add_argument(
        "--error_model",
        type=str,
        required=True,
        help="The error model to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="The seed for the random number generator.",
    )
    return parser.parse_args()


def run_single_experiment(lattice_size, chi_max, error, bias_prob, error_model):
    """Run a single experiment."""
    rep_code = qec.repetition_code(lattice_size)
    surface_code = qec.hypergraph_product(rep_code, rep_code)

    _, success = decode_css(
        code=surface_code,
        error=error,
        chi_max=chi_max,
        bias_type=error_model,
        bias_prob=bias_prob,
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


def generate_errors(lattice_size, error_rate, num_experiments, error_model, seed):
    """Generate errors for the experiments."""
    seed_seq = np.random.SeedSequence(seed)
    errors = []

    rep_code = qec.repetition_code(lattice_size)
    surface_code = qec.hypergraph_product(rep_code, rep_code)

    for _ in range(num_experiments):
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        random_seed = rng.integers(1, 10**8 + 1)
        error = generate_pauli_error_string(
            len(surface_code),
            error_rate,
            seed=random_seed,
            error_model=error_model,
        )
        errors.append(error)

    return errors


def run_experiment(
    lattice_size, chi_max, error_rate, num_experiments, error_model, seed
):
    """Run the experiment consisting of multiple single experiments."""
    logging.info(
        f"Starting {num_experiments} experiments for LATTICE_SIZE={lattice_size},"
        f"CHI_MAX={chi_max}, ERROR_RATE={error_rate}, ERROR_MODEL={error_model}, SEED={seed}"
    )

    failures = []
    errors = generate_errors(
        lattice_size, error_rate, num_experiments, error_model, seed
    )

    for l in tqdm(range(num_experiments)):
        try:
            failure = run_single_experiment(
                lattice_size=lattice_size,
                chi_max=chi_max,
                error=errors[l],
                bias_prob=error_rate,
                error_model=error_model,
            )
            failures.append(failure)
        except Exception as e:
            logging.error(f"Experiment {l} failed with error: {str(e)}", exc_info=True)
            failures.append(1)

        logging.info(
            f"Finished experiment {l} for LATTICE_SIZE={lattice_size}, CHI_MAX={chi_max},"
            f"ERROR_RATE={error_rate}, ERROR_MODEL={error_model}, SEED={seed}"
        )

    return {
        "failures": failures,
        "errors": errors,
        "lattice_size": lattice_size,
        "chi_max": chi_max,
        "error_rate": error_rate,
        "error_model": error_model,
        "seed": seed,
    }


def save_experiment_data(data, lattice_size, chi_max, error_rate, error_model, seed):
    """Save the experiment data."""
    file_key = f"latticesize{lattice_size}_bonddim{chi_max}_errorrate{error_rate}_errormodel{error_model}_seed{seed}.pkl"
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
        args.lattice_size,
        args.bond_dim,
        args.error_rate,
        args.num_experiments,
        args.error_model,
        args.seed,
    )
    save_experiment_data(
        experiment_data,
        args.lattice_size,
        args.bond_dim,
        args.error_rate,
        args.error_model,
        args.seed,
    )


if __name__ == "__main__":
    main()
