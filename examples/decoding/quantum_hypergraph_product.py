"""This script launches decoding of quantum hypergraph product codes."""

import os
import sys
import pickle
import logging
import argparse
from multiprocessing import Pool

import numpy as np
import qecstruct as qec
from scipy.stats import sem

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
project_root_iq = os.getenv("MDOPT_PATH", "/home/bereza/mdopt")

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
examples_path_iq = os.getenv("MDOPT_EXAMPLES_PATH", "/home/bereza/mdopt/examples")

sys.path.append(project_root_beluga)
sys.path.append(examples_path_beluga)

try:
    from examples.decoding.decoding import (
        decode_css,
        css_code_stabilisers,
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
        "--bias_prob", type=float, required=True, help="The decoder bias probability."
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
    parser.add_argument(
        "--num_processes",
        type=int,
        required=True,
        help="The number of processes to use in parallel.",
    )
    parser.add_argument(
        "--silent",
        type=bool,
        required=True,
        help="Whether to silence the output.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        required=True,
        help="The numerical tolerance for the MPS within the decoder.",
    )
    parser.add_argument(
        "--cut",
        type=float,
        required=True,
        help="Singular values smaller than that will be discarded in the SVD.",
    )
    return parser.parse_args()


def generate_errors(system_size, error_rate, num_experiments, error_model, seed):
    """Generate errors for the experiments."""
    seed_seq = np.random.SeedSequence(seed)
    errors = []

    check_degree, bit_degree = 4, 3
    num_checks = int(system_size * bit_degree / check_degree)
    if system_size / num_checks != check_degree / bit_degree:
        raise ValueError("The Tanner graph of the code must be bipartite.")

    classical_code = qec.random_regular_code(
        system_size, num_checks, bit_degree, check_degree, qec.Rng(seed)
    )
    qhgp_code = qec.hypergraph_product(classical_code, classical_code)

    for _ in range(num_experiments):
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        error = generate_pauli_error_string(
            len(qhgp_code),
            error_rate,
            rng=rng,
            error_model=error_model,
        )
        errors.append(error)

    return errors


def run_single_experiment(
    system_size, chi_max, error, bias_prob, error_model, silent, tolerance, cut, seed
):
    """Run a single experiment."""
    check_degree, bit_degree = 4, 3
    num_checks = int(system_size * bit_degree / check_degree)
    if system_size / num_checks != check_degree / bit_degree:
        raise ValueError("The Tanner graph of the code must be bipartite.")
    classical_code = qec.random_regular_code(
        system_size, num_checks, bit_degree, check_degree, qec.Rng(seed)
    )
    qhgp_code = qec.hypergraph_product(classical_code, classical_code)

    try:
        container, success = decode_css(
            code=qhgp_code,
            error=error,
            chi_max=chi_max,
            multiply_by_stabiliser=False,
            bias_type=error_model,
            bias_prob=bias_prob,
            renormalise=True,
            silent=silent,
            contraction_strategy="Optimised",
            tolerance=tolerance,
            cut=cut,
        )
    except Exception as e:
        logging.error(f"Error during decoding: {e}", exc_info=True)
        try:
            logging.info("Trying to decode with multiply_by_stabiliser=True.")
            container, success = decode_css(
                code=qhgp_code,
                error=error,
                chi_max=chi_max,
                multiply_by_stabiliser=True,
                bias_type=error_model,
                bias_prob=bias_prob,
                renormalise=True,
                silent=silent,
                contraction_strategy="Optimised",
                tolerance=tolerance,
                cut=cut,
            )
        except Exception as ex:
            logging.error(
                f"Decoding has not been completed due to: {ex}", exc_info=True
            )
            container, success = np.nan, np.nan

    if success == 1:
        if not silent:
            logging.info("Decoding successful.")
        return container, 0
    if success == 0:
        if not silent:
            logging.info("Decoding failed.")
        return container, 1
    if not silent:
        logging.info("Decoding has not been completed.")
    return np.nan, np.nan


def run_experiment(
    system_size,
    chi_max,
    error_rate,
    bias_prob,
    num_experiments,
    error_model,
    seed,
    errors,
    silent,
    num_processes=1,
    tolerance=1e-8,
    cut=1e-8,
):
    """Run the experiment consisting of multiple single experiments in parallel."""
    logging.info(
        f"Starting {num_experiments} experiments for SYSTEM_SIZE={system_size},"
        f" CHI_MAX={chi_max}, ERROR_RATE={error_rate}, BIAS_PROB={bias_prob},"
        f" TOLERANCE={tolerance}, CUT={cut}, ERROR_MODEL={error_model}, SEED={seed}"
    )

    check_degree, bit_degree = 4, 3
    num_checks = int(system_size * bit_degree / check_degree)
    if system_size / num_checks != check_degree / bit_degree:
        raise ValueError("The Tanner graph of the code must be bipartite.")
    classical_code = qec.random_regular_code(
        system_size, num_checks, bit_degree, check_degree, qec.Rng(seed)
    )
    qhgp_code = qec.hypergraph_product(classical_code, classical_code)
    code_parameters = (
        len(css_code_stabilisers(qhgp_code)[0][0]),
        len(qhgp_code) - qhgp_code.num_x_stabs() - qhgp_code.num_z_stabs(),
    )

    args = [
        (
            system_size,
            chi_max,
            errors[i],
            bias_prob,
            error_model,
            silent,
            tolerance,
            cut,
            seed,
        )
        for i in range(num_experiments)
    ]

    with Pool(num_processes) as pool:
        results = pool.starmap(run_single_experiment, args)

    logging.info(
        f"Starting {num_experiments} experiments for SYSTEM_SIZE={system_size},"
        f" CHI_MAX={chi_max}, ERROR_RATE={error_rate}, BIAS_PROB={bias_prob},"
        f" TOLERANCE={tolerance}, CUT={cut}, ERROR_MODEL={error_model}, SEED={seed}"
    )

    containers = [result[0] for result in results]
    failures = [result[1] for result in results]

    return {
        "logicals_distributions": containers,
        "failures": failures,
        "errors": errors,
        "lattice_size": system_size,
        "chi_max": chi_max,
        "error_rate": error_rate,
        "bias_prob": bias_prob,
        "error_model": error_model,
        "seed": seed,
        "tolerance": tolerance,
        "cut": cut,
        "code_parameters": code_parameters,
    }


def save_experiment_data(
    data,
    system_size,
    chi_max,
    error_rate,
    error_model,
    bias_prob,
    num_experiments,
    seed,
    tolerance,
    cut,
):
    """Save the experiment data."""
    error_model = error_model.replace(" ", "")
    file_key = f"latticesize{system_size}_bonddim{chi_max}_errorrate{error_rate}_errormodel{error_model}_bias_prob{bias_prob}_numexperiments{num_experiments}_tolerance{tolerance}_cut{cut}_seed{seed}.pkl"
    with open(file_key, "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    logging.info(
        f"Saved data for {file_key} with "
        f"{np.nanmean(data['failures'])*100:.2f}±{sem(data['failures'], nan_policy='omit')*100:.2f}% failure rate."
    )


def main():
    """Main entry point."""
    args = parse_arguments()
    errors = generate_errors(
        args.system_size,
        args.error_rate,
        args.num_experiments,
        args.error_model,
        args.seed,
    )
    experiment_data = run_experiment(
        args.system_size,
        args.bond_dim,
        args.error_rate,
        args.bias_prob,
        args.num_experiments,
        args.error_model,
        args.seed,
        errors,
        args.silent,
        args.num_processes,
        args.tolerance,
        args.cut,
    )
    save_experiment_data(
        experiment_data,
        args.system_size,
        args.bond_dim,
        args.error_rate,
        args.error_model,
        args.bias_prob,
        args.num_experiments,
        args.seed,
        args.tolerance,
        args.cut,
    )


if __name__ == "__main__":
    main()
