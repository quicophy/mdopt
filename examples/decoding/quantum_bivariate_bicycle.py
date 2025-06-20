"""This script launches decoding of quantum bivariate bicycle code."""

import os
import sys
import pickle
import logging
import argparse
from multiprocessing import Pool

import numpy as np
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

sys.path.append(project_root_narval)
sys.path.append(examples_path_narval)

try:
    from examples.decoding.decoding import (
        decode_css,
        create_bb_code,
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
        description="Launch decoding of quantum bivariate bicycle code."
    )
    parser.add_argument(
        "--order_x", type=int, required=True, help="Order along x for BB code."
    )
    parser.add_argument(
        "--order_y", type=int, required=True, help="Order along y for BB code."
    )
    parser.add_argument(
        "--poly_a",
        type=str,
        required=True,
        help="Polynomial A (sympy expr), e.g. '1+x+y'.",
    )
    parser.add_argument(
        "--poly_b",
        type=str,
        required=True,
        help="Polynomial B (sympy expr), e.g. '1+x**2+y**2'.",
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
        "--error_model", type=str, required=True, help="The error model to use."
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
        "--silent", type=bool, required=True, help="Whether to silence the output."
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


def generate_errors(
    order_x,
    order_y,
    poly_a,
    poly_b,
    error_rate,
    num_experiments,
    error_model,
    seed,
):
    """Generate errors for the experiments."""
    seed_seq = np.random.SeedSequence(seed)
    errors = []

    bb_code = create_bb_code(order_x, order_y, poly_a, poly_b)

    for _ in range(num_experiments):
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        error = generate_pauli_error_string(
            len(bb_code),
            error_rate,
            rng=rng,
            error_model=error_model,
        )
        errors.append(error)

    return errors


def run_single_experiment(
    order_x,
    order_y,
    poly_a,
    poly_b,
    chi_max,
    error,
    bias_prob,
    error_model,
    silent,
    tolerance,
    cut,
):
    """Run a single experiment."""
    bb_code = create_bb_code(order_x, order_y, poly_a, poly_b)

    try:
        logicals_distribution, success = decode_css(
            code=bb_code,
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
            logicals_distribution, success = decode_css(
                code=bb_code,
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
            logging.info("Decoding finished with multiply_by_stabiliser=True.")
        except Exception as ex:
            logging.error(
                f"Decoding has not been completed due to: {ex}", exc_info=True
            )
            logicals_distribution, success = np.nan, np.nan

    if success == 1:
        if not silent:
            logging.info("Decoding successful.")
        return logicals_distribution, 0
    if success == 0:
        if not silent:
            logging.info("Decoding failed.")
        return logicals_distribution, 1
    if not silent:
        logging.info("Decoding has not been completed.")
    return np.nan, np.nan


def run_experiment(
    order_x,
    order_y,
    poly_a,
    poly_b,
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
        f"Starting {num_experiments} experiments for ORDERS={order_x, order_y},"
        f" CHI_MAX={chi_max}, ERROR_RATE={error_rate}, BIAS_PROB={bias_prob},"
        f" TOLERANCE={tolerance}, CUT={cut}, ERROR_MODEL={error_model}, SEED={seed}"
    )

    args = [
        (
            order_x,
            order_y,
            poly_a,
            poly_b,
            chi_max,
            errors[i],
            bias_prob,
            error_model,
            silent,
            tolerance,
            cut,
        )
        for i in range(num_experiments)
    ]

    with Pool(num_processes) as pool:
        results = pool.starmap(run_single_experiment, args)

    logging.info(
        f"Starting {num_experiments} experiments for ORDERS={order_x, order_y},"
        f" CHI_MAX={chi_max}, ERROR_RATE={error_rate}, BIAS_PROB={bias_prob},"
        f" TOLERANCE={tolerance}, CUT={cut}, ERROR_MODEL={error_model}, SEED={seed}"
    )

    logicals_distributions = [result[0] for result in results]
    failures = [result[1] for result in results]

    return {
        "logicals_distributions": logicals_distributions,
        "failures": failures,
        "errors": errors,
        "lattice_size": order_x,
        "order_x": order_x,
        "order_y": order_y,
        "chi_max": chi_max,
        "error_rate": error_rate,
        "bias_prob": bias_prob,
        "error_model": error_model,
        "seed": seed,
        "tolerance": tolerance,
        "cut": cut,
        "polynomials": [poly_a, poly_b],
    }


def save_experiment_data(
    data,
    order_x,
    order_y,
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
    file_key = f"latticesize{order_x*order_y}_bonddim{chi_max}_errorrate{error_rate}_errormodel{error_model}_bias_prob{bias_prob}_numexperiments{num_experiments}_tolerance{tolerance}_cut{cut}_seed{seed}.pkl"
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
        args.order_x,
        args.order_y,
        args.poly_a,
        args.poly_b,
        args.error_rate,
        args.num_experiments,
        args.error_model,
        args.seed,
    )
    experiment_data = run_experiment(
        args.order_x,
        args.order_y,
        args.poly_a,
        args.poly_b,
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
        args.order_x,
        args.order_y,
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
