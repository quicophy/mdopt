"""This script launches decoding of quantum surface code."""

import os
import sys
import pickle
import logging
import argparse
from functools import partial
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
project_root_fir = os.getenv(
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
examples_path_fir = os.getenv(
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
    from mdopt.decoding.decoding import (
        str_to_bool,
        decode_css,
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
        description="Launch decoding of quantum surface code."
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
        type=str_to_bool,
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
    parser.add_argument(
        "--qubit_order_strategy",
        type=str,
        default="Natural",
        choices=["Natural", "Optimized"],
        help=(
            "Qubit order along the MPS chain: the code's natural order, or the "
            "reverse Cuthill-McKee order, which lowers the bond dimension the "
            "contraction needs (measured: 5x faster on the surface code at "
            "chi=64, and convergence at chi=64 instead of 128 on the [[72,12,6]] "
            "bivariate-bicycle code)."
        ),
    )
    parser.add_argument(
        "--prior_reweighting",
        type=prior_reweighting_choice,
        default=None,
        choices=[None, "BP"],
        help=(
            "None keeps the plain error prior. BP replaces it by the "
            "belief-propagation posterior marginals before the contraction and "
            "undoes the change exactly at marginalization, so the truncation "
            "error is no longer amplified by the inverse of the syndrome mass. "
            "Tags the output file with _priorBP."
        ),
    )
    parser.add_argument(
        "--objective",
        default="amplitude",
        choices=["amplitude", "probability"],
        help=(
            "amplitude (default, historical) ranks logical classes by the sum "
            "of square-rooted error probabilities; probability ranks them by "
            "the class probability, i.e. degenerate maximum likelihood. Tags "
            "the output file with _objProb."
        ),
    )
    parser.add_argument(
        "--store_certificate",
        action="store_true",
        help=(
            "Ask the decoder for its truncation certificate and store the "
            "per-shot survival factor and rigorous relative-error bound in the "
            "pickle (keys 'survival' and 'bound', NaN where the decode did not "
            "complete). A bound above one marks a verdict that may be "
            "truncation residue."
        ),
    )
    return parser.parse_args()


def prior_reweighting_choice(value):
    """argparse type for --prior_reweighting: the string 'None' means no reweighting."""
    return None if value == "None" else value


def certificate_values(outcome):
    """Split a decode_css outcome into (distribution, success, survival, bound).

    With ``return_certificate=True`` the decoder appends a TruncationCertificate
    to its usual pair; without it the survival and bound are reported as NaN so
    callers can treat both shapes alike.
    """
    if len(outcome) == 3:
        distribution, success, certificate = outcome
        return distribution, success, certificate.survival, certificate.bound
    distribution, success = outcome
    return distribution, success, np.nan, np.nan


def generate_errors(lattice_size, error_rate, num_experiments, error_model, seed):
    """Generate errors for the experiments."""
    seed_seq = np.random.SeedSequence(seed)
    errors = []

    rep_code = qec.repetition_code(lattice_size)
    surface_code = qec.hypergraph_product(rep_code, rep_code)

    for _ in range(num_experiments):
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        error = generate_pauli_error_string(
            len(surface_code),
            error_rate,
            rng=rng,
            error_model=error_model,
        )
        errors.append(error)

    return errors


def run_single_experiment(
    lattice_size,
    chi_max,
    error,
    bias_prob,
    error_model,
    silent,
    tolerance,
    cut,
    qubit_order_strategy="Natural",
    seed=None,
    prior_reweighting=None,
    store_certificate=False,
    objective="amplitude",
):
    """Run a single experiment.

    ``qubit_order_strategy="Optimized"`` orders the qubits along the chain by
    reverse Cuthill-McKee before decoding, which cuts the bond dimension the
    contraction needs; the default keeps the campaign's natural order.
    ``seed`` seeds the random stabilizer of the multiply-by-stabilizer
    fallback, so a run that needs it is still reproducible.

    Returns ``(logicals_distribution, failure, survival, bound)``; the last two
    come from the decoder's truncation certificate when ``store_certificate``
    is set and are NaN otherwise (and whenever the decode did not complete).
    """
    rep_code = qec.repetition_code(lattice_size)
    surface_code = qec.hypergraph_product(rep_code, rep_code)

    def _decode(multiply_by_stabilizer, rng=None):
        return certificate_values(
            decode_css(
                code=surface_code,
                error=error,
                chi_max=chi_max,
                multiply_by_stabilizer=multiply_by_stabilizer,
                rng=rng,
                bias_type=error_model,
                bias_prob=bias_prob,
                renormalize=True,
                silent=silent,
                contraction_strategy="Optimized",
                tolerance=tolerance,
                cut=cut,
                qubit_order_strategy=qubit_order_strategy,
                prior_reweighting=prior_reweighting,
                return_certificate=store_certificate,
                objective=objective,
            )
        )

    try:
        logicals_distribution, success, survival, bound = _decode(False)
    except Exception as e:
        logging.error(f"Error during decoding: {e}", exc_info=True)
        try:
            logging.info("Trying to decode with multiply_by_stabilizer=True.")
            logicals_distribution, success, survival, bound = _decode(
                True, rng=np.random.default_rng(seed)
            )
            logging.info("Decoding finished with multiply_by_stabilizer=True.")
        except Exception as ex:
            logging.error(
                f"Decoding has not been completed due to: {ex}", exc_info=True
            )
            logicals_distribution, success = np.nan, np.nan
            survival, bound = np.nan, np.nan

    if success == 1:
        if not silent:
            logging.info("Decoding successful.")
        return logicals_distribution, 0, survival, bound
    if success == 0:
        if not silent:
            logging.info("Decoding failed.")
        return logicals_distribution, 1, survival, bound
    if not silent:
        logging.info("Decoding has not been completed.")
    return np.nan, np.nan, np.nan, np.nan


def run_experiment(
    lattice_size,
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
    qubit_order_strategy="Natural",
    prior_reweighting=None,
    store_certificate=False,
    objective="amplitude",
):
    """Run the experiment consisting of multiple single experiments in parallel."""
    logging.info(
        f"Starting {num_experiments} experiments for LATTICE_SIZE={lattice_size},"
        f" CHI_MAX={chi_max}, ERROR_RATE={error_rate}, BIAS_PROB={bias_prob},"
        f" TOLERANCE={tolerance}, CUT={cut}, ERROR_MODEL={error_model}, SEED={seed},"
        f" PRIOR_REWEIGHTING={prior_reweighting}, STORE_CERTIFICATE={store_certificate},"
        f" OBJECTIVE={objective}"
    )

    worker = partial(
        run_single_experiment,
        qubit_order_strategy=qubit_order_strategy,
        prior_reweighting=prior_reweighting,
        store_certificate=store_certificate,
        objective=objective,
    )

    args = [
        (
            lattice_size,
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

    if num_processes == 1:
        results = [worker(*arg) for arg in args]
    else:
        with Pool(num_processes) as pool:
            results = pool.starmap(worker, args)

    logging.info(
        f"Finished {num_experiments} experiments for LATTICE_SIZE={lattice_size},"
        f" CHI_MAX={chi_max}, ERROR_RATE={error_rate}, BIAS_PROB={bias_prob},"
        f" TOLERANCE={tolerance}, CUT={cut}, ERROR_MODEL={error_model}, SEED={seed}"
    )

    logicals_distributions = [result[0] for result in results]
    failures = [result[1] for result in results]

    data = {
        "logicals_distributions": logicals_distributions,
        "failures": failures,
        "lattice_size": lattice_size,
        "chi_max": chi_max,
        "error_rate": error_rate,
        "bias_prob": bias_prob,
        "error_model": error_model,
        "seed": seed,
        "tolerance": tolerance,
        "cut": cut,
        "qubit_order_strategy": qubit_order_strategy,
        "prior_reweighting": prior_reweighting,
        "objective": objective,
    }
    if store_certificate:
        data["survival"] = np.array([result[2] for result in results], dtype=float)
        data["bound"] = np.array([result[3] for result in results], dtype=float)
    return data


def save_experiment_data(
    data,
    lattice_size,
    chi_max,
    error_rate,
    error_model,
    bias_prob,
    num_experiments,
    seed,
    tolerance,
    cut,
    qubit_order_strategy="Natural",
    prior_reweighting=None,
    objective="amplitude",
):
    """Save the experiment data."""
    error_model = error_model.replace(" ", "")
    # The ordering and the prior change the decoder's results, so both are
    # recorded in the data itself; the filename tags are derived from the
    # stored values. The prior tag keeps reweighted pickles out of the plain
    # series when plotting scripts glob by lattice size, chi and rate.
    qubit_order_strategy = data.get("qubit_order_strategy", qubit_order_strategy)
    order_tag = (
        "" if qubit_order_strategy == "Natural" else f"_order{qubit_order_strategy}"
    )
    prior_reweighting = data.get("prior_reweighting", prior_reweighting)
    prior_tag = f"_prior{prior_reweighting}" if prior_reweighting else ""
    objective = data.get("objective", objective)
    objective_tag = "_objProb" if objective == "probability" else ""
    file_key = f"latticesize{lattice_size}_bonddim{chi_max}_errorrate{error_rate}_errormodel{error_model}_bias_prob{bias_prob}_numexperiments{num_experiments}_tolerance{tolerance}_cut{cut}_seed{seed}{order_tag}{prior_tag}{objective_tag}.pkl"
    with open(file_key, "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    logging.info(
        f"Saved data for {file_key} with "
        f"{np.nanmean(data['failures']) * 100:.2f}±{sem(data['failures'], nan_policy='omit') * 100:.2f}% failure rate."
    )


def main():
    """Main entry point."""
    args = parse_arguments()
    errors = generate_errors(
        args.lattice_size,
        args.error_rate,
        args.num_experiments,
        args.error_model,
        args.seed,
    )
    experiment_data = run_experiment(
        args.lattice_size,
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
        qubit_order_strategy=args.qubit_order_strategy,
        prior_reweighting=args.prior_reweighting,
        objective=args.objective,
        store_certificate=args.store_certificate,
    )
    save_experiment_data(
        experiment_data,
        args.lattice_size,
        args.bond_dim,
        args.error_rate,
        args.error_model,
        args.bias_prob,
        args.num_experiments,
        args.seed,
        args.tolerance,
        args.cut,
        qubit_order_strategy=args.qubit_order_strategy,
        prior_reweighting=args.prior_reweighting,
        objective=args.objective,
    )


if __name__ == "__main__":
    main()
