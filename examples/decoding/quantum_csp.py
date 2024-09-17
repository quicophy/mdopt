"""This script launches quantum decoding on Compute Canada clusters."""

import os
import sys
import json
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
        description="Launch quantum CSP code decoding on Compute Canada clusters."
    )
    parser.add_argument(
        "--num_qubits",
        type=int,
        nargs="+",
        required=True,
        help="The number of qubits in the code.",
    )
    parser.add_argument(
        "--bond_dim",
        type=int,
        required=True,
        help="Maximum bond dimension to keep during contraction.",
    )
    parser.add_argument(
        "--error_rate",
        type=float,
        nargs="+",
        required=True,
        help="The error rate.",
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
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="The batch to process.",
    )
    return parser.parse_args()


def run_experiment(
    batch, num_qubits, code_file, bond_dim, error_rate, num_experiments, seed
):
    logging.info(
        f"Starting experiments for Batch={batch}, Qubits={num_qubits}, Code={code_file}, BOND_DIM={bond_dim}, ERROR_RATE={error_rate}, SEED={seed}"
    )

    with open(code_file, "r") as code_json:
        code_data = json.load(code_json)
        x_code = qec.LinearCode(
            qec.BinaryMatrix(
                num_columns=code_data["num_qubits"], rows=code_data["x_stabs"]
            )
        )
        z_code = qec.LinearCode(
            qec.BinaryMatrix(
                num_columns=code_data["num_qubits"], rows=code_data["z_stabs"]
            )
        )
        quantum_csp_code = qec.CssCode(x_code=x_code, z_code=z_code)

    seed_seq = np.random.SeedSequence(seed)
    failures = []

    for l in tqdm(range(num_experiments)):
        new_seed = seed_seq.spawn(1)[0]
        rng = np.random.default_rng(new_seed)
        random_integer = rng.integers(1, 10**8 + 1)
        experiment_seed = random_integer

        try:
            error = generate_pauli_error_string(
                len(quantum_csp_code),
                error_rate,
                seed=experiment_seed,
                error_model="Depolarising",
            )
            error = pauli_to_mps(error)

            _, success = decode_css(
                code=quantum_csp_code,
                error=error,
                chi_max=bond_dim,
                bias_type="Depolarising",
                bias_prob=error_rate,
                renormalise=True,
                silent=True,
                contraction_strategy="Optimised",
            )

            failures.append(1 - success)
        except Exception as e:
            logging.error(f"Experiment {l} failed with error: {str(e)}", exc_info=True)
            failures.append(1)

    # Store results in a structured filename
    result_filename = (
        f"numqubits{num_qubits}_bonddim{bond_dim}_errorrate{error_rate:.12f}_"
        f"seed{seed}_batch{batch}_{os.path.splitext(os.path.basename(code_file))[0]}.npy"
    )
    np.save(result_filename, np.array(failures))

    logging.info(
        f"Completed experiments for {result_filename} with {np.mean(failures)*100:.2f}% failure rate."
    )


def main():
    args = parse_arguments()

    for num_qubits in args.num_qubits:
        code_path = f"/home/bereza/scratch/data-csp-codes/batch_{args.batch}/codes/qubits_{num_qubits}"

        for code in os.listdir(code_path):
            if code.endswith(".json"):
                code_file = os.path.join(code_path, code)

                for error_rate in args.error_rate:
                    run_experiment(
                        batch=args.batch,
                        num_qubits=num_qubits,
                        code_file=code_file,
                        bond_dim=args.bond_dim,
                        error_rate=error_rate,
                        num_experiments=args.num_experiments,
                        seed=args.seed,
                    )


if __name__ == "__main__":
    main()
