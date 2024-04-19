"""This script is used to launch calculations on Compute Canada clusters."""

import os
import sys
import logging
import numpy as np
from tqdm import tqdm
import qecstruct as qec
from scipy.stats import unitary_group

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Append paths using environment variables or hardcoded fallbacks
project_root = os.getenv(
    "MDOPT_PATH", "/home/bereza/projects/def-ko1/bereza/project-mdopt/mdopt"
)
examples_path = os.getenv(
    "MDOPT_EXAMPLES_PATH",
    "/home/bereza/projects/def-ko1/bereza/project-mdopt/mdopt/examples",
)

sys.path.append(project_root)
sys.path.append(examples_path)

try:
    from mdopt.mps.utils import create_custom_product_state
    from mdopt.optimiser.utils import (
        SWAP,
        XOR_BULK,
        XOR_LEFT,
        XOR_RIGHT,
    )
    from examples.decoding.decoding import (
        linear_code_constraint_sites,
        linear_code_prepare_message,
    )
    from examples.decoding.decoding import (
        apply_bitflip_bias,
        apply_constraints,
        decode_linear,
    )
except ImportError as e:
    logging.error(
        "Failed to import required modules. Ensure paths are correct.", exc_info=True
    )
    sys.exit(1)

NUM_EXPERIMENTS = 100
CUT = 1e-16
NUM_DMRG_RUNS = 1
CHI_MAX_DMRG = 1e4

SEED = 123
seed_seq = np.random.SeedSequence(SEED)

system_sizes = [24, 48, 96, 192]
max_bond_dims = [1024]
error_rates = np.linspace(0.1, 0.3, 10)
failures_statistics = {}

for NUM_BITS in system_sizes:
    for CHI_MAX_CONTRACTOR in max_bond_dims:
        for PROB_ERROR in tqdm(error_rates):
            logging.info(
                f"Starting experiments for NUM_BITS={NUM_BITS}, CHI_MAX={CHI_MAX_CONTRACTOR}, PROB_ERROR={PROB_ERROR}"
            )
            failures = []

            for l in range(NUM_EXPERIMENTS):
                new_seed = seed_seq.spawn(1)[0]
                rng = np.random.default_rng(new_seed)
                random_integer = rng.integers(1, 10**8 + 1)
                SEED = random_integer

                CHECK_DEGREE, BIT_DEGREE = 4, 3
                NUM_CHECKS = int(BIT_DEGREE * NUM_BITS / CHECK_DEGREE)
                if NUM_BITS / NUM_CHECKS != CHECK_DEGREE / BIT_DEGREE:
                    raise ValueError("The Tanner graph of the code must be bipartite.")
                PROB_BIAS = PROB_ERROR
                CHI_MAX_DMRG = CHI_MAX_CONTRACTOR

                code = qec.random_regular_code(
                    NUM_BITS, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qec.Rng(SEED)
                )
                code_constraint_sites = linear_code_constraint_sites(code)

                INITIAL_CODEWORD, PERTURBED_CODEWORD = linear_code_prepare_message(
                    code, PROB_ERROR, error_model=qec.BinarySymmetricChannel, seed=SEED
                )
                tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]

                initial_codeword_state = create_custom_product_state(
                    INITIAL_CODEWORD, form="Right-canonical"
                )
                perturbed_codeword_state = create_custom_product_state(
                    PERTURBED_CODEWORD, form="Right-canonical"
                )

                perturbed_codeword_state = apply_bitflip_bias(
                    mps=perturbed_codeword_state,
                    sites_to_bias="All",
                    prob_bias_list=PROB_BIAS,
                    renormalise=True,
                )

                try:
                    perturbed_codeword_state = apply_constraints(
                        perturbed_codeword_state,
                        code_constraint_sites,
                        tensors,
                        chi_max=CHI_MAX_CONTRACTOR,
                        renormalise=True,
                        result_to_explicit=False,
                        strategy="Naive",
                        silent=True,
                    )

                    dmrg_container, success = decode_linear(
                        message=perturbed_codeword_state,
                        codeword=initial_codeword_state,
                        code=code,
                        num_runs=NUM_DMRG_RUNS,
                        chi_max_dmrg=CHI_MAX_DMRG,
                        cut=CUT,
                        silent=True,
                    )
                except Exception as e:
                    logging.error(f"Failed in DMRG decoding: {str(e)}", exc_info=True)
                    success = 0

                failures.append(1 - success)

            failures_statistics[(NUM_BITS, CHI_MAX_CONTRACTOR, PROB_ERROR)] = failures
            failures_key = (
                f"numbits{NUM_BITS}_bonddim{CHI_MAX_CONTRACTOR}_errorprob{PROB_ERROR}"
            )
            np.save(f"data/{failures_key}.npy", failures)
            logging.info(
                f"Completed experiments for {failures_key} with {np.mean(failures)*100:.2f}% failure rate."
            )
