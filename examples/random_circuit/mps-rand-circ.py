"""This script is used to launch calculations on Compute Canada clusters."""

import os
import sys
import logging
import numpy as np
from scipy.stats import unitary_group

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Append paths using environment variables or hardcoded fallbacks
project_root = os.getenv(
    'MDOPT_PATH', '/home/bereza/projects/def-ko1/bereza/project-mdopt/mdopt'
    )
examples_path = os.getenv(
    'MDOPT_EXAMPLES_PATH', '/home/bereza/projects/def-ko1/bereza/project-mdopt/mdopt/examples'
    )

sys.path.append(project_root)
sys.path.append(examples_path)

try:
    from mdopt.mps.utils import create_simple_product_state
    from mdopt.contractor.contractor import mps_mpo_contract
    from examples.random_circuit.random_circuit import create_mpo_from_layer
except ImportError as e:
    logging.error("Failed to import required modules. Ensure paths are correct.", exc_info=True)
    sys.exit(1)

PHYS_DIM = 2
circ_depths = [4, 6, 8, 10, 12, 14, 16, 18, 20]
bond_dims = [8, 10, 12, 14, 16, 18, 20, 22, 24]
num_qubits = [27, 81, 243]

tails = {}
for NUM_QUBITS in num_qubits:
    for BOND_DIM in bond_dims:
        for NUM_LAYERS_CIRC in circ_depths:
            tails_iter = []
            state = create_simple_product_state(
                num_sites=NUM_QUBITS,
                phys_dim=PHYS_DIM,
                which="0",
                form="Right-canonical",
                tolerance=np.inf,
            )

            for k in range(NUM_LAYERS_CIRC):
                layer = unitary_group.rvs(PHYS_DIM**2, size=NUM_QUBITS // PHYS_DIM - k % 2)
                mpo = create_mpo_from_layer(layer)
                state = mps_mpo_contract(
                    mps=state,
                    mpo=mpo,
                    start_site=k % 2,
                    renormalise=True,
                    chi_max=1e4,
                    cut=1e-12,
                    inplace=False,
                    result_to_explicit=False,
                )

                state, errors = state.compress(
                    chi_max=BOND_DIM,
                    cut=1e-12,
                    renormalise=True,
                    strategy="svd",
                    return_truncation_errors=True,
                )

                fidels = [1 - error for error in errors]
                tails_iter.append(np.prod(fidels))

            tails_key = f"data/numqubits{NUM_QUBITS}_bonddim{BOND_DIM}_circlayers{NUM_LAYERS_CIRC}"
            tails[tails_key] = tails_iter

            np.save(f"{tails_key}.npy", tails)

logging.info("Calculation completed successfully.")
