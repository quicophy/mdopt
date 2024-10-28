# `mdopt` — Discrete Optimisation in the MPS-MPO Language

[![codecov](https://codecov.io/gh/quicophy/mdopt/branch/main/graph/badge.svg?token=4G7VWYX0S2)](https://codecov.io/gh/quicophy/mdopt)
[![tests](https://github.com/quicophy/mdopt/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/quicophy/mdopt/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/mdopt/badge/?version=latest)](https://mdopt.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/quicophy/mdopt/main.svg)](https://results.pre-commit.ci/latest/github/quicophy/mdopt/main)
[![lint](https://github.com/quicophy/mdopt/actions/workflows/lint.yml/badge.svg)](https://github.com/quicophy/mdopt/actions/workflows/lint.yml)
[![mypy](https://github.com/quicophy/mdopt/actions/workflows/mypy.yml/badge.svg?branch=main)](https://github.com/quicophy/mdopt/actions/workflows/mypy.yml)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-Unitary%20Fund-brightgreen.svg?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAACgAAAASCAYAAAApH5ymAAAAt0lEQVRIic2WUQ6AIAiGsXmC7n9Gr1Dzwcb%2BUAjN8b%2B0BNwXApbKRRcF1nGmN5y0Jon7WWO%2B6pgJLhtynzUHKTMNrNo4ZPPldikW10f7qYBEMoTmJ73z2NFHcJkAvbLUpVYmvwIigKeRsjdQEtagZ2%2F0DzsHG2h9iICrRwh2qObbGPIfMDPCMjHNQawpbc71bBZhsrpNYs3qqCFmO%2FgBjHTEqKm7eIdMg9p7PCvma%2Fz%2FwQAMfRHRDTlhQGoOLve1AAAAAElFTkSuQmCC)](http://unitary.fund)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)


### `mdopt` is a python package built on top of numpy for discrete optimisation in the tensor-network (specifically, MPS-MPO) language.

## Installation

To install the current release, use the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install mdopt
```

Otherwise, clone the repository and use [poetry](https://python-poetry.org/).

```bash
poetry install
```

## `mdopt` at a glance

```python
import logging
import numpy as np
import qecstruct as qec
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
    decode_message,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

NUM_BITS = 24
CHI_MAX = 256
NUM_EXPERIMENTS = 10

SEED = 123
seed_seq = np.random.SeedSequence(SEED)

error_rates = np.linspace(0.1, 0.3, 10)
failures_statistics = {}

for ERROR_RATE in error_rates:
    logging.info(
        f"Starting experiments for NUM_BITS={NUM_BITS}, CHI_MAX={CHI_MAX}, ERROR_RATE={ERROR_RATE}"
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
        PROB_BIAS = ERROR_RATE

        code = qec.random_regular_code(
            NUM_BITS, NUM_CHECKS, BIT_DEGREE, CHECK_DEGREE, qec.Rng(SEED)
        )
        code_constraint_sites = linear_code_constraint_sites(code)

        INITIAL_CODEWORD, PERTURBED_CODEWORD = linear_code_prepare_message(
            code, ERROR_RATE, error_model=qec.BinarySymmetricChannel, seed=SEED
        )
        tensors = [XOR_LEFT, XOR_BULK, SWAP, XOR_RIGHT]

        initial_codeword_state = create_custom_product_state(
            INITIAL_CODEWORD, form="Right-canonical"
        )
        perturbed_codeword_state = create_custom_product_state(
            PERTURBED_CODEWORD, form="Right-canonical"
        )

        logging.info("Applying bitflip bias to the perturbed codeword state.")
        perturbed_codeword_state = apply_bitflip_bias(
            mps=perturbed_codeword_state,
            sites_to_bias="All",
            prob_bias_list=PROB_BIAS,
            renormalise=True,
        )

        try:
            logging.info("Applying constraints to the perturbed codeword state.")
            perturbed_codeword_state = apply_constraints(
                perturbed_codeword_state,
                code_constraint_sites,
                tensors,
                chi_max=CHI_MAX,
                renormalise=True,
                result_to_explicit=False,
                strategy="Optimised",
                silent=False,
            )
            logging.info("Decoding the perturbed codeword state using DMRG.")
            dmrg_container, success = decode_message(
                message=perturbed_codeword_state,
                codeword=initial_codeword_state,
                chi_max_dmrg=CHI_MAX,
            )
            if success == 1:
                logging.info("Decoding successful.")
            else:
                logging.info("Decoding failed.")
        except Exception as e:
            logging.error(f"Failed in DMRG decoding: {str(e)}", exc_info=True)
            success = 0

        failures.append(1 - success)
        logging.info(
            f"Finished experiment {l} for NUM_BITS={NUM_BITS}, CHI_MAX={CHI_MAX}, ERROR_RATE={ERROR_RATE}"
        )

    failures_statistics[(NUM_BITS, CHI_MAX, ERROR_RATE)] = failures
    failures_key = (
        f"numbits{NUM_BITS}_bonddim{CHI_MAX}_errorrate{ERROR_RATE}"
    )
    logging.info(
        f"Completed experiments for {failures_key} with {np.mean(failures)*100:.2f}% failure rate."
    )
```

For more examples, see the `mdopt`
[examples folder](https://github.com/quicophy/mdopt/tree/main/examples).

## Cite
If you happen to find `mdopt` useful in your research, please consider supporting development by citing it.
```
@software{mdopt2022,
  author = {Aleksandr Berezutskii},
  title = {mdopt: Discrete optimisation in the tensor-network (specifically, MPS-MPO) language.},
  url = {https://github.com/quicophy/mdopt},
  year = {2022},
}
```

## Contribution guidelines

If you want to contribute to `mdopt`, be sure to follow GitHub's contribution guidelines.
This project adheres to our [code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to
uphold this code.

We use [GitHub issues](https://github.com/quicophy/mdopt/issues) for
tracking requests and bugs, please direct specific questions to the maintainers.

The `mdopt` project strives to abide by generally accepted best practices in
open-source software development, such as:

*   apply the desired changes and resolve any code
    conflicts,
*   run the tests and ensure they pass,
*   build the package from source.
