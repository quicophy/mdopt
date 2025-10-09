Getting started
===============

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install mdopt

Or from source using Poetry:

.. code-block:: bash

   git clone https://github.com/quicophy/mdopt.git
   cd mdopt
   poetry install

Minimal example
---------------------

Run this quick check to verify your setup:

.. code-block:: python

    import numpy as np
    import qecstruct as qec
    from examples.decoding.decoding import decode_css

    # Define a small instance of the surface code
    LATTICE_SIZE = 3
    surface_code = qec.hypergraph_product(
        qec.repetition_code(LATTICE_SIZE),
        qec.repetition_code(LATTICE_SIZE),
    )

    # Input an error and choose decoder controls
    logicals, success = decode_css(
        code=surface_code,
        error="IIXIIIIIIIIII",
        bias_prob=0.01,
        bias_type="Bitflip",
        chi_max=64,
        renormalise=True,
        contraction_strategy="Optimised",
        tolerance=1e-12,
        silent=False,
    )

Workflow at a glance
--------------------

1. **Formulate** your optimisation problem in the TN language (MPS state, MPO constraints/operators).
2. **Apply** constraints/operations with a chosen contraction strategy and bond-dimension limit.
3. **Optimise** (e.g., DMRG-like decoding) and **evaluate** success metrics.

Platforms
---------

Tested on macOS and Linux (Compute Canada clusters). Windows is currently not supported.
