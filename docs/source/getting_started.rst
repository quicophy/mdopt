Getting started
===============

Installation
------------

You can install the package from PyPI:

.. code-block:: bash

   pip install mdopt

Or, alternatively, from source using Poetry:

.. code-block:: bash

   git clone https://github.com/quicophy/mdopt.git
   cd mdopt
   poetry install

Minimal example
---------------------

Run this quick decoding example to verify your setup:

.. code-block:: python

    import logging

    import numpy as np
    import qecstruct as qec
    from mdopt.decoding import decode_css

    # The library does not configure logging; opt in to see progress with silent=False.
    logging.basicConfig(level=logging.INFO)

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

Decoding a detector error model
-------------------------------

Circuit-level noise enters through a `stim <https://github.com/quantumlib/Stim>`__
detector error model (DEM). Every error mechanism becomes one MPS site, each
detector a parity-check (XOR) constraint and each logical observable a readout,
so the same MPS-MPO machinery returns the maximum-likelihood observable flip for
a sampled syndrome:

.. code-block:: python

    import stim
    from mdopt.decoding import decode_dem, dem_to_problem

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.008,
        before_measure_flip_probability=0.008,
        after_reset_flip_probability=0.008,
    )
    # Maximum likelihood wants the undecomposed hyperedges, so keep decompose_errors off.
    problem = dem_to_problem(
        circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    )
    sampler = circuit.compile_detector_sampler(seed=1)
    detections, observables = sampler.sample(1, separate_observables=True)
    class_masses, predicted_flips = decode_dem(
        problem, detections[0].astype(int), chi_max=32
    )
    print(predicted_flips, observables[0].astype(int))

:func:`~mdopt.decoding.dem.decode_dem` returns the probability mass of every
observable-flip class together with the most likely one. A class mass that turns
materially negative, or a vector that collapses to zero, raises an
``ArithmeticError``: the contraction is not converged at that ``chi_max`` and the
shot should be decoded again at a larger bond dimension. The harnesses behind the
decoder's validation campaigns live in ``examples/decoding/dem_campaign`` in the
repository, with a README on how every number is regenerated.

Workflow at a glance
--------------------

1. Formulate your optimisation problem in the MPS-MPO formalism (MPS state, MPO constraints/operators).
2. Apply constraints/operations with a chosen contraction and truncation strategies.
3. Optimise (e.g., DMRG-like decoding) and evaluate success metrics.

For decoding, steps 1-3 are wrapped by :func:`~mdopt.decoding.decoding.decode_css`
(CSS codes from ``qecstruct``), :func:`~mdopt.decoding.decoding.decode_custom`
(any stabiliser code given as Pauli strings),
:func:`~mdopt.decoding.decoding.decode_message` (classical linear codes) and
:func:`~mdopt.decoding.dem.decode_dem` (detector error models).

Platforms
---------

The package has been tested on macOS and Linux (within Compute Canada clusters). Windows is currently not supported.
