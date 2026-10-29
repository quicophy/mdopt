Project overview & references
=============================

Motivation
----------

mdopt expresses discrete optimization tasks in the tensor-network (MPS/MPO) formalism.
This enables controlled approximations (e.g., bond-dimension truncation) and leverages
well-studied contraction/optimization techniques for problems such as decoding quantum
error-correcting codes. The decoders cover code-capacity noise on stabilizer codes
(:mod:`mdopt.decoding.decoding`) and circuit-level noise through stim detector error
models (:mod:`mdopt.decoding.dem`), with one MPS site per error mechanism.

Intended users
--------------

- Physicists and quantum-information researchers
- Practitioners exploring tensor-network methods for discrete optimization

References & related resources
------------------------------

- Project repository: https://github.com/quicophy/mdopt
- Documentation: https://mdopt.readthedocs.io/en/latest/
- Citation details: please see the README
