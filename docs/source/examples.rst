Examples
========


.. toctree::
   :maxdepth: 1
   :caption: Decoding

   classical_ldpc.ipynb
   quantum_three_qubit.ipynb
   quantum_five_qubit.ipynb
   shor.ipynb
   quantum_surface.ipynb

- :doc:`classical_ldpc` — Builds an MPS that encodes the superposition of all codewords of a random-regular Gallager (3,4) classical LDPC code and translates parity checks into MPO (XOR) constraints. Demonstrates “dephasing DMRG” to solve the main-component (max-likelihood) decoding problem and validates the end-to-end pipeline.

- :doc:`quantum_three_qubit` — Minimal quantum decoding demo using the 3-qubit repetition code under bit-flip noise. Illustrates the tensor network site layout and shows agreement between the analytical logical failure rate and the numerical results.

- :doc:`quantum_five_qubit` — Decode the [[5,1,3]] five-qubit “perfect code” using our MPS decoder under bit-flip noise. Dives into the details of the error passing through the decoder.

- :doc:`shor` — Demonstrates the 9-qubit Shor code with separated X/Z error handling in the TN framework. Serves as a bridge between toy codes and LDPC-like examples.

- :doc:`quantum_surface` — Small planar surface-code instances (e.g., perfect syndrome). Compare contraction strategies and explore accuracy/cost trade-offs as a function of maximum bond dimension.


.. toctree::
   :maxdepth: 1
   :caption: Miscellaneous

   ground_state.ipynb
   mps-rand-circ.ipynb
   maxbonddim.ipynb
   main_component.ipynb

- :doc:`ground_state` — Solve a simple 1D quantum Ising chain using an MPS ground-state search. Compares observables and magnetisation curves from exact diagonalisation and DMRG to confirm correctness. A gentle introduction to MPS/MPO mechanics outside of decoding.

- :doc:`mps-rand-circ` — Simulates random (checkerboard-style) circuits on MPS to study entanglement growth and contraction behaviour. Tracks fidelity decay at fixed bond dimension, providing a benchmarking harness for contraction strategies.

- :doc:`main_component` — Defines and solves the Main Component Problem (finding the basis state that contributes most to a given state) as a sanity check for the dephasing DMRG optimiser. Compares solutions from exact diagonalisation, standard DMRG, and dephasing DMRG, demonstrating agreement.

- :doc:`maxbonddim` — Tests an MPO order-optimisation strategy (based on matrix bandwidth minimisation) to reduce intermediate bond growth during operator application. Shows how reordering lowers contraction cost and improves practical performance within experiments.
