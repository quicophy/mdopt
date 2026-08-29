Notebook results in the documentation
=====================================

Every notebook under ``examples/`` is rendered on this site with the results it
was committed with. Sphinx does **not** execute them at build time
(``nbsphinx_execute = "never"``), because several take hours and one needs
simulation data that is not in the repository. What you read here is therefore
exactly what the author last ran.

The path a result takes
-----------------------

.. code-block:: text

   examples/<topic>/<notebook>.ipynb        you run it; outputs are saved in the file
             |
             |  git commit + push to main
             v
   docs/source/notebooks/<notebook>.ipynb   docs-sync.yml copies it
             |
             |  readthedocs build
             v
   this website                             nbsphinx renders the stored outputs

Three consequences worth knowing:

* **A notebook committed without outputs shows nothing on the site.** The cells
  appear, the results do not.
* **You never edit** ``docs/source/notebooks/`` **by hand.** It is overwritten
  from ``examples/`` on every push to ``main``.
* ``examples/misc/gpu_example.ipynb`` is excluded from the sync and kept exactly
  as committed. It targets Colab with a GPU and cannot run in CI.

Regenerating the results
------------------------

.. code-block:: bash

   python scripts/run_notebooks.py --inplace                    # all of them
   python scripts/run_notebooks.py --inplace examples/decoding/shor.ipynb

``--inplace`` writes the executed outputs back into the notebook. Commit the
notebook afterwards and the docs follow on the next push to ``main``.

Expect this to take hours. Approximate wall times on a laptop:

============================  ==========
notebook                      full run
============================  ==========
``main_component``            seconds
``quantum_five_qubit``        under a minute
``ground_state``              under a minute
``shor``                      ~3 min
``mps-rand-circ``             ~2 min
``dephasing_dmrg_debug_bb``   ~2 min
``quantum_three_qubit``       ~1.6 h
``maxbonddim``                ~1.2 h
``quantum_surface``           ~5 h
``classical_ldpc``            ~2.5 h
============================  ==========

The shot counts in the four expensive notebooks were chosen to land in that
range. They were once far larger -- ``quantum_surface`` alone needed 33 hours,
which nobody was going to run -- so the results here are deliberately noisier
than a cluster campaign would give. The thesis figures come from the scripts in
``mdopt/examples/decoding/plotting/`` and their cluster datasets, not from these
notebooks.

What CI does instead
--------------------

``notebooks.yml`` executes every notebook on each pull request with
``MDOPT_NB_FAST=1``, which shrinks the workloads to a token size and finishes in
about fourteen minutes. That checks the notebooks still *run* against the current
library -- it does not produce results, and its output is discarded.

This matters: two notebooks were silently broken for ten months by a single
commit, because nothing executed them. ``scripts/run_notebooks.py`` refuses
``--inplace`` when ``MDOPT_NB_FAST=1``, so a CI-scale run can never be published
as a result.
