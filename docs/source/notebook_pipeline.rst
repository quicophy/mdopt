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
   docs/source/notebooks/<notebook>.ipynb   docs-sync.yml copies it and opens
             |                               a small sync PR (main is protected,
             |  sync PR merged               so it cannot push directly); until
             |                               that PR merges, the site still
             |  readthedocs build            serves the previous outputs
             v
   this website                             nbsphinx renders the stored outputs

Three consequences worth knowing:

* **A notebook committed without outputs shows nothing on the site.** The cells
  appear, the results do not.
* **You never edit** ``docs/source/notebooks/`` **by hand.** Every relevant
  push to ``main`` opens (or updates) a small sync PR that overwrites it from
  ``examples/``; the copies change when that PR is merged.
* ``examples/misc/gpu_example.ipynb`` is excluded from the sync and kept exactly
  as committed. It targets Colab with a GPU and cannot run in CI.

Regenerating the results
------------------------

.. code-block:: bash

   python scripts/run_notebooks.py --inplace                    # all of them
   python scripts/run_notebooks.py --inplace examples/decoding/shor.ipynb

``--inplace`` writes the executed outputs back into the notebook. Commit the
notebook afterwards; the next push to ``main`` opens the sync PR, and the docs
follow once it is merged.

Expect this to take hours. Measured on an M-series laptop, one notebook at a
time (``classical_ldpc`` is the one estimate here -- it has not yet completed
a full run):

===========================  =================
notebook                     full run
===========================  =================
``main_component``           4 s
``ground_state``             11 s
``mps-rand-circ``            26 s
``dephasing_dmrg_debug_bb``  39 s
``shor``                     39 s
``quantum_five_qubit``       43 s
``maxbonddim``               1.2 h
``quantum_three_qubit``      1.6 h
``quantum_surface``          4.9 h
``classical_ldpc``           ~2.5 h (estimate)
===========================  =================

The shot counts in the four expensive notebooks were chosen to land in that
range. They were once far larger -- ``quantum_surface`` alone needed 33 hours,
which nobody was going to run -- so the results here are deliberately noisier
than a cluster campaign would give. The thesis figures come from the scripts in
``mdopt/examples/decoding/plotting/`` and their cluster datasets, not from these
notebooks.

The sync PR
-----------

The copy from ``examples/`` to ``docs/source/notebooks`` is delivered by a bot
pull request from the rolling ``docs-sync-bot`` branch, because ``main`` is
protected and rejects direct pushes. Its full contract:

* **It opens (or updates) automatically** when a push to ``main`` touches
  ``examples/**/*.ipynb``, ``examples/**/*.png``, ``docs/source/notebooks/**``,
  ``generate_docs.sh`` or the workflow itself -- *and* the copy step finds real
  drift. No drift, no PR.
* **There is only ever one.** Consecutive drifts force-push the same branch,
  updating the open PR in place.
* **It cleans up after itself.** If the drift disappears (a notebook reverted,
  or synchronised another way), the next run closes the stale PR and deletes
  the branch, so outdated copies cannot be merged.
* **Merging it does not loop.** The merge triggers a run that finds no drift
  and exits without committing.
* **Its checks need a nudge.** GitHub does not start CI for pull requests
  opened with the default workflow token, so close and reopen the PR (or push
  any commit to it) before a checks-gated merge. The PR body says so too.
* **Manual dispatch:** ``gh workflow run docs-sync.yml --ref main`` runs the
  sync on demand.

One trap to know: GitHub honours CI-skip markers (such as ``[skip ci]``)
anywhere in a commit message, and a squash merge concatenates every branch
commit's title and body into one message. A commit that merely *quotes* such a
marker therefore silences all workflows on the merge push -- including this
sync. If a merge to ``main`` starts no workflows, check the squash message
first, then dispatch the sync manually.

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
