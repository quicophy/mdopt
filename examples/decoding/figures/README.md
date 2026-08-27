# Figures

Each figure below maps to the script that regenerates it. **The scripts now live
in the package** at `mdopt/examples/decoding/plotting/`; the data they read and
the figures they write stay here, under the repo-level `examples/decoding/`.

**The figures themselves are not tracked in git.** They are reproducible from the
scripts below and would otherwise add several megabytes of binaries to the
repository, so `.gitignore` keeps this directory local and tracks only this
manifest. The two hand-made decoder illustrations cannot be rebuilt and stay
tracked one directory up, in `examples/decoding/`.

The scripts are cwd-independent — they resolve these assets through
`mdopt/examples/paths.py`. Run one from anywhere:

```bash
python -m mdopt.examples.decoding.plotting.gen_surface_plots
```

Set `MDOPT_EXAMPLES_ASSETS` to the top-level `examples/` directory containing
`decoding/` when the assets are not inside a checkout (see `mdopt/examples/paths.py`).

FAST = replots from a stored pickle. SIM = re-runs the simulation.

Every dataset lives under `decoding/data/`: the per-experiment runs in
`data/<code-family>/` and the small replot caches in `data/cache/`.

| Figure | Script | Cost | Cache / input |
|---|---|---|---|
| `bb-tn-failure-rate.pdf` | `gen_bb_plot.py` | FAST | `data/quantum-bivariate-bicycle/` |
| `csp-tn-failure-rate.pdf` | `gen_csp_plots.py` | FAST | `data/quantum-csp-batch-9/` |
| `csp-bp-failure-rate.pdf` | `gen_csp_comparison.py` | FAST | `data/cache/bp_results.pkl` |
| `csp-chi-convergence.pdf`, `csp-ler-ratio.pdf`, `csp-ler-vs-chi.pdf` | `gen_chi_min_plot.py` | FAST | `data/cache/bp_results.pkl` |
| `surface-failure-rate.pdf` | `gen_surface_plots.py` | FAST | `data/quantum-surface/` |
| `surface-vert-horiz-speedup.pdf` | `aggregate_vert_horiz_surface.py` | FAST | `data/cache/vert_horiz_surface_L5_data.pkl` |
| `surface-vert-horiz-ablation.pdf`, `qubit-ordering-comparison.pdf`, `qubit-ordering-speedup.pdf` | `aggregate_vert_horiz_surface_variants.py` | FAST | `data/cache/vert_horiz_surface_L5*_data.pkl` |
| `bb-vert-horiz-speedup.pdf` | `aggregate_vert_horiz_bb.py` | FAST | writes `data/cache/vert_horiz_bb_data.pkl` |
| `bb-qubit-ordering.pdf` | `aggregate_qubit_order_bb.py` | FAST | `data/cache/qubit_order_bb_data.pkl` |
| `5qubit-erasure-failure-rate.pdf` | `plot_erasure_from_pickle.py` | FAST | `data/cache/failure_rate_5qubit_erasure_data.pkl` |
| `5qubit-erasure-failure-rate.pdf` | `gen_failure_rate_5qubit_erasure.py` | SIM | writes the `.pkl` above |
| `3qubit-failure-rate.pdf` | `gen_failure_rate_3qubit.py` | SIM | `data/cache/failure_rate_3qubit_data.pkl` |
| `3qubit-heatmaps.pdf`, `3qubit-logical-probs.pdf`, `3qubit-truncation-error.pdf`, `3qubit-failure-rate.pdf` | `gen_plots_3qubit.py` | SIM | — |
| `5qubit-heatmaps.pdf`, `5qubit-logical-probs.pdf`, `5qubit-truncation-error.pdf`, `5qubit-correction-histograms.pdf` | `gen_plots_5qubit.py` | SIM | — |
| `5qubit-correction-histograms.pdf` | `gen_hist_5qubit.py` | SIM | — |
| `shor-correction-histograms.pdf` | `gen_hist_shor.py` | SIM | — |
| `bb-qubit-ordering.pdf` | `gen_qubit_order_bb.py` | SIM | writes `data/cache/qubit_order_bb_data.pkl` |
| `ldpc-heatmaps-n16.pdf` | `gen_classical_heatmaps.py` | SIM | writes into `data/classical-ldpc/` |
| `ldpc-bdim-scaling-n24.pdf` | `gen_classical_bond_dim.py` | SIM | `data/classical-ldpc/` |
| `ldpc-mps-vs-bp-n20.pdf` | `gen_classical_mps_n20.py` | SIM | `data/classical-ldpc/` |

## Not produced by a plotting script

These exist in this folder but no script in `mdopt/examples/decoding/plotting/`
rebuilds them, so
deleting one is not recoverable the way the rest are.

| File | Source |
|---|---|
| `csp-bp-osd-average.pdf` | `python -m mdopt.examples.decoding.quantum_csp_bp` (BP-OSD baseline run) |
| `threshold_estimate.pdf` | `pdflatex ../threshold_estimate.tex` — keeps the snake_case name because pdflatex names its output after the source |
| `csp-chi-min.pdf` | no known producer — keep, cannot rebuild |
| `surface-bdim-failure-rate.pdf` | no known producer — keep, cannot rebuild |
| `surface-failure-rate.png` | no known producer (the PDF version is rebuildable) |
| `../5qubit-decoder.png`, `../shor-decoder.png` | hand-made illustrations — cannot rebuild, tracked in git |

## Naming

Figures are named `<subject>-<content>.pdf` in kebab-case: subject first, so
everything about one code sorts together. The names below that the thesis uses
are reproduced **exactly**, so a rebuilt figure can be copied into
`~/phd-thesis-udes/figures/` with no rename. The rest follow the same convention.

Pinned by the thesis:

| figure |
|---|
| `3qubit-failure-rate.pdf` |
| `3qubit-logical-probs.pdf` |
| `3qubit-truncation-error.pdf` |
| `5qubit-correction-histograms.pdf` |
| `5qubit-erasure-failure-rate.pdf` |
| `5qubit-heatmaps.pdf` |
| `5qubit-logical-probs.pdf` |
| `bb-tn-failure-rate.pdf` |
| `csp-bp-failure-rate.pdf` |
| `csp-tn-failure-rate.pdf` |
| `qubit-ordering-comparison.pdf` |
| `qubit-ordering-speedup.pdf` |
| `shor-correction-histograms.pdf` |
| `surface-bdim-scaling.pdf` |
| `surface-failure-rate.pdf` |

The thesis `tn-*.pdf`, `rep-code-tanner.pdf` and `shor-code-tanner.pdf` are TikZ
diagrams drawn in the thesis repo, not produced here.
