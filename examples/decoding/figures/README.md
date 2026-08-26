# Figures

Every PDF here is a build artifact: it can be regenerated from a script in
`../plotting/`. Nothing in this folder is hand-edited, so it is always safe to
delete a figure and rebuild it.

**The figures themselves are not tracked in git.** They are reproducible from the
scripts below and would otherwise add several megabytes of binaries to the
repository, so `.gitignore` keeps this directory local and tracks only this
manifest. The two hand-made decoder illustrations cannot be rebuilt and stay
tracked one directory up, in `examples/decoding/`.

All plotting scripts are **cwd-independent** — they anchor their paths on
`__file__`, `chdir` to `examples/decoding/`, and write figures here via an
absolute path. Run them from anywhere:

```bash
poetry run python examples/decoding/plotting/<script>.py
```

## Rebuild cost

- **FAST** — reads a cached `.pkl` or a `data-*/` directory and only draws.
  Seconds to a few minutes.
- **SIM** — reruns the decoder before plotting. Minutes to many hours.
  Where a `.pkl` cache is listed, the script reuses it if present, so prefer
  the FAST re-plot script when one exists.

| Figure | Script | Cost | Cache / input |
|---|---|---|---|
| `bb_mps_mpo.pdf` | `gen_bb_plot.py` | FAST | `data-quantum-bivariate-bicycle/` |
| `csp_mps_mpo.pdf` | `gen_csp_plots.py` | FAST | `data-quantum-csp-batch-9/` |
| `csp_comparison.pdf` | `gen_csp_comparison.py` | FAST | `bp_results.pkl` |
| `csp_chi_conv.pdf`, `csp_ler_ratio.pdf`, `csp_ler_vs_chi.pdf` | `gen_chi_min_plot.py` | FAST | `bp_results.pkl` |
| `failure_rate_surface_lattice_size.pdf` | `gen_surface_plots.py` | FAST | `data-quantum-surface/` |
| `vert_horiz_surface.pdf` | `aggregate_vert_horiz_surface.py` | FAST | `vert_horiz_surface_L5_data.pkl` |
| `figs_vert_horiz_surface_v1/v2/v3.pdf` | `aggregate_vert_horiz_surface_variants.py` | FAST | `vert_horiz_surface_L5*_data.pkl` |
| `vert_horiz_bb.pdf` | `aggregate_vert_horiz_bb.py` | FAST | writes `vert_horiz_bb_data.pkl` |
| `qubit_order_bb.pdf` | `aggregate_qubit_order_bb.py` | FAST | `qubit_order_bb_data.pkl` |
| `failure_rate_5qubit_erasure.pdf` | `plot_erasure_from_pickle.py` | FAST | `failure_rate_5qubit_erasure_data.pkl` |
| `failure_rate_5qubit_erasure.pdf` | `gen_failure_rate_5qubit_erasure.py` | SIM | writes the `.pkl` above |
| `failure_rate_3qubit.pdf` | `gen_failure_rate_3qubit.py` | SIM | `failure_rate_3qubit_data.pkl` |
| `heatmaps_3qubit.pdf`, `logical_probs_3qubit.pdf`, `truncation_error_3qubit.pdf`, `failure_rate_3qubit.pdf` | `gen_plots_3qubit.py` | SIM | — |
| `heatmaps_5qubit.pdf`, `logical_probs_5qubit.pdf`, `truncation_error_5qubit.pdf`, `correction_histograms_5qubit.pdf` | `gen_plots_5qubit.py` | SIM | — |
| `correction_histograms_5qubit.pdf` | `gen_hist_5qubit.py` | SIM | — |
| `correction_histograms_shor.pdf` | `gen_hist_shor.py` | SIM | — |
| `qubit_order_bb.pdf` | `gen_qubit_order_bb.py` | SIM | writes `qubit_order_bb_data.pkl` |
| `heatmaps_n16.pdf` | `gen_classical_heatmaps.py` | SIM | writes into `data-classical-ldpc/` |
| `bond_dim_scaling_n24.pdf` | `gen_classical_bond_dim.py` | SIM | `data-classical-ldpc/` |
| `mps_vs_bp_n20.pdf` | `gen_classical_mps_n20.py` | SIM | `data-classical-ldpc/` |

## Thesis figure names

The thesis (`~/phd-thesis-udes/figures/`) uses different filenames. Copying a
rebuilt figure into the thesis requires this rename. Rows marked ✓ were
confirmed by content hash; the rest are matched by name and content.

| this folder | thesis `figures/` | |
|---|---|---|
| `failure_rate_3qubit.pdf` | `3qubit-failure-rate.pdf` | ✓ |
| `heatmaps_3qubit.pdf` | `3qubit-heatmaps.pdf` | ✓ |
| `logical_probs_3qubit.pdf` | `3qubit-logical-probs.pdf` | ✓ |
| `truncation_error_3qubit.pdf` | `3qubit-truncation-error.pdf` | |
| `correction_histograms_5qubit.pdf` | `5qubit-correction-histograms.pdf` | ✓ |
| `logical_probs_5qubit.pdf` | `5qubit-logical-probs.pdf` | ✓ |
| `heatmaps_5qubit.pdf` | `5qubit-heatmaps.pdf` | |
| `failure_rate_5qubit_erasure.pdf` | `5qubit-erasure-failure-rate.pdf` | |
| `correction_histograms_shor.pdf` | `shor-correction-histograms.pdf` | ✓ |
| `csp_mps_mpo.pdf` | `csp-tn-failure-rate.pdf` | ✓ |
| `csp_comparison.pdf` | `csp-bp-failure-rate.pdf` | ✓ |
| `bb_mps_mpo.pdf` | `bb-tn-failure-rate.pdf` | |
| `surface_bond_dimension_scaling.pdf` | `surface-bdim-scaling.pdf` | |
| `failure_rate_surface_lattice_size.pdf` | `surface-failure-rate.pdf` | |
| `vert_horiz_surface.pdf` / `qubit_order_bb.pdf` | `qubit-ordering-speedup.pdf` / `qubit-ordering-comparison.pdf` | unverified |

The thesis `tn-*.pdf`, `rep-code-tanner.pdf` and `shor-code-tanner.pdf` are
TikZ diagrams drawn in the thesis repo, not produced here.
