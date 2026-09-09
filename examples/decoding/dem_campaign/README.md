# DEM decoding campaign: harnesses and how to regenerate the data

These three scripts produced every number and figure in the DEM decoder
report on pull request #544. The per-shot records they write are **not
versioned**: every harness is deterministic and seeded, so the files are
reproduced bit for bit by rerunning the script. Outputs land in
`data-dem-campaign/` next to the scripts, which the repository's `data*/`
rule keeps out of git.

| Script | Experiment | Output files | Shots | Cost |
| --- | --- | --- | --- | --- |
| `nishimori.py` | Code-capacity bit-flip thresholds, MAP vs MWPM, rotated surface code d = 5, 7, 9, 11 | `N_d{d}_p{p:.4f}.jsonl`, 36 cells | 16000 per near-threshold cell for d ≤ 9, 4000 otherwise | ~26 CPU-hours (d = 9 near threshold is half of it) |
| `dem_rerun.py` | Circuit-level d = 5, r = 5, p = 0.5 % memory-Z with a per-shot χ ladder, calibration audit, MWPM and belief-matching | `D2_d5r5_p0.005.jsonl` | 2000 | ~83 CPU-hours (149 s per shot) |
| `repro_tnd3d.py` | Fig. 1d of Piveteau, Chubb and Renes, PRX Quantum 5, 040303: memory-X, rounds = d, p = 0.6 to 0.9 % | `T_d{d}r{d}_p{p:.4f}.jsonl`, 4 cells per distance | 2000 per cell | ~12 CPU-hours at d = 3 (5.6 s per shot) |

All costs are single-core wall time on an Apple M-series laptop with
`chi_max` as set in the scripts. Every run is resumable: a cell whose file
already holds enough records is skipped, and a partial file is continued
from its last record.

## Running

From the repository root, inside the Poetry environment:

```bash
cd examples/decoding/dem_campaign

# Nishimori thresholds. Split by distance across processes if you like:
# the loop in __main__ is resumable and each cell is independent.
python nishimori.py

# Circuit-level d=5 ladder cell. RERUN_SHOTS overrides the shot count.
RERUN_SHOTS=2000 python dem_rerun.py

# Fig. 1d reproduction. REPRO_D selects the distance, REPRO_SHOTS the count.
REPRO_D=3 REPRO_SHOTS=2000 python repro_tnd3d.py
```

`dem_rerun.py` and `repro_tnd3d.py` also decode every shot with belief
matching when the optional `beliefmatching` package is importable
(`pip install beliefmatching`; it is not a declared dependency because its
numpy pin cannot be reconciled with ours). Without it the `bm` column is
simply omitted.

Close the laptop lid and the run sleeps. Use `caffeinate -is -w <pid>` on
macOS, or run on a machine that does not suspend.

## Seeding

- `nishimori.py`: the error stream of the cell (d, p) is
  `numpy.random.default_rng(seed + d * 1000 + int(p * 1e5))` with `seed = 0`,
  drawn as one `(shots, n)` Bernoulli block. Changing the shot count of a
  cell therefore does not change the errors of the shots already recorded.
- `dem_rerun.py`: the stim detector sampler is seeded with 5555 for the
  d = 5 cell.
- `repro_tnd3d.py`: the sampler seed of the cell (d, p) is
  `int(p * 1e6) + d`.

The MPS decoder itself is deterministic, so a regenerated file matches the
original record for record; only the timing column `t` differs.

## Record formats

`nishimori.py` writes one JSON object per shot:

```
{"i": shot, "truth": observable flip, "map": MPS-MAP verdict,
 "mwpm": PyMatching verdict, "t": decode seconds}
```

`dem_rerun.py` and `repro_tnd3d.py` write, per shot, the verdict and
posterior margin at every rung of the ladder plus the escalated value when
the top two rungs disagreed:

```
{"i", "truth", "flips": {chi: verdict}, "margins": {chi: max posterior},
 "dev_<chi>": max posterior change from the previous rung,
 "map", "margin", "chi": the value and rung finally used,
 "dev_esc": present when escalated to chi=128,
 "artefact_<chi>": present when that rung raised ArithmeticError,
 "mwpm", "bm", "t"}
```

A logical error rate is `mean(rec[decoder] != rec["truth"])` over the
records that carry that decoder's column. The online calibration check
compares `sum(1 - rec["margin"])` with the observed MAP failures and the
harness aborts when the two differ by more than three standard deviations.

## Reproducing the reported figures

- **Nishimori thresholds**: for each
  distance, the logical error rate of `map` and of `mwpm` against p from the
  `N_*` files, with binomial error bars. The threshold quoted in the report
  is the crossing of the d ≥ 9 curves from a linear fit in (p − p_th) L^{1/ν}
  with ν fixed at the literature value; the plain crossing and the ω = 2
  finite-size correction are reported alongside. Literature targets:
  MWPM 10.31 % (Wang, Harrington and Preskill), optimal 10.94(2) %.
- **Fig. 1d comparison**: per p, the logical error rate of `map`, `bm` and
  `mwpm` from the `T_d3r3_*` files, overlaid on the published d = 3 curve.
- **Circuit-level d = 5 cell**: the ladder error rates
  `mean(rec["flips"][str(chi)] != rec["truth"])` for χ = 16, 32, 64, the
  final `map`, `bm` and `mwpm` rates, and the calibration z printed by the
  harness at the end of the run.

The numbers behind the report are printed by each harness when a cell
completes, so a regeneration can be checked against the PR text without
any plotting code.
