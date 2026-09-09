# Benchmarks

`bench_suite.py` is the profiling and correctness harness behind the
optimisation work. Every workload is deterministic (fixed seeds) and returns a
*fingerprint*; `baseline.json` holds the fingerprints of the unoptimised code,
and a change is only accepted if the fingerprints still match.

| Workload | What it exercises | Contract |
| --- | --- | --- |
| `surface_bitflip` | 5x5 surface code, bit-flip noise, 6 shots, χ = 64, natural qubit order | verdicts exact, posterior entries within 1e-2 |
| `css_optimised` | the same code under `qubit_order_strategy="Optimised"` (reverse Cuthill-McKee) | as above |
| `shor_depolarising` | Shor code, depolarising noise, 40 shots, χ = 128: the dense-readout path | as above |
| `classical_ldpc` | random (3,4) LDPC code, XOR constraints + dephasing-DMRG readout | overlaps exact to 1e-10 |
| `dmrg_ground_state` | DMRG on a 24-site transverse-field Ising chain | energy exact to 1e-10 |
| `dem_d3` | circuit-level detector error model, d=3 r=3 p=0.8%, eight busiest syndromes, χ = 32 | verdicts exact, class masses within 1e-2 |
| `dem_d5` | the same at d=5 r=5 p=0.5%, busiest syndrome, χ = 32 | as above |

"Exact" means the value must agree to 1e-10; χ-truncated posterior entries get
1e-2 because a different but equally valid SVD gauge in a near-degenerate
spectrum changes which directions the truncation keeps (verdicts and
converged results are unaffected). A NaN never matches.

## Running

```bash
# time every workload and compare against the committed baseline
python benchmarks/bench_suite.py --check

# one workload, with a cProfile dump and a text top-30 in benchmarks/results/
python benchmarks/bench_suite.py --workload dem_d5 --profile

# record a new baseline for a workload (only after the change is validated
# some other way: exact enumeration, agreement at converged chi)
python benchmarks/bench_suite.py --workload NAME --write-baseline
```

`--check` and `--write-baseline` are mutually exclusive, and a targeted
`--write-baseline` touches only the workloads it ran. `results/` is
gitignored; `summary.json` there keeps the last wall time per workload.

## Writing a baseline from the reference code

The baseline must come from code *without* the optimisations under test.
With a clean checkout of `main` next to this one:

```bash
PYTHONPATH=/path/to/mdopt-main python benchmarks/bench_suite.py --workload NAME --write-baseline
PYTHONPATH=/path/to/this-checkout python benchmarks/bench_suite.py --workload NAME --check
```

`python -c` puts the current directory first on `sys.path`, so verify which
package a run imports (`mdopt.__file__`) from a neutral directory before
trusting a measurement.
