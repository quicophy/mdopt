"""Reproduce the two bit-flip thresholds of the surface code from one harness.

Literature targets: MWPM crosses at 10.31% (Wang-Harrington-Preskill), optimal
maximum-likelihood at the RBIM Nishimori point, 10.94(2)%. Code-capacity
bit-flip noise, rotated surface code, both decoders on identical sampled
error sets.

The per-shot records are not versioned (see README.md); rerunning this file
reproduces them exactly, since every cell's error stream is seeded as
seed + distance * 1000 + int(p * 1e5) and the decoder is deterministic.
Output: data-dem-campaign/N_d{d}_p{p:.4f}.jsonl, one JSON record per shot.
"""

import json, time
from pathlib import Path
import resource
import sys


def _rss_gb():
    # ru_maxrss is bytes on macOS but KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak * 1024 / 2**30


import numpy as np, pymatching
from qldpc.codes import SurfaceCode
from mdopt.decoding.dem import DemProblem, decode_dem

RESULTS = Path(__file__).parent / "data-dem-campaign"


def bitflip_problem(distance, p):
    code = SurfaceCode(distance, rotated=True)
    n = code.num_qubits
    h_z = np.asarray(code.matrix_z) % 2  # Z-checks detect X errors
    logicals = np.asarray(code.get_logical_ops())
    zbar = logicals[1]  # symplectic row: (x|z)
    obs = sorted(int(q) for q in np.nonzero(zbar[n:])[0])  # Z-support of Z-bar
    problem = DemProblem(
        probs=[p] * n,
        detector_rows=[sorted(map(int, np.nonzero(row)[0])) for row in h_z],
        observable_rows=[obs],
        num_detectors=h_z.shape[0],
        num_observables=1,
    )
    return problem, h_z, np.array([1 if q in obs else 0 for q in range(n)])


def run(distance, p, shots, chi=128, seed=0):
    tag = f"N_d{distance}_p{p:.4f}"
    path = RESULTS / f"{tag}.jsonl"
    done = sum(1 for _ in open(path)) if path.exists() else 0
    if done >= shots:
        return
    problem, h_z, obs_vec = bitflip_problem(distance, p)
    matcher = pymatching.Matching.from_check_matrix(
        h_z, weights=np.log((1 - p) / p), faults_matrix=obs_vec.reshape(1, -1)
    )
    rng = np.random.default_rng(seed + distance * 1000 + int(p * 1e5))
    errors = rng.random((shots, len(problem.probs))) < p
    with open(path, "a") as sink:
        for i in range(done, shots):
            mech = errors[i].astype(int)
            syndrome = h_z @ mech % 2
            truth = int(obs_vec @ mech % 2)
            t0 = time.perf_counter()
            _, flips = decode_dem(problem, syndrome, chi_max=chi)
            rec = {
                "i": i,
                "truth": truth,
                "map": int(flips[0]),
                "mwpm": int(matcher.decode(syndrome)[0]) % 2,
                "t": round(time.perf_counter() - t0, 4),
            }
            sink.write(json.dumps(rec) + "\n")
            if (i + 1) % 500 == 0:
                sink.flush()
                print(
                    f"[{tag}] {i+1}/{shots} rss={_rss_gb():.2f}GB",
                    flush=True,
                )
    rows = [json.loads(l) for l in open(path)]
    m = sum(r["map"] != r["truth"] for r in rows)
    w = sum(r["mwpm"] != r["truth"] for r in rows)
    print(
        f"[{tag}] n={len(rows)} MAP {m/len(rows):.4f} MWPM {w/len(rows):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    # The final shot counts of the campaign: 16000 per near-threshold cell for
    # d <= 9, 4000 at d = 11 and on the three wing rates. The run is resumable
    # (a cell with enough records is skipped), so the loop can be split across
    # processes by distance; d = 9 near threshold is the slow block.
    near = (0.095, 0.100, 0.104, 0.108, 0.112, 0.116)
    for distance in (5, 7, 9):
        for p in near:
            run(distance, p, shots=16000)
    for p in near:
        run(11, p, shots=4000)
    for p in (0.070, 0.085, 0.130):
        for distance in (5, 7, 9, 11):
            run(distance, p, shots=4000)
    print("NISHIMORI DONE", flush=True)
