"""Reproduce the two bit-flip thresholds of the surface code from one harness.

Literature targets: MWPM crosses at 10.31% (Wang-Harrington-Preskill), optimal
maximum-likelihood at the RBIM Nishimori point, 10.94(2)%. Code-capacity
bit-flip noise, rotated surface code, both decoders on identical sampled
error sets.

NOTE: this file is the reconstructed record of the harness that produced the
Nishimori campaign (the original lived in a /tmp scratchpad that macOS purged
after three days, along with the raw per-shot data; the derived results are
recorded in PR #544). Rerunning it reproduces the campaign exactly: every
cell's error stream is seeded as seed + distance * 1000 + int(p * 1e5).
"""

import json, time
from pathlib import Path
import numpy as np, psutil, pymatching
from qldpc.codes import SurfaceCode
from mdopt.decoding.dem import DemProblem, decode_dem

RESULTS = Path(__file__).parent / "dem_results"
PROC = psutil.Process()


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
                    f"[{tag}] {i+1}/{shots} rss={PROC.memory_info().rss/2**30:.2f}GB",
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
    # Near-threshold grid (d<=9 later topped up to 16000 shots), then wings.
    for distance in (5, 7, 9, 11):
        for p in (0.095, 0.100, 0.104, 0.108, 0.112, 0.116):
            run(distance, p, shots=4000)
    for p in (0.070, 0.085, 0.130):
        for distance in (5, 7, 9, 11):
            run(distance, p, shots=4000)
    print("NISHIMORI DONE", flush=True)
