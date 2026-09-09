"""Phase D rerun driver: per-shot chi ladder, calibrated, triple-paired.

This is the harness behind the d=5, r=5, p=0.5% circuit-level cell reported
in PR #544 (output: data-dem-campaign/D2_d5r5_p0.005.jsonl; see README.md
for how to regenerate it). Lessons from an earlier, contaminated chi=8 run
and its post-mortem audit are structural here:
- Per-shot convergence certificates are unreliable: representative spread
  is gauge-blind to truncation, and one chi-doubling can plateau (a shot
  agreed to 2e-9 between chi=16 and 32 and was wrong until chi=64). So
  EVERY shot is decoded at the full ladder (16, 32, 64); a flip
  disagreement between the top two rungs escalates that shot to chi=128.
  Convergence is then an *ensemble* statement: LER(32) vs LER(64) within
  binomial error, plus a small escalation fraction.
- The decoder's own posteriors are audited online: a true MAP decoder is
  calibrated, so observed failures must match mean(1 - margin) at the top
  rung. A |z| > 3 divergence aborts.
- Every shot is also decoded by MWPM and belief-matching on the same
  syndrome: MAP must not statistically lose to either.
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

try:
    from beliefmatching import BeliefMatching
except ImportError:  # pragma: no cover - optional third decoder
    # beliefmatching's metadata pins a numpy range Poetry cannot reconcile
    # with ours, so it is not a declared dependency; `pip install
    # beliefmatching` works. Without it the harness still runs and simply
    # records no belief-matching column.
    BeliefMatching = None

from mdopt.decoding.dem import decode_dem, dem_to_problem

RESULTS = Path(__file__).parent / "data-dem-campaign"
RSS_LIMIT_GB = 7.0
CALIBRATION_Z_ABORT = 3.0


def _rss_gb():
    # ru_maxrss is bytes on macOS but KiB on Linux; a plain /2**30 would
    # under-report 1024x on the cluster and never trip the guard.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak * 1024 / 2**30


def surface(distance, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def run(tag, circuit, shots, seed, ladder=(16, 32, 64), escalate=128):
    path = RESULTS / f"{tag}.jsonl"
    done = sum(1 for _ in open(path)) if path.exists() else 0
    problem = dem_to_problem(
        circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    )
    matcher = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    # BeliefMatching requires the decomposed DEM: BP runs on the full
    # hypergraph (the ^-hints preserve it) and matching uses the decomposition.
    beliefm = (
        BeliefMatching(
            circuit.detector_error_model(decompose_errors=True), max_bp_iters=20
        )
        if BeliefMatching is not None
        else None
    )
    if beliefm is None:
        print(f"[{tag}] beliefmatching not installed; bm column omitted", flush=True)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detections, observables = sampler.sample(shots, separate_observables=True)
    print(
        f"[{tag}] ladder={ladder} escalate={escalate} resuming at {done}/{shots}",
        flush=True,
    )

    pred_fail = obs_fail = 0.0
    escalations = 0
    with open(path, "a") as sink:
        for i in range(done, shots):
            if _rss_gb() > RSS_LIMIT_GB:
                print(f"[{tag}] MEMORY GUARD at {i}", flush=True)
                return
            syn = detections[i].astype(int)
            truth = int(observables[i][0])
            t0 = time.perf_counter()
            rec = {"i": i, "truth": truth}
            flips_by_chi, margins_by_chi = {}, {}
            prev_norm = None
            for chi in ladder:
                try:
                    masses, flips = decode_dem(problem, syn, chi_max=chi)
                except ArithmeticError as exc:
                    # A truncation artefact (negative class mass or a
                    # collapsed vector) at this rung: record it and move up
                    # the ladder rather than abort a multi-day run.
                    rec[f"artefact_{chi}"] = str(exc)
                    continue
                norm = masses / masses.sum()
                flips_by_chi[chi] = int(flips[0])
                margins_by_chi[chi] = float(np.max(norm))
                if prev_norm is not None:
                    rec[f"dev_{chi}"] = float(np.max(np.abs(norm - prev_norm)))
                prev_norm = norm
            if not flips_by_chi:
                rec["error"] = "every rung raised; scored as a failure"
                rec["map"], rec["margin"], rec["chi"] = 1 - truth, 0.0, ladder[-1]
                # Keep the online calibration consistent: margin 0 predicts a
                # failure with probability 1, and one is observed.
                pred_fail += 1.0
                rec["flips"], rec["margins"] = {}, {}
                rec["t"] = round(time.perf_counter() - t0, 3)
                rec["mwpm"] = int(matcher.decode(syn)[0]) % 2
                if beliefm is not None:
                    rec["bm"] = int(beliefm.decode(syn)[0]) % 2
                obs_fail += 1
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
                continue
            usable = [c for c in ladder if c in flips_by_chi]
            top, second = usable[-1], (usable[-2] if len(usable) > 1 else usable[-1])
            rec["flips"] = flips_by_chi
            rec["margins"] = margins_by_chi
            rec["map"] = flips_by_chi[top]
            rec["margin"] = margins_by_chi[top]
            rec["chi"] = top
            if flips_by_chi[top] != flips_by_chi[second] or len(usable) < len(ladder):
                try:
                    masses, flips = decode_dem(problem, syn, chi_max=escalate)
                    norm = masses / masses.sum()
                    rec["map"] = int(flips[0])
                    rec["margin"] = float(np.max(norm))
                    rec["chi"] = escalate
                    rec["dev_esc"] = float(np.max(np.abs(norm - prev_norm)))
                except ArithmeticError as exc:
                    rec[f"artefact_{escalate}"] = str(exc)
                escalations += 1
            rec["t"] = round(time.perf_counter() - t0, 3)
            rec["mwpm"] = int(matcher.decode(syn)[0]) % 2
            if beliefm is not None:
                rec["bm"] = int(beliefm.decode(syn)[0]) % 2
            pred_fail += 1.0 - rec["margin"]
            obs_fail += int(rec["map"] != truth)
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            n = i - done + 1
            if n % 200 == 0:
                z = (obs_fail - pred_fail) / np.sqrt(max(pred_fail, 1.0))
                print(
                    f"[{tag}] {i+1}/{shots} calib: predicted {pred_fail:.1f} "
                    f"observed {int(obs_fail)} z={z:+.2f} escalations={escalations}",
                    flush=True,
                )
                if abs(z) > CALIBRATION_Z_ABORT:
                    print(
                        f"[{tag}] CALIBRATION ABORT: |z|={abs(z):.1f} - decoder "
                        f"is not MAP at chi={top}",
                        flush=True,
                    )
                    return

    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    for dec in ("map", "mwpm", "bm"):
        # Only rows carrying this decoder count, in numerator and
        # denominator: a run resumed after (un)installing beliefmatching
        # has a bm column on part of its rows.
        scored = [r for r in rows if dec in r]
        if scored:
            f = sum(r[dec] != r["truth"] for r in scored)
            print(
                f"[{tag}] {dec}: fails={f}/{len(scored)} ({f/len(scored):.4f})",
                flush=True,
            )
    for chi in ladder:
        # Rows from a resumed run with a different ladder lack this rung;
        # score only the rows that carry it, in numerator and denominator.
        scored = [r for r in rows if str(chi) in {str(k) for k in r.get("flips", {})}]
        if scored:
            f = sum(r["flips"][str(chi)] != r["truth"] for r in scored)
            print(
                f"[{tag}] ladder LER(chi={chi}): {f}/{len(scored)} "
                f"({f/len(scored):.4f})",
                flush=True,
            )
    # Recomputed from every loaded row so a resumed run's summary covers the
    # records it inherited, not just this invocation's.
    pf = sum(1 - r["margin"] for r in rows)
    of = sum(r["map"] != r["truth"] for r in rows)
    print(
        f"[{tag}] final calibration: predicted {pf:.1f} observed {of} "
        f"z={(of-pf)/np.sqrt(max(pf,1)):+.2f}",
        flush=True,
    )
    print(f"[{tag}] complete ({n})", flush=True)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    shots = int(os.environ.get("RERUN_SHOTS", 2000))
    run("D2_d5r5_p0.005", surface(5, 5, 0.005), shots, seed=5555)
    print("RERUN DONE", flush=True)
