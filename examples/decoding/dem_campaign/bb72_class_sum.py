"""Class-sum decoder against BP-OSD on the bivariate bicycle [[72, 12, 6]] code.

Memory experiment under circuit-level depolarizing noise, built with qldpc.
Every shot is decoded twice from the same detector error model: by BP-OSD
(min-sum, scaling 0.9, 100 iterations, OSD-CS order 10) and by
:class:`mdopt.decoding.class_sum.ClassSumDecoder`, which uses the same BP-OSD
settings for its candidates. The summary reports both failure counts, the
shots that only one decoder gets right and the exact McNemar p-value.

Settings come from the environment: BB_ROUNDS (default 6), BB_P (0.005),
BB_SHOTS (2000), BB_SEED (95), BB_CHI (16), BB_BASIS (Z).
Output: data-dem-campaign/B_r{rounds}_p{p:.4f}_{basis}_seed{seed}_n{shots}.jsonl,
one record per shot. The run is resumable (see README.md). The shot count is
part of the file name because stim's samples depend on it for a given seed.
"""

import json
import os
import time
from math import comb
from pathlib import Path

import numpy as np
import sympy
from ldpc import BpOsdDecoder
from qldpc import circuits, codes

from mdopt.decoding.class_sum import (
    DEFAULT_BPOSD_OPTIONS,
    ClassSumDecoder,
    problem_matrices,
)
from mdopt.decoding.dem import dem_to_problem

RESULTS = Path(__file__).parent / "data-dem-campaign"


def circuit(rounds, p, basis="Z"):
    """Memory experiment of the [[72, 12, 6]] code with depolarizing noise."""
    x, y = sympy.symbols("x y")
    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    return circuits.get_memory_experiment(
        code,
        basis,
        num_rounds=rounds,
        noise_model=circuits.DepolarizingNoiseModel(p),
    )


def mcnemar(fixes, breaks):
    """Exact two-sided McNemar p-value for the discordant pair counts."""
    total = fixes + breaks
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(fixes, breaks) + 1))
    return min(1.0, 2 * tail / 2**total)


def run(rounds, p, shots, seed, chi_max=16, basis="Z"):
    """Decode ``shots`` samples with both decoders and return the records."""
    memory = circuit(rounds, p, basis)
    dem = memory.detector_error_model(decompose_errors=False, flatten_loops=True)
    problem = dem_to_problem(dem)
    check_matrix, observable_matrix = problem_matrices(problem)
    bposd = BpOsdDecoder(
        check_matrix, error_channel=list(problem.probs), **DEFAULT_BPOSD_OPTIONS
    )
    decoder = ClassSumDecoder(problem, chi_max=chi_max)
    sampler = memory.compile_detector_sampler(seed=seed)
    detections, flips = sampler.sample(shots, separate_observables=True)

    RESULTS.mkdir(exist_ok=True)
    name = f"B_r{rounds}_p{p:.4f}_{basis}_seed{seed}_n{shots}.jsonl"
    path = RESULTS / name
    records = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            records = {rec["shot"]: rec for rec in map(json.loads, handle)}
    start = time.time()
    with open(path, "a", encoding="utf-8") as handle:
        for shot in range(shots):
            if shot in records:
                continue
            syndrome = detections[shot].astype(np.uint8)
            truth = flips[shot].astype(np.uint8)
            reference = (observable_matrix @ bposd.decode(syndrome)) % 2
            began = time.time()
            predicted, info = decoder.decode(syndrome)
            record = {
                "shot": shot,
                "bposd_wrong": bool((reference != truth).any()),
                "class_sum_wrong": bool((predicted != truth).any()),
                "margin": float(info["margin"]),
                "lost": float(info["lost"]),
                "active": int(info["active"]),
                "free": int(info["free"]),
                "seconds": round(time.time() - began, 4),
            }
            records[shot] = record
            handle.write(json.dumps(record) + "\n")
            if len(records) % 500 == 0:
                handle.flush()
                elapsed = time.time() - start
                print(f"{len(records)} of {shots} shots, {elapsed:.0f} s", flush=True)
    return [records[shot] for shot in range(shots)]


def summarize(records):
    """Failure counts of both decoders and the paired McNemar test."""
    bposd = sum(rec["bposd_wrong"] for rec in records)
    class_sum = sum(rec["class_sum_wrong"] for rec in records)
    fixes = sum(rec["bposd_wrong"] and not rec["class_sum_wrong"] for rec in records)
    breaks = sum(rec["class_sum_wrong"] and not rec["bposd_wrong"] for rec in records)
    return {
        "shots": len(records),
        "bposd_failures": bposd,
        "class_sum_failures": class_sum,
        "fixes": fixes,
        "breaks": breaks,
        "mcnemar_p": mcnemar(fixes, breaks),
    }


if __name__ == "__main__":
    summary = summarize(
        run(
            rounds=int(os.environ.get("BB_ROUNDS", 6)),
            p=float(os.environ.get("BB_P", 0.005)),
            shots=int(os.environ.get("BB_SHOTS", 2000)),
            seed=int(os.environ.get("BB_SEED", 95)),
            chi_max=int(os.environ.get("BB_CHI", 16)),
            basis=os.environ.get("BB_BASIS", "Z"),
        )
    )
    print(json.dumps(summary), flush=True)
    print("BB72 CLASS SUM DONE", flush=True)
