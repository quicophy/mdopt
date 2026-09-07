"""Reproduce Fig. 1d of Piveteau, Chubb & Renes, PRX Quantum 5, 040303 (2024).

Their setup, verified against scripts/sweep_circ_matching.jl in ChriPiv/tndecoder3d:
stim surface_code:rotated_memory_x, rounds = d, after_clifford_depolarization =
before_measure_flip_probability = after_reset_flip_probability = p, with
d in {3, 5, 7} and p from 0.6% to 0.9%. They compare their 3D TN decoder
(threshold ~0.8%) against PyMatching (~0.78%); belief-matching was not
publicly available to them, so ours is an extra column, and our chi ladder
plus escalation replaces their fixed simple-update/MPS bond dimensions.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import stim

import dem_rerun
from dem_rerun import run


def circuit(distance, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


if __name__ == "__main__":
    distance = int(os.environ.get("REPRO_D", 3))
    shots = int(os.environ.get("REPRO_SHOTS", 2000))
    for p in (0.006, 0.007, 0.008, 0.009):
        run(
            f"T_d{distance}r{distance}_p{p:.4f}",
            circuit(distance, p),
            shots=shots,
            seed=int(p * 1e6) + distance,
            ladder=(16, 32, 64),
            escalate=128,
        )
    print("REPRO DONE", flush=True)
