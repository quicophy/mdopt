"""Class-sum decoder with PriorTilt solutions among its candidates.

PriorTilt (Ye and Delfosse, arXiv:2611.13373) is a belief-propagation decoder
that perturbs a few priors at random; runs with different random seeds return
different likely errors of the same syndrome. This script hands those to
:class:`mdopt.decoding.class_sum.ClassSumDecoder` through its
``candidate_sources`` option and compares, on the same shots of a stim
circuit: BP-OSD, PriorTilt alone, the class sum with BP-OSD candidates and the
class sum with the PriorTilt solutions added.

PriorTilt is not a dependency of mdopt and is not distributed with it. Its
code is licensed CC BY-NC-SA 4.0, for non-commercial use: get it from
https://github.com/ionq-publications/PriorTiltDecoder, build its extension
(``pip install ./decoder``, needs a Rust toolchain) and put the repository on
``PYTHONPATH``. The repository also holds the paper's circuits (``StimCircuit/``)
and the parameters PriorTilt was tuned with, which exist for the paper's codes
only.

Settings come from the environment: PT_CIRCUIT (path of a stim circuit,
required), PT_CODE (BB144), PT_P (0.005), PT_SHOTS (200), PT_SEED (1),
PT_RUNS (5, PriorTilt runs per syndrome), PT_STATES (256).
Output: data-dem-campaign/P_{circuit}_seed{seed}_n{shots}_k{runs}.jsonl, one
record per shot. The run is resumable (see README.md). The shot count is part
of the file name because stim's samples depend on it for a given seed.
"""

import json
import os
import time
from math import comb
from pathlib import Path

import numpy as np
import stim
from ldpc import BpOsdDecoder

from mdopt.decoding.class_sum import (
    DEFAULT_BPOSD_OPTIONS,
    ClassSumDecoder,
    problem_matrices,
)
from mdopt.decoding.dem import dem_to_problem

RESULTS = Path(__file__).parent / "data-dem-campaign"
DECODERS = ("bposd", "priortilt", "class_sum", "class_sum_priortilt")


def mcnemar(fixes, breaks):
    """Exact two-sided McNemar p-value for the discordant pair counts."""
    total = fixes + breaks
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(fixes, breaks) + 1))
    return min(1.0, 2 * tail / 2**total)


def column_map(theirs, check, probs):
    """For each column of PriorTilt's check matrix, the mechanism of mdopt's
    model with the same detectors.

    PriorTilt merges the error instructions that share their detectors into
    one column; where mdopt keeps several, the most likely one stands for the
    column.
    """
    ours = check.tocsc()
    by_detectors = {}
    for mechanism in range(ours.shape[1]):
        key = ours.indices[ours.indptr[mechanism] : ours.indptr[mechanism + 1]]
        key = tuple(sorted(key.tolist()))
        if key not in by_detectors or probs[mechanism] > probs[by_detectors[key]]:
            by_detectors[key] = mechanism
    theirs = theirs.tocsc()
    theirs.sum_duplicates()
    mapping = np.empty(theirs.shape[1], dtype=np.int64)
    for column in range(theirs.shape[1]):
        rows = theirs.indices[theirs.indptr[column] : theirs.indptr[column + 1]]
        mapping[column] = by_detectors[tuple(sorted(rows.tolist()))]
    return mapping


def priortilt_source(dem, check, probs, code, p, runs, seed):
    """A candidate source for ``ClassSumDecoder``: the solutions of ``runs``
    PriorTilt decoders with different seeds, as mechanism sets of mdopt's
    model. Runs that do not converge return nothing."""
    import priortilt  # pylint: disable=import-outside-toplevel,import-error

    decoders = [
        priortilt.PriorTilt(dem, code=code, p=p, seed=seed + run) for run in range(runs)
    ]
    mapping = column_map(priortilt.dem_to_matrices(dem).check_matrix, check, probs)

    def source(syndrome):
        solutions = []
        for decoder in decoders:
            errors = decoder.decode_errors(syndrome)
            if decoder.converged:
                solution = np.zeros(check.shape[1], dtype=np.uint8)
                solution[mapping[np.flatnonzero(errors)]] = 1
                solutions.append(solution)
        return solutions

    return source, decoders[0]


def load_records(path):
    """Records of an earlier run, by shot."""
    records = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[record["shot"]] = record
    return records


def summarize(records):
    """Failure counts and the paired comparisons of the class sum with
    PriorTilt candidates against the other decoders."""
    wrong = {name: sum(r[name] != r["truth"] for r in records) for name in DECODERS}
    print(f"shots {len(records)}: " + ", ".join(f"{k} {v}" for k, v in wrong.items()))
    for other in ("class_sum", "priortilt"):
        fixes = sum(
            r[other] != r["truth"] and r["class_sum_priortilt"] == r["truth"]
            for r in records
        )
        breaks = sum(
            r[other] == r["truth"] and r["class_sum_priortilt"] != r["truth"]
            for r in records
        )
        print(
            f"class_sum_priortilt against {other}: fixes {fixes}, breaks {breaks}, "
            f"McNemar p = {mcnemar(fixes, breaks):.3g}"
        )


def main():  # pylint: disable=too-many-locals
    """Decode the shots that are not on file yet, then summarize all of them."""
    path = Path(os.environ["PT_CIRCUIT"])
    code = os.environ.get("PT_CODE", "BB144")
    p = float(os.environ.get("PT_P", "0.005"))
    shots = int(os.environ.get("PT_SHOTS", "200"))
    seed = int(os.environ.get("PT_SEED", "1"))
    runs = int(os.environ.get("PT_RUNS", "5"))
    states = int(os.environ.get("PT_STATES", "256"))

    circuit = stim.Circuit.from_file(str(path))
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    problem = dem_to_problem(dem)
    check, observables = problem_matrices(problem)
    probs = np.asarray(problem.probs, dtype=float)
    powers = 1 << np.arange(problem.num_observables)

    source, tilt = priortilt_source(dem, check, probs, code, p, runs, seed)
    bposd = BpOsdDecoder(check, error_channel=list(probs), **DEFAULT_BPOSD_OPTIONS)
    options = {"truncation": "basis", "max_states": states, "extra_centers": 64}
    plain = ClassSumDecoder(problem, **options)
    helped = ClassSumDecoder(problem, candidate_sources=[source], **options)

    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, flips = sampler.sample(shots, separate_observables=True)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"P_{path.stem}_seed{seed}_n{shots}_k{runs}.jsonl"
    records = load_records(out)
    with open(out, "a", encoding="utf-8") as handle:
        for shot in range(shots):
            if shot in records:
                continue
            syndrome = detectors[shot].astype(np.uint8)
            start = time.time()
            record = {"shot": shot, "truth": int(flips[shot].astype(int) @ powers)}
            solution = np.asarray(bposd.decode(syndrome), dtype=np.uint8)
            record["bposd"] = int(((observables @ solution) % 2) @ powers)
            record["priortilt"] = int(np.asarray(tilt.decode(syndrome)) @ powers)
            record["class_sum"] = int(plain.decode(syndrome)[1]["predicted_class"])
            _, info = helped.decode(syndrome)
            record["class_sum_priortilt"] = int(info["predicted_class"])
            record["external"] = int(info["external"])
            record["seconds"] = round(time.time() - start, 3)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            records[shot] = record
    summarize([records[shot] for shot in range(shots)])


if __name__ == "__main__":
    main()
