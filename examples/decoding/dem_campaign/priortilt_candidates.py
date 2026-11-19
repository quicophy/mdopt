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
Output: data-dem-campaign/P_{circuit}_{code}_p{p}_seed{seed}_n{shots}_k{runs}_m{states}.jsonl,
one record per shot. The run is resumable (see README.md): every setting that
changes a verdict is part of the file name, the shot count because stim's
samples depend on it for a given seed, and PriorTilt is reseeded from each
syndrome, so a resumed run gives the records of an uninterrupted one.
"""

import hashlib
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


def run_seed(seed, run, syndrome):
    """Seed of PriorTilt run ``run`` on ``syndrome``.

    PriorTilt draws its prior offsets from a generator that advances with
    every decode. Reseeding it from the syndrome before each decode makes a
    solution a function of the syndrome alone, so a resumed run decodes the
    remaining shots exactly as an uninterrupted one would.
    """
    digest = hashlib.blake2b(
        np.ascontiguousarray(syndrome, dtype=np.uint8).tobytes(), digest_size=6
    ).digest()
    return (seed * 1_000_003 + run * 7919 + int.from_bytes(digest, "little")) % 2**62


def priortilt_source(dem, check, probs, code, p, runs, seed):
    """A candidate source for ``ClassSumDecoder`` and PriorTilt as a decoder
    of its own.

    The source returns the solutions of ``runs`` PriorTilt runs with different
    seeds, as mechanism sets of mdopt's model; runs that do not converge
    return nothing. The second function is the first of those runs alone and
    returns the observable flips PriorTilt predicts.
    """
    import priortilt  # pylint: disable=import-outside-toplevel,import-error

    decoder = priortilt.PriorTilt(dem, code=code, p=p, seed=seed)
    mapping = column_map(priortilt.dem_to_matrices(dem).check_matrix, check, probs)

    def source(syndrome):
        solutions = []
        for run in range(runs):
            decoder.seed = run_seed(seed, run, syndrome)
            errors = decoder.decode_errors(syndrome)
            if decoder.converged:
                solution = np.zeros(check.shape[1], dtype=np.uint8)
                solution[mapping[np.flatnonzero(errors)]] = 1
                solutions.append(solution)
        return solutions

    def alone(syndrome):
        decoder.seed = run_seed(seed, 0, syndrome)
        return np.asarray(decoder.decode(syndrome))

    return source, alone


def load_records(path):
    """Records of an earlier run, by shot.

    A run that was interrupted in the middle of a write leaves an incomplete
    last line. That line is cut off the file, so the run resumes after the
    last complete record. A broken line anywhere else is corruption and
    raises.
    """
    records = {}
    if not path.exists():
        return records
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    complete = 0
    for number, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines) - 1:
                break
            raise
        records[record["shot"]] = record
        complete += 1
    if complete < len(lines) or (lines and not lines[-1].endswith("\n")):
        kept = "".join(lines[:complete])
        if kept and not kept.endswith("\n"):
            kept += "\n"
        path.write_text(kept, encoding="utf-8")
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
    name = f"P_{path.stem}_{code}_p{p}_seed{seed}_n{shots}_k{runs}_m{states}"
    out = RESULTS / f"{name}.jsonl"
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
            record["priortilt"] = int(tilt(syndrome) @ powers)
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
