#!/usr/bin/env python
"""Execute the example notebooks and fail if any of them errors.

This is a smoke test, not a results run. It answers one question: does each
notebook still execute against the current library? Two silent breakages that
motivated it both came from 0978dbf (Backend options, #424) and sat unnoticed
for ten months:

  * ``mps_mpo_contract`` lost its ``result_to_explicit`` argument, which
    ``mps-rand-circ`` still passed;
  * the truncation helpers coerced ``int(chi_max)`` before taking a min, so the
    ``chi_max=np.inf`` idiom four notebooks use for their untruncated reference
    curve raised ``OverflowError``.

Executed output is deliberately **discarded**. CI runs with ``MDOPT_NB_FAST=1``,
which shrinks the workloads to a token size, so its outputs are not the numbers
the documentation should show. The committed outputs come from a full-scale run
and reach the docs through ``generate_docs.sh``.

Usage::

    python scripts/run_notebooks.py                 # every eligible notebook
    python scripts/run_notebooks.py path/to/nb.ipynb
    MDOPT_NB_FAST=1 python scripts/run_notebooks.py  # CI-scale workloads
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent

# gpu_example targets Colab with a GPU and force-reinstalls numpy mid-session; it
# cannot run here and is kept exactly as committed. tmp* is scratch, plotting* is
# data-only and needs the gitignored simulation tree.
SKIP_PREFIXES = ("tmp", "plotting")
SKIP_NAMES = frozenset({"gpu_example.ipynb"})


def eligible() -> list[Path]:
    """Every notebook this script is willing to execute, in a stable order."""
    found = []
    for path in sorted(glob.glob(str(REPO_ROOT / "examples" / "*" / "*.ipynb"))):
        name = os.path.basename(path)
        if name in SKIP_NAMES or name.startswith(SKIP_PREFIXES):
            continue
        found.append(Path(path))
    return found


def execute(path: Path, timeout: int, inplace: bool = False) -> tuple[bool, float, str]:
    """Run one notebook. Returns (ok, seconds, detail).

    With ``inplace``, the executed notebook (outputs and all) is written back.
    Only ever do that for a full-scale run: saving MDOPT_NB_FAST output would
    publish token-workload numbers as if they were results.
    """
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        # Notebooks resolve some paths relative to their own directory.
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    started = time.perf_counter()

    def save() -> None:
        if inplace:
            nbformat.write(notebook, path)

    try:
        client.execute()
    except CellExecutionError:
        elapsed = time.perf_counter() - started
        for index, cell in enumerate(notebook.cells):
            errors = [
                out
                for out in cell.get("outputs", [])
                if out.get("output_type") == "error"
            ]
            if errors:
                first = errors[0]
                save()
                return (
                    False,
                    elapsed,
                    f"cell {index}: {first.get('ename')}: {first.get('evalue')}",
                )
        save()
        return False, elapsed, "cell error"
    except Exception as exc:  # timeout, dead kernel, ...
        save()
        return False, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"
    save()
    return True, time.perf_counter() - started, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("MDOPT_NB_TIMEOUT", "1800")),
        help="per-cell timeout in seconds (default 1800)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="write executed outputs back into the notebook (full-scale runs only)",
    )
    args = parser.parse_args()

    targets = [p.resolve() for p in args.notebooks] if args.notebooks else eligible()
    if not targets:
        print("no notebooks to run", file=sys.stderr)
        return 1

    fast = bool(os.environ.get("MDOPT_NB_FAST"))
    if fast and args.inplace:
        print(
            "refusing --inplace with MDOPT_NB_FAST=1: that would commit "
            "token-workload numbers as results",
            file=sys.stderr,
        )
        return 2
    print(
        f"Executing {len(targets)} notebook(s); MDOPT_NB_FAST={'1' if fast else '0'}\n"
    )

    failures = []
    for path in targets:
        rel = path.relative_to(REPO_ROOT)
        print(f"==> {rel}", flush=True)
        ok, seconds, detail = execute(path, args.timeout, args.inplace)
        print(
            f"    {'OK  ' if ok else 'FAIL'} {seconds:8.1f}s  {detail}".rstrip(),
            flush=True,
        )
        if not ok:
            failures.append((rel, detail))

    print()
    if failures:
        print(f"{len(failures)} of {len(targets)} notebooks failed:")
        for rel, detail in failures:
            print(f"  {rel}: {detail}")
        return 1
    print(f"All {len(targets)} notebooks executed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
