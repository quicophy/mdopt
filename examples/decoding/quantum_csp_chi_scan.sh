#!/bin/bash
# ============================================================================
# quantum_csp_chi_scan.sh
#
# Compute Canada submission script for the chi(N) scaling experiment.
# Sweeps bond dimension chi across {4,8,16,32,64,128,256} for each
# (N, code_id) pair that has TN data in data/quantum-csp-batch-9.
# N ∈ {60,70,80,90} -- see NS below. get_code_ids() still carries ids for
# 30, 40 and 50 because their sweeps were collected in an earlier run; every
# size already holds chi ∈ {4,8,16,32,64,128,256} in data/quantum-csp-batch-9,
# so re-running them would only duplicate work. Add them back to NS if that
# data is ever discarded.
# Results go into data/quantum-csp-batch-9/ alongside the existing data there.
#
# Fixed parameters:
#   error_rate = 0.001  (clearest TN vs BP-OSD signal)
#   error_model = Bitflip
#   bias_prob   = 0.001
#   num_experiments = 5000  (~50 expected failures/code at p=0.01)
#   seed = 0
#
# Usage (local, for testing a single job):
#   bash quantum_csp_chi_scan.sh
#
# Usage (SLURM array, recommended for Compute Canada):
#   sbatch --array=0-<N_JOBS-1> quantum_csp_chi_scan.sh
# ============================================================================

#SBATCH --time=100:00:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/chi_scan_%A_%a.out
#SBATCH --error=logs/chi_scan_%A_%a.err

set -euo pipefail

# Prevent BLAS/OpenMP from oversubscribing cores when running parallel workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ── Parameters ────────────────────────────────────────────────────────────────
BATCH=9
ERROR_RATE=0.001
BIAS_PROB=0.001
ERROR_MODEL="Bitflip"
NUM_EXPERIMENTS=5000
SEED=0
TOLERANCE=0.0
CUT=0.0
NUM_PROCESSES=1
SILENT=true

# Project root: SLURM sets SLURM_SUBMIT_DIR to the directory where sbatch was
# called. For local testing, falls back to $PWD (run from project root).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT_DIR="${PROJECT_ROOT}/examples/decoding/data/quantum-csp-batch-9"

# Bond dimensions to scan (chi=400 already exists in batch-9; reused by analysis)
BOND_DIMS=(4 8 16 32 64 128 256)

# Code IDs matching data/quantum-csp-batch-9 exactly
# (case statement used instead of declare -A for bash 3.x compatibility)
get_code_ids() {
    case $1 in
        30) echo "5 8 9 17 24 30 35 39 44 77 84 87 88 91" ;;
        40) echo "0 1 2 3 4 6 10 11 13 14 19 20 21 22 24 25 28 29 30 31 33 35 36 37 39 41 43 44 45 46 47 48 49 50 51 52 54 55 56 57 58 59 60 61 63 65 66 67 68 69 71 72 75 76 77 78 79 81 84 85 86 89 90 91 94 95 96 97 98 99" ;;
        50) echo "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 71 72 73 74 75 76 77 78 79 81 82 83 84 85 86 87 88 89 90 91 92 93 95 96 97 98 99" ;;
        60) echo "0 1 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 47 49 50 51 52 54 55 56 57 58 59 60 64 66 67 69 71 72 73 74 76 77 78 79 81 85 86 87 88 89 90 91 93 94 96 97 99" ;;
        70) echo "16 20 22 25 29 30 32 33 39 41 45 46 47 54 59 61 62 64 67 72 73 78 80 85 87 95 98" ;;
        80) echo "1 3 7 25 35 38 42 47 50 53 74 79" ;;
        90) echo "17 24 30" ;;
    esac
}
# Deliberately not all seven sizes -- 30, 40 and 50 are already collected.
NS=(60 70 80 90)

mkdir -p "$OUTPUT_DIR" "${PROJECT_ROOT}/logs"

# ── Build flat job list ───────────────────────────────────────────────────────
# Each entry: "N chi code_id"
JOB_LIST=()
for N in "${NS[@]}"; do
    IDS=($(get_code_ids "$N"))
    for CHI in "${BOND_DIMS[@]}"; do
        for CODE_ID in "${IDS[@]}"; do
            JOB_LIST+=("$N $CHI $CODE_ID")
        done
    done
done

# ── Worker function ───────────────────────────────────────────────────────────
TOTAL=${#JOB_LIST[@]}

run_one_job() {
    local IDX="$1"
    local N CHI CODE_ID FNAME
    read -r N CHI CODE_ID <<< "${JOB_LIST[$IDX]}"
    FNAME="latticesize${N}_bonddim${CHI}_errorrate${ERROR_RATE}_errormodel${ERROR_MODEL}_bias_prob${BIAS_PROB}_numexperiments${NUM_EXPERIMENTS}_tolerance${TOLERANCE}_cut${CUT}_batch${BATCH}_codeid${CODE_ID}_seed${SEED}.pkl"

    if [[ -f "${OUTPUT_DIR}/${FNAME}" ]]; then
        echo "[$((IDX+1))/${TOTAL}] Skipping (already done): N=${N}, chi=${CHI}, code_id=${CODE_ID}"
        return 0
    fi

    echo "[$((IDX+1))/${TOTAL}] Starting: N=${N}, chi=${CHI}, code_id=${CODE_ID}"

    poetry run python -m mdopt.examples.decoding.quantum_csp \
        --num_qubits   "$N"                \
        --batch        "$BATCH"            \
        --code_id      "$CODE_ID"          \
        --bond_dim     "$CHI"              \
        --error_rate   "$ERROR_RATE"       \
        --bias_prob    "$BIAS_PROB"        \
        --num_experiments "$NUM_EXPERIMENTS" \
        --error_model  "$ERROR_MODEL"      \
        --seed         "$SEED"             \
        --num_processes "$NUM_PROCESSES"   \
        --silent       "$SILENT"           \
        --tolerance    "$TOLERANCE"        \
        --cut          "$CUT"

    mv "${PROJECT_ROOT}/${FNAME}" "$OUTPUT_DIR/" 2>/dev/null || true
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    run_one_job "$SLURM_ARRAY_TASK_ID"
else
    WORKERS="${WORKERS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
    echo "Total jobs: ${TOTAL} | Parallel workers: ${WORKERS}"
    for IDX in $(seq 0 $((TOTAL - 1))); do
        while (( $(jobs -rp | wc -l) >= WORKERS )); do
            sleep 1
        done
        run_one_job "$IDX" &
    done
    wait
fi

echo "Done."
