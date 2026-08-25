#!/bin/bash
# =============================================================================
# quantum_csp_rerun_cc.sh -- CSP re-runs on fir (Compute Canada)
#
# Submits the three CSP jobs identified by the 2026-08-24 data audit. Select
# one with the first argument:
#
#   bash quantum_csp_rerun_cc.sh chi_extension   #   60 jobs,  0.3M shots  P1
#   bash quantum_csp_rerun_cc.sh nan_refill      # 1920 jobs,  9.6M shots  P2
#   bash quantum_csp_rerun_cc.sh deep_p          #  360 jobs, 18.0M shots  P3
#   bash quantum_csp_rerun_cc.sh --dry-run chi_extension
#
# WHY THESE THREE
#
# 1. chi_extension -- the stored chi in {288,320,352,384} data for n = 80, 90 is
#    INVALID and must be discarded, not appended to. Its failure rate is flat in
#    chi (0.0692 / 0.0691 / 0.0695 / 0.0694 at n = 80) while chi = 256 gives
#    0.0001 and chi = 400 gives 0.0001. The rate equals ~90% of the fraction of
#    shots that carry any error at all (7.7% observed vs 8.6% of shots having a
#    non-trivial error at n = 90), i.e. nearly every real error failed at all
#    four bond dimensions. That is a broken run, not truncation.
#
# 2. nan_refill -- chi in {150, 220} at p >= 0.008 is up to 57% NaN. The NaNs
#    came from an unguarded division by an underflowed norm in
#    CanonicalMPS.marginal, fixed on 2026-08-24. data_handling.py drops NaNs via
#    np.nanmean, which is unbiased only if NaN is independent of failure -- it is
#    not, since underflow correlates with hard errors. Those cells are biased LOW.
#
# 3. deep_p -- n = 90, p = 1e-4, chi = 400 has 5 failures in 4.24M shots
#    (p_L ~ 1.2e-6). ~21M shots total are needed for ~20% relative error.
#
# The chi = 400 dataset is CLEAN (318M shots, 0.0% NaN) and needs nothing else.
#
# PREREQUISITE -- READ THIS FIRST
#
# The decoder fixes are NOT on origin/main. A fresh clone reproduces the NaNs
# this script exists to eliminate. Verify before submitting anything:
#     python -c "import inspect, mdopt.mps.canonical as c; \
#         assert 'orth_centre_norm > 0' in inspect.getsource(c.CanonicalMPS.marginal), \
#         'STOP: unpatched mdopt -- the NaN fix is missing'; print('mdopt OK')"
# =============================================================================

set -euo pipefail

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=true; shift; fi
JOB="${1:-}"

if [ -z "$JOB" ]; then
    echo "usage: $0 [--dry-run] {chi_extension|nan_refill|deep_p}" >&2
    exit 1
fi

# ── Environment ──────────────────────────────────────────────────────────────
# Skipped under --dry-run so the plan can be previewed off-cluster.
if [ "$DRY_RUN" = false ]; then
module load python/3.11.5
if [ ! -d "$HOME/envs/myenv" ]; then
    virtualenv --no-download "$HOME/envs/myenv"
fi
source "$HOME/envs/myenv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy scipy opt_einsum tqdm qecstruct more_itertools networkx
pip install git+ssh://git@github.com/quicophy/matrex.git

# Refuse to run against an unpatched mdopt: without the marginal() zero-guard
# this reproduces exactly the NaNs the re-run is meant to remove.
python - <<'PYCHECK'
import inspect, sys
import mdopt.mps.canonical as canonical
src = inspect.getsource(canonical.CanonicalMPS.marginal)
if "orth_centre_norm > 0" not in src:
    sys.exit("STOP: mdopt is missing the marginal() zero-guard (the NaN fix). "
             "Sync the patched working tree to the cluster before submitting.")
print("mdopt NaN fix present -- proceeding.")
PYCHECK
fi

# ── Fixed parameters ─────────────────────────────────────────────────────────
BATCH=9
ERROR_MODEL="Bitflip"
BIAS_PROB=1e-3
TOLERANCE=0
CUT=0
# false, matching every other cluster script. The decoder's diagnostics -- a
# collapsed posterior, a negative logical amplitude (an exact run cannot produce
# one, so it flags unconverged chi), a DMRG sweep that stopped below its own
# upper bound -- are all gated on `not silent`. Suppressing them in a run whose
# whole purpose is to explain an anomaly would throw away the evidence.
SILENT=false
WALLTIME="12:00:00"
# Memory is set per mode from a measurement, not guessed. Peak RSS of ONE decode
# at n = 90 on a light (weight-2) error:
#     chi=64 411 MB | chi=128 740 MB | chi=192 1263 MB | chi=256 1553 MB | chi=288 2160 MB
# Heavier errors need more. With 16 workers the node must hold ~16 of these, so
# the original chi-extension request of --mem=10000 (625 MB/worker) was far too
# small, and even 30000 (1.9 GB/worker) is marginal at chi >= 288.

case "$JOB" in
  chi_extension)
    # Replace the invalid chi = 288..384 data. Same codes/seed/shots as the
    # chi <= 256 sweep so the points are directly comparable.
    NUMS_QUBITS=(80 90); BOND_DIMS=(288 320 352 384)
    ERROR_RATES=(0.001); SEEDS=(0); NUM_EXPERIMENTS=5000
    # chi up to 384 at n = 90, and p = 1e-3 means ~9% of shots are non-trivial,
    # so several workers can hold a large MPS at once. 4 GB/worker.
    NUM_PROCESSES=16; MEM=64000
    ;;
  nan_refill)
    # High-p cells that were NaN-contaminated at chi = 150 and 220.
    NUMS_QUBITS=(60 80 90); BOND_DIMS=(150 220)
    ERROR_RATES=(0.008 0.01); SEEDS=(0 1 2 3 4); NUM_EXPERIMENTS=5000
    # Lower chi, but p = 0.01 makes nearly every shot non-trivial, so most
    # workers are busy at once. 2 GB/worker.
    NUM_PROCESSES=16; MEM=32000
    ;;
  deep_p)
    # Push n = 90, p = 1e-4 from 5 failures towards ~25. p_L ~ 1.2e-6 means
    # ~21M shots are needed in total and 4.24M already exist, so aim for ~18M.
    # Seeds start at 100 to stay clear of the existing 0..99 and 111..1199.
    NUMS_QUBITS=(90); BOND_DIMS=(400)
    ERROR_RATES=(0.0001); SEEDS=($(seq 2000 2119)); NUM_EXPERIMENTS=50000
    # chi = 400 is the heaviest, but at p = 1e-4 ~99% of shots hit the trivial
    # fast path, so few workers hold a big MPS simultaneously -- which is why
    # the existing chi = 400 runs completed cleanly at 30000.
    NUM_PROCESSES=16; MEM=32000
    ;;
  *)
    echo "unknown job '$JOB' (expected chi_extension|nan_refill|deep_p)" >&2
    exit 1
    ;;
esac

# Read the code ids straight off disk rather than hardcoding them: a truncated
# list would silently change the code-average the re-run is meant to match.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${REPO_ROOT}/examples/decoding/data-quantum-csp-batch-${BATCH}"
CODE_DIR_ROOT="${REPO_ROOT}/examples/decoding/data-csp-codes/batch_${BATCH}/codes"

code_ids_for() {
    local dir="${CODE_DIR_ROOT}/qubits_${1}"
    if [ ! -d "$dir" ]; then
        echo "no code directory ${dir}" >&2
        return 1
    fi
    ls "$dir" | sed -n 's/^code_\([0-9]*\)\.json$/\1/p' | sort -n
}

# ── Submit ───────────────────────────────────────────────────────────────────
n_jobs=0
for seed in "${SEEDS[@]}"; do
  for num_qubits in "${NUMS_QUBITS[@]}"; do
    for code_id in $(code_ids_for "$num_qubits"); do
      for bond_dim in "${BOND_DIMS[@]}"; do
        for error_rate in "${ERROR_RATES[@]}"; do
          tag="csp-${JOB}-${num_qubits}-${bond_dim}-${error_rate}-${code_id}-${seed}"
          job_script="submit-${tag}.sh"
          cat > "$job_script" <<EOS
#!/bin/bash
#SBATCH --time=${WALLTIME}
#SBATCH --cpus-per-task=${NUM_PROCESSES}
#SBATCH --mem=${MEM}
#SBATCH --job-name=${tag}
#SBATCH --output=${tag}-%j.out

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
module load python/3.11.5
source "\$HOME/envs/myenv/bin/activate"

# quantum_csp.py writes its pickle to a BARE filename, i.e. into the job's
# working directory -- historically the submit directory, from where
# cleanup_cc.sh swept the files up afterwards. Land them straight in the dataset
# directory instead, keeping the repo importable via PYTHONPATH so that
# python -m still resolves after the cd.
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"
mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

python -m mdopt.examples.decoding.quantum_csp \\
    --num_qubits ${num_qubits} --batch ${BATCH} --code_id ${code_id} \\
    --bond_dim ${bond_dim} --error_rate ${error_rate} \\
    --num_experiments ${NUM_EXPERIMENTS} --bias_prob ${BIAS_PROB} \\
    --error_model "${ERROR_MODEL}" --seed ${seed} \\
    --num_processes ${NUM_PROCESSES} --silent ${SILENT} \\
    --tolerance ${TOLERANCE} --cut ${CUT}
EOS
          if [ "$DRY_RUN" = true ]; then
              echo "[dry-run] would submit ${tag}"
          else
              sbatch "$job_script"
          fi
          rm -f "$job_script"
          n_jobs=$((n_jobs+1))
        done
      done
    done
  done
done

echo "${JOB}: ${n_jobs} jobs $([ "$DRY_RUN" = true ] && echo 'previewed' || echo 'submitted')."
[ "$DRY_RUN" = true ] || echo "Track with: squeue -u \${USER}"
