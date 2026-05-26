#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Low-RUL Time-TRUST stability sweep
# ============================================================
# Runs structured Time-TRUST stability experiments across:
#   - C-MAPSS subsets: FD001, FD002, FD003, FD004
#   - MLP architectures: h10, h10_10, h10_10_10
#   - modes: sensors, windows
#
# Recommended first pass:
#   DRAWS_PER_BIN=3
#
# More robust final pass, if runtime is acceptable:
#   DRAWS_PER_BIN=5
#
# Example:
#   bash scripts/run_lowrul_stability_sweep.sh
#
# Optional overrides:
#   DRAWS_PER_BIN=5 bash scripts/run_lowrul_stability_sweep.sh
#   MODES="sensors" bash scripts/run_lowrul_stability_sweep.sh
#   FORCE=1 bash scripts/run_lowrul_stability_sweep.sh
# ============================================================

# -----------------------------
# Main experimental parameters
# -----------------------------
SEQ_LEN="${SEQ_LEN:-30}"
STEP="${STEP:-1}"
C="${C:-50}"
N_PER_BIN="${N_PER_BIN:-100}"
DRAWS_PER_BIN="${DRAWS_PER_BIN:-5}"
MILP_TIME_CAP="${MILP_TIME_CAP:-60}"
RUL_BINS="${RUL_BINS:-low:0:30}"
RESULTS_ROOT="${RESULTS_ROOT:-results/time_trust_stability_lowrul}"
LOG_ROOT="${LOG_ROOT:-logs/time_trust_stability_lowrul}"
FORCE="${FORCE:-0}"

# Space-separated modes. Override with: MODES="sensors" bash ...
MODES="${MODES:-sensors windows}"

# -----------------------------
# Datasets and MLP architectures
# -----------------------------
DATASETS=("FD001" "FD002" "FD003" "FD004")

# Each architecture is one quoted string because --hidden expects variable-length args.
HIDDENS=(
  "10"
  "10 10"
  "10 10 10"
)

mkdir -p "${LOG_ROOT}"
mkdir -p "${RESULTS_ROOT}"

# -----------------------------
# Environment: force CPU and reduce TF logs
# -----------------------------
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL="2"

# Optional: fix Python hash seed for slightly better reproducibility.
export PYTHONHASHSEED="0"

# -----------------------------
# Print sweep summary
# -----------------------------
echo "============================================================"
echo "Low-RUL Time-TRUST stability sweep"
echo "============================================================"
echo "Datasets        : ${DATASETS[*]}"
echo "Hidden configs  : ${HIDDENS[*]}"
echo "Modes           : ${MODES}"
echo "RUL bins        : ${RUL_BINS}"
echo "C               : ${C}"
echo "N per bin       : ${N_PER_BIN}"
echo "Draws per bin   : ${DRAWS_PER_BIN}"
echo "MILP time cap   : ${MILP_TIME_CAP}"
echo "Results root    : ${RESULTS_ROOT}"
echo "Logs root       : ${LOG_ROOT}"
echo "Force rerun     : ${FORCE}"
echo "============================================================"

# -----------------------------
# Run sweep
# -----------------------------
for DATASET in "${DATASETS[@]}"; do
  for HIDDEN in "${HIDDENS[@]}"; do
    HIDDEN_TAG="h${HIDDEN// /_}"

    COMBO_RESULTS_DIR="${RESULTS_ROOT}/${DATASET}/W${SEQ_LEN}_step${STEP}/${HIDDEN_TAG}"
    SUMMARY_FILE="${COMBO_RESULTS_DIR}/stability_summary.csv"

    if [[ -f "${SUMMARY_FILE}" && "${FORCE}" != "1" ]]; then
      echo "[SKIP] ${DATASET} ${HIDDEN_TAG}: ${SUMMARY_FILE} already exists. Use FORCE=1 to rerun."
      continue
    fi

    LOG_FILE="${LOG_ROOT}/${DATASET}_${HIDDEN_TAG}_C${C}_n${N_PER_BIN}_draws${DRAWS_PER_BIN}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "[RUN] dataset=${DATASET} hidden='${HIDDEN}' modes='${MODES}'"
    echo "[LOG] ${LOG_FILE}"
    echo "------------------------------------------------------------"

    # shellcheck disable=SC2086
    python scripts/run_trust_stability_experiment.py \
      --dataset "${DATASET}" \
      --seq-len "${SEQ_LEN}" \
      --step "${STEP}" \
      --hidden ${HIDDEN} \
      --modes ${MODES} \
      --rul-bins "${RUL_BINS}" \
      --analysis-unit bin \
      --draw-source train \
      --C "${C}" \
      --n-per-bin "${N_PER_BIN}" \
      --draws-per-bin "${DRAWS_PER_BIN}" \
      --repeat-solve 1 \
      --milp-time-cap "${MILP_TIME_CAP}" \
      --results-root "${RESULTS_ROOT}" \
      --verbose 2>&1 | tee "${LOG_FILE}"

    echo "[DONE] dataset=${DATASET} hidden=${HIDDEN_TAG}"
  done
done

# -----------------------------
# Optional post-hoc selection stability
# -----------------------------
echo ""
echo "============================================================"
echo "Running post-hoc selection stability analysis"
echo "============================================================"

POSTHOC_OUT="${RESULTS_ROOT}/selection_stability_posthoc"
python scripts/analyze_trust_selection_stability.py \
  --root "${RESULTS_ROOT}" \
  --out "${POSTHOC_OUT}"

echo ""
echo "============================================================"
echo "Sweep complete"
echo "============================================================"
echo "Main results root : ${RESULTS_ROOT}"
echo "Logs root         : ${LOG_ROOT}"
echo "Post-hoc output   : ${POSTHOC_OUT}"
echo ""
echo "Useful commands:"
echo "  find ${RESULTS_ROOT} -name stability_summary.csv -print"
echo "  find ${RESULTS_ROOT} -name selection_stability_by_n0.csv -print"
echo "  cat ${POSTHOC_OUT}/selection_stability_by_n0.csv"
echo "  cat ${POSTHOC_OUT}/group_survival_similarity.csv"
