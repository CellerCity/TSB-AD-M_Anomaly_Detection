#!/usr/bin/env bash
# Ablation sweep over (context_length, horizon) for TimesFM zero-shot anomaly detection.
#
# - Configurable per dataset (SMD, Exathlon, etc).
# - Skips cells whose output CSV already exists (resume-friendly after a crash).
# - Logs each cell's stdout/stderr to logs/.
# - Continues past individual cell failures.
# - Calls average_methods.py at the end to produce one summary CSV per dataset.
#
# Usage:
#   bash run_ablation.sh --dataset SMD
#   bash run_ablation.sh --dataset SMD --score_method mse
#   bash run_ablation.sh --dataset SMD --score_methods "interval mse"
#   bash run_ablation.sh --dataset Exathlon --force
#   bash run_ablation.sh --dataset SMD --data_pattern './custom/path/*.csv'

set -u  # error on unset vars; deliberately NOT using -e so one cell failing
        # doesn't abort the whole sweep.

# -------------------- DEFAULT CONFIG --------------------
CONTEXTS=(256 512 1024)
HORIZONS=(5 10 30 50 100)
SCORE_METHODS=("interval")    # space-separated list; e.g. --score_methods "interval mse"
AGG_METHOD="l2"
DATASET=""
DATA_PATTERN=""           # if empty, derived from DATASET
DATA_ROOT="./mTSBench"    # base dir for dataset folders
RESULTS_ROOT="ablation_results"
LOGS_ROOT="ablation_logs"
SUMMARY_DIR="ablation_summaries"
FORCE=0

# -------------------- ARG PARSING --------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";      shift 2 ;;
        --data_pattern)  DATA_PATTERN="$2"; shift 2 ;;
        --data_root)     DATA_ROOT="$2";    shift 2 ;;
        --score_method)  SCORE_METHODS=("$2"); shift 2 ;;
        --score_methods) read -ra SCORE_METHODS <<< "$2"; shift 2 ;;
        --agg_method)    AGG_METHOD="$2";   shift 2 ;;
        --force)         FORCE=1;           shift   ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required (e.g. --dataset SMD)" >&2
    exit 1
fi

# Derive data pattern from dataset name if not overridden.
if [[ -z "$DATA_PATTERN" ]]; then
    DATA_PATTERN="${DATA_ROOT}/${DATASET}/*test.csv"
fi

# Per-dataset output dirs so sweeps don't collide.
RESULTS_DIR="${RESULTS_ROOT}/${DATASET}"
LOGS_DIR="${LOGS_ROOT}/${DATASET}"
SUMMARY_CSV="${SUMMARY_DIR}/${DATASET}_summary.csv"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$SUMMARY_DIR"

# Confirm the glob actually matches something before we start.
shopt -s nullglob
MATCHED=( $DATA_PATTERN )
shopt -u nullglob
if [[ ${#MATCHED[@]} -eq 0 ]]; then
    echo "ERROR: data_pattern '$DATA_PATTERN' matched 0 files." >&2
    echo "  Check that the dataset exists under $DATA_ROOT/$DATASET/" >&2
    echo "  Or pass --data_pattern explicitly." >&2
    exit 1
fi

echo "================================================================"
echo "Ablation sweep: $DATASET"
echo "  data pattern:   $DATA_PATTERN"
echo "  files matched:  ${#MATCHED[@]}"
echo "  contexts:       ${CONTEXTS[*]}"
echo "  horizons:       ${HORIZONS[*]}"
echo "  score_methods:  ${SCORE_METHODS[*]}"
echo "  agg_method:     $AGG_METHOD"
echo "  results dir:    $RESULTS_DIR/"
echo "  logs dir:       $LOGS_DIR/"
echo "  summary csv:    $SUMMARY_CSV"
echo "  force re-run:   $FORCE"
echo "================================================================"

# -------------------- RUN GRID --------------------
TOTAL=$((${#CONTEXTS[@]} * ${#HORIZONS[@]} * ${#SCORE_METHODS[@]}))
COUNTER=0
SUCCEEDED=()
SKIPPED=()
FAILED=()

# Parallel arrays for the summary call.
SUMMARY_FILES=()
SUMMARY_CTX=()
SUMMARY_HOR=()
SUMMARY_SCORE=()

START_TIME=$(date +%s)

for sm in "${SCORE_METHODS[@]}"; do
  for ctx in "${CONTEXTS[@]}"; do
    for hor in "${HORIZONS[@]}"; do
        COUNTER=$((COUNTER + 1))
        TAG="c${ctx}_h${hor}_${sm}"
        OUT_CSV="${RESULTS_DIR}/${TAG}.csv"
        LOG_FILE="${LOGS_DIR}/${TAG}.log"

        echo ""
        echo "[${COUNTER}/${TOTAL}] $DATASET context=${ctx} horizon=${hor} score=${sm}"

        if [[ -f "$OUT_CSV" && $FORCE -eq 0 ]]; then
            echo "  -> Output exists, skipping (use --force to re-run): $OUT_CSV"
            SKIPPED+=("$TAG")
            SUMMARY_FILES+=("$OUT_CSV")
            SUMMARY_CTX+=("$ctx")
            SUMMARY_HOR+=("$hor")
            SUMMARY_SCORE+=("$sm")
            continue
        fi

        CELL_START=$(date +%s)
        python timesFM_modularised.py \
            --data_pattern   "$DATA_PATTERN" \
            --context_length $ctx \
            --horizon        $hor \
            --score_method   "$sm" \
            --agg_method     "$AGG_METHOD" \
            --save_path      "$OUT_CSV" \
            > "$LOG_FILE" 2>&1
        STATUS=$?
        CELL_END=$(date +%s)
        ELAPSED=$((CELL_END - CELL_START))

        if [[ $STATUS -eq 0 && -f "$OUT_CSV" ]]; then
            echo "  -> OK in ${ELAPSED}s   ($OUT_CSV)"
            SUCCEEDED+=("$TAG")
            SUMMARY_FILES+=("$OUT_CSV")
            SUMMARY_CTX+=("$ctx")
            SUMMARY_HOR+=("$hor")
            SUMMARY_SCORE+=("$sm")
        else
            echo "  -> FAILED (exit=$STATUS, see $LOG_FILE)"
            FAILED+=("$TAG")
        fi
    done
  done
done

TOTAL_TIME=$(($(date +%s) - START_TIME))

# -------------------- SUMMARY --------------------
echo ""
echo "================================================================"
echo "Sweep complete for $DATASET in $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s"
echo "  succeeded: ${#SUCCEEDED[@]} / $TOTAL"
echo "  skipped:   ${#SKIPPED[@]} (output already existed)"
echo "  failed:    ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  failed cells: ${FAILED[*]}"
fi
echo "================================================================"

if [[ ${#SUMMARY_FILES[@]} -eq 0 ]]; then
    echo "No output files to summarize. Exiting."
    exit 1
fi

echo ""
echo "Aggregating ${#SUMMARY_FILES[@]} cells into $SUMMARY_CSV ..."
python average_methods.py \
    --inputs        "${SUMMARY_FILES[@]}" \
    --contexts      "${SUMMARY_CTX[@]}" \
    --horizons      "${SUMMARY_HOR[@]}" \
    --score_methods "${SUMMARY_SCORE[@]}" \
    --output        "$SUMMARY_CSV"

echo ""
echo "Done. Final summary: $SUMMARY_CSV"