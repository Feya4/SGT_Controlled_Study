#!/usr/bin/env bash
# Verify environment imports and execute *short* training smoke tests.
# Run from any directory after sourcing tools/runtime_paths.local.sh.
# No historical experiment directory is edited. Full training is NOT run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-check}"
case "$MODE" in check|smoke) ;; *) echo "Usage: bash tools/smoke_all_datasets.sh check|smoke" >&2; exit 2 ;; esac
LOG_DIR="${SGT_RUNTIME_LOG_DIR:-$ROOT/runtime_verification_logs}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$LOG_DIR/runtime_${MODE}_$STAMP.tsv"
printf 'dataset\tmodel\tstatus\tlog\n' > "$REPORT"

# Static verification is independent of GPU or dataset file availability.
if ! (cd "$ROOT" && python tools/verify_release.py) > "$LOG_DIR/static_$STAMP.log" 2>&1; then
  echo "Static verification failed; see $LOG_DIR/static_$STAMP.log" >&2
  exit 1
fi
if (cd "$ROOT/SGT" && bash run_cgt_bracs.sh check) > "$LOG_DIR/environment_$STAMP.log" 2>&1; then
  echo "Runtime import check: PASS ($LOG_DIR/environment_$STAMP.log)"
else
  echo "Runtime import check: FAIL ($LOG_DIR/environment_$STAMP.log)" >&2
  tail -n 20 "$LOG_DIR/environment_$STAMP.log" >&2
  exit 1
fi
if [ "$MODE" = check ]; then
  echo "Import/static checks passed. This does NOT validate graph loading or training."
  echo "To execute a short training test: bash tools/smoke_all_datasets.sh smoke"
  exit 0
fi

# Preflight processed datasets; do not generate new split files in a smoke test.

missing=0
for v in OUT OUT_MORPH ISIC_OUT CAM_OUT; do
  path="${!v:-}"
  for split in train val test; do
    if [ -z "$path" ] || [ ! -d "$path/tissue_graphs/$split" ]; then
      echo "MISSING: $v/tissue_graphs/$split" >&2; missing=1
    fi
  done
done
for split in train val test; do
  if [ -z "${BACH_OUT:-}" ] || [ ! -d "$BACH_OUT/fold0/tissue_graphs/$split" ]; then
    echo "MISSING: BACH_OUT/fold0/tissue_graphs/$split" >&2; missing=1
  fi
done
[ "$missing" = 0 ] || { echo "Prepare existing graph data; do not regenerate splits for this check." >&2; exit 2; }

# Default to two representative model paths per dataset; use MODELS to test all five.
# Example: MODELS='cgt sgt sgt_concat sgt_mshuf sgt_pshuf' bash tools/smoke_all_datasets.sh smoke
MODELS="${MODELS:-cgt sgt}"
fail=0
for ds in bracs bach isic camelyon; do
  for model in $MODELS; do
    log="$LOG_DIR/${STAMP}_${ds}_${model}.log"
    echo "Smoke: $ds / $model"
    if [ "$ds" = bracs ]; then classes=3; else classes=7; fi
    if (cd "$ROOT/SGT" && env \
        DATASET="$ds" MODEL="$model" CLASSES="$classes" \
        EPOCHS=100 SEEDS=0 FOLD=0 VAL_SPLIT=val TESTSEL=0 SELECT=val \
        GPU="${GPU:-0}" bash run_cgt_bracs.sh smoke1) > "$log" 2>&1; then
      printf '%s\t%s\tPASS\t%s\n' "$ds" "$model" "$log" >> "$REPORT"
    else
      printf '%s\t%s\tFAIL\t%s\n' "$ds" "$model" "$log" >> "$REPORT"
      tail -n 12 "$log" >&2
      fail=1
    fi
  done
done
echo "Report: $REPORT"
[ "$fail" = 0 ] || { echo "One or more smoke tests FAILED" >&2; exit 1; }
echo "All requested short training smoke tests passed; full reruns and numerical reproduction remain unverified."
