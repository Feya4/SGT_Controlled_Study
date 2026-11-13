#!/usr/bin/env bash
# [rerun-addition] One-command pipeline for re-running CGT on BRACS.
#
#   BRACS=/data/BRACS_RoI/previous_version  OUT=/data/cgt_data  GPU=2  bash run_cgt.sh all
#
# Steps (run one at a time by naming it instead of "all"):
#   check    environment + GPU sanity check, compiles algos.pyx
#   tg       build tissue graphs for train/val/test   (slow: hours)
#   weights  build connectivity weights from the graphs (minutes)
#   infer    evaluate the released checkpoint on test (expect ~66.9 weighted F1)
#   train    train from scratch, then test the 3 selected checkpoints
#
# Optional variables: VAL_SPLIT=test|val (default test = original code),
#                     SEED=0  EPOCHS=200  BATCH=4  LR=1e-4
# Logs go to ./rerun_logs/. On the shared server run it detached, e.g.
#   nohup env BRACS=... OUT=... GPU=2 bash run_cgt.sh all > rerun_all.log 2>&1 &
set -euo pipefail

STEP="${1:-all}"
: "${BRACS:?set BRACS to the folder that holds train/ val/ test/ of the BRACS RoI set}"
: "${OUT:?set OUT to a folder for tissue graphs and weights}"
GPU="${GPU:-0}"
VAL_SPLIT="${VAL_SPLIT:-test}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-4}"
LR="${LR:-1e-4}"

export CUDA_VISIBLE_DEVICES="$GPU"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$HERE/rerun_logs"; mkdir -p "$LOGDIR" "$OUT"
STAMP="$(date +%Y%m%d_%H%M%S)"
cd "$HERE/core"

log() { echo -e "\n===== [$STEP] $* ($(date '+%F %T'), GPU $GPU) ====="; }

do_check() {
  log "environment check"
  python - <<'EOF'
import sys, torch
print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda build', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print('GPU:', torch.cuda.get_device_name(0), '| compute capability', cap)
    if cap[0] >= 9:
        print('WARNING: torch 1.9.1+cu111 has no kernels for compute capability >= 9.0 (H100 / RTX 50xx).')
    x = torch.randn(256, 256, device='cuda'); (x @ x).sum().item(); print('GPU matmul OK')
import dgl, histocartography, sklearn, yaml
print('dgl', dgl.__version__, '| histocartography OK | sklearn', sklearn.__version__)
import wrapper   # compiles algos.pyx with Cython on first import
print('algos.pyx compiled OK')
EOF
}

do_tg() {
  for split in train val test; do
    [ -d "$BRACS/$split" ] || { echo "missing $BRACS/$split"; exit 1; }
    log "tissue graphs: $split"
    python build_tg.py --image_path "$BRACS/$split" --save_path "$OUT" --split "$split"
  done
}

do_weights() {
  log "connectivity weights"
  python build_weights.py --tg_root "$OUT"
}

do_infer() {
  log "inference with released checkpoint"
  python inference.py --tg_path "$OUT/tissue_graphs/" --pretrained ./logs/model_best_val.pt
}

do_train() {
  log "training (val_split=$VAL_SPLIT seed=$SEED epochs=$EPOCHS)"
  python train.py --tg_path "$OUT/tissue_graphs/" --val_split "$VAL_SPLIT" --seed "$SEED" \
                  --epochs "$EPOCHS" -b "$BATCH" -l "$LR" --model_path "$HERE/runs_${VAL_SPLIT}_seed${SEED}"
}

run() { "do_$1" 2>&1 | tee "$LOGDIR/${1}_${STAMP}.log"; }

case "$STEP" in
  all)   for s in check tg weights infer train; do STEP=$s; run "$s"; done ;;
  check|tg|weights|infer|train) run "$STEP" ;;
  *) echo "unknown step: $STEP"; exit 1 ;;
esac
