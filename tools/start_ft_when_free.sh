#!/usr/bin/env bash
# [sgt] Fine-tuning setting: every model starts from CGT's released checkpoint (added parts at zero, so each
# starts as exactly that model), same recipe, seeds 0-4, 100 epochs, both checkpoint rules.
# Starts one queue per model in MODELS, each only when the CPU load is below LOAD_MAX, so running jobs keep
# their speed.  Defaults: MODELS="cgt sgt", FT_GPU=1, LOAD_MAX=30.
#   nohup bash tools/start_ft_when_free.sh > ft_waiter.log 2>&1 &
#   MODELS="sgt_mshuf sgt_pshuf sgt_concat" FT_GPU=0 nohup bash tools/start_ft_when_free.sh > ft_waiter2.log 2>&1 &
S2="${S2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../SGT" && pwd)}"
if [ -n "${SGT_ENV_FILE:-}" ]; then source "$SGT_ENV_FILE"; fi
cd "$S2" || exit 1
unset MORPH MODEL; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CK="$S2/core/logs/model_best_val.pt"; [ -f "$CK" ] || { echo "missing $CK"; exit 1; }
G="${FT_GPU:-1}"; LMAX="${LOAD_MAX:-30}"
for M in ${MODELS:-cgt sgt}; do
  if [ -f "log_ft_$M.txt" ] && pgrep -f "MODEL=$M GPU=[0-9] EPOCHS=100 INIT=" >/dev/null; then
    echo "$(date '+%m-%d %H:%M') $M: fine-tuning queue already running - skipped"; continue
  fi
  while :; do
    L=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$L" -v m="$LMAX" 'BEGIN{exit !(l<m)}' && break
    echo "$(date '+%m-%d %H:%M') $M: CPU load $L >= $LMAX - waiting"; sleep 300
  done
  echo "$(date '+%m-%d %H:%M') $M: starting on GPU $G -> log_ft_$M.txt"
  nohup bash -c "MODEL=$M GPU=$G EPOCHS=100 INIT=$CK SEEDS='0 1 2 3 4' VAL_SPLIT=val bash run_cgt_bracs.sh train" > "log_ft_$M.txt" 2>&1 &
  sleep 600   # let the load average reflect the new run before starting the next
done
echo "$(date '+%m-%d %H:%M') all requested queues started"
