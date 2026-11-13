#!/usr/bin/env bash
# [sgt] ISIC scheduler: every model x seed, one training per (model, seed), started while fewer than MAXRUNS of your
# trainings run (BRACS and BACH ones included) and the CPU load is below LOAD_MAX; GPUs alternate.
# Models start in ORDER; a model starts only after every run of the previous models has started.
#   EPOCHS=100 nohup bash tools/run_isic.sh > isic_jobs.log 2>&1 &
# Defaults: ORDER="sgt sgt_concat sgt_mshuf sgt_pshuf cgt", SEEDS="0 1 2 3 4", EPOCHS=100,
#           MAXRUNS=10, LOAD_MAX=40, GPUS="1 0", THREADS=6. Finished runs are skipped, so it can be restarted;
#           an unfinished run folder is moved to _stopped_runs_isic/ and started again.
# Needs ISIC_OUT (from: DATASET=isic bash run_cgt_bracs.sh isic_split).
S2=${S2:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../SGT" && pwd)"}
if [ -n "${SGT_ENV_FILE:-}" ]; then source "$SGT_ENV_FILE"; fi
cd "$S2" || exit 1
: "${ISIC_OUT:?export ISIC_OUT (the folder with tissue_graphs/ and weights/)}"
[ -d "$ISIC_OUT/weights/test" ] || { echo "no weights in $ISIC_OUT: run the isic_split step first"; exit 1; }
unset MORPH MODEL TASK CLASSES INIT FOLD; export ISIC_OUT OMP_NUM_THREADS="${THREADS:-6}" MKL_NUM_THREADS="${THREADS:-6}"
EP="${EPOCHS:-100}"; MAXR="${MAXRUNS:-10}"; LMAX="${LOAD_MAX:-40}"; GAP="${GAP:-60}"; POLL="${POLL:-120}"
read -r -a GL <<< "${GPUS:-1 0}"
# your python trainings only (all of them share the CPU); a shell merely mentioning train.py does not count
nrun() { pgrep -u "$(id -u)" -f "^python[0-9.]* .*train[.]py " | wc -l; }
declare -A TAGS=([cgt]="" [sgt]="sgtfull_" [cgt_mase]="cgtmase_" [cgt_sre]="cgtsre_" [sgt_mshuf]="sgtmshuf_"
                 [sgt_pshuf]="sgtpshuf_" [sgt_concat]="sgtconcat_")
i=0
for M in ${ORDER:-sgt sgt_concat sgt_mshuf sgt_pshuf cgt}; do
  [ -n "${TAGS[$M]+x}" ] || { echo "unknown model $M"; exit 1; }
  for s in ${SEEDS:-0 1 2 3 4}; do
    d="runs_${TAGS[$M]}isic_e${EP}_val_seed$s"
    ls "$d"/logs/*/test_results.txt >/dev/null 2>&1 && continue                    # finished
    pgrep -u "$(id -u)" -f "^python[0-9.]* .*train[.]py .*/${d}( |$)" >/dev/null && continue   # running
    [ -d "$d" ] && { mkdir -p _stopped_runs_isic; mv "$d" "_stopped_runs_isic/${d}_$(date +%m%d%H%M)"; }  # unfinished: start over
    while :; do
      n=$(nrun); L=$(cut -d' ' -f1 /proc/loadavg)
      [ "$n" -lt "$MAXR" ] && awk -v l="$L" -v m="$LMAX" 'BEGIN{exit !(l<m)}' && break
      sleep "$POLL"
    done
    g="${GL[$((i % ${#GL[@]}))]}"; i=$((i + 1))
    DATASET=isic MODEL=$M SEEDS=$s GPU=$g EPOCHS=$EP VAL_SPLIT=val bash run_cgt_bracs.sh train \
      > "log_isic_e${EP}_${M}_s${s}.txt" 2>&1 &
    echo "$(date '+%m-%d %H:%M') started $M seed $s on GPU $g"
    sleep "$GAP"          # let the new run appear and the load average catch up
  done
done
echo "$(date '+%m-%d %H:%M') every run started"
wait
echo "$(date '+%m-%d %H:%M') all done"
