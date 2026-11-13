#!/usr/bin/env bash
# [sgt] CAMELYON16-derived patches (Level 2, slide-disjoint split), end to end and unattended; stops at the first failed check:
#   1 trial (2 patches per label and split): 0 failed, median >= 20 nodes, split + smoke test pass
#   2 graphs for all 12,024 patches (one builder, gentle on the CPU)  3 train/val/test + connectivity weights
#   4 smoke test on the full data   5 waits until the ISIC trainings have finished, then the queue (25 runs)
#   6 predictions + every comparison (Delta = SGT - control, from the saved predictions) -> camelyon_final.txt
#   nohup bash tools/camelyon_pipeline.sh > camelyon_pipeline.log 2>&1 &
# Restartable: finished graphs and runs are kept and skipped.
set -u
G=${G:-/workdir1.8t/fei27/CGT/GNNMD}
source ~/cgt_sgt_env.sh >/dev/null 2>&1; cd "$G/CGT_sgt2" || exit 1
R=$PWD
export CAM_ROOT=$G/BACH/camon2018
export CAM_SPLIT=$CAM_ROOT/data/classifier/revised_graph_experiment_seed42/precomputed_graphs_fast_slide_disjoint_L2/split.csv
export OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4}
unset MODEL MORPH TASK CLASSES INIT FOLD OUT OUT_MORPH
say() { echo "$(date '+%m-%d %H:%M') $*"; }
stop() { say "STOP: $*"; exit 1; }
quiet() { "$@" 2>&1 | grep -v -i -E 'warn|dgl.graph\(data\)|deprecated|start_label|patch size|out=out|rcount'; return "${PIPESTATUS[0]}"; }

env CAM_OUT="$G/cam_data" EPOCHS=100 bash tools/run_camelyon.sh > cam_jobs.log 2>&1
n=$(ls -d runs_*cam_e100_val_seed*/logs/*/test_results.txt 2>/dev/null | wc -l); say "finished runs: $n / 25"

say "6/6 predictions and comparisons -> camelyon_final.txt"
{
  for M in sgt sgt_concat sgt_mshuf sgt_pshuf cgt; do
    env CAM_OUT="$G/cam_data" DATASET=camelyon EPOCHS=100 MODEL=$M bash run_cgt_bracs.sh predict
  done
  cd "$R/core"
  for X in sgtconcat sgtmshuf sgtpshuf cgt; do
    P=$([ $X = cgt ] && echo "" || echo "${X}_")
    echo "##### SGT - $X (Delta = SGT - control; from the saved predictions)"
    python compare_runs.py --a "$R/runs_${P}cam_e100_val_seed*" --b "$R/runs_sgtfull_cam_e100_val_seed*" \
           --names "$X" SGT --from_preds --perclass
  done
  echo "===== done $(date '+%m-%d %H:%M') ====="
} 2>&1 | grep -v -i -E 'warn|dgl.graph\(data\)|deprecated' > "$R/camelyon_final.txt"
say "all done: camelyon_final.txt"
