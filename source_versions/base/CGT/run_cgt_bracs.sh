#!/usr/bin/env bash
# [rerun-addition] Run the GitHub CGT code on the HACT tissue graphs used by SGT (same graphs, split and labels).
#
#   export GRAPHS=/workdir1.8t/fei27/CGT/GNNMD/data/tissue_graphs_official_morph
#   export OUT=/workdir1.8t/fei27/CGT/GNNMD/cgt_github_data
#   GPU=0 bash run_cgt_bracs.sh check      # environment, GPU, Cython build of algos.pyx
#   bash run_cgt_bracs.sh convert          # HACT graphs -> CGT input format (minutes)
#   bash run_cgt_bracs.sh weights          # connectivity matrices (build_weights.py)
#   GPU=0 bash run_cgt_bracs.sh infer      # released checkpoint on the test split (diagnostic)
#   GPU=0 SEEDS="0 1 2" bash run_cgt_bracs.sh train   # CGT's own training: 200 epochs, batch 4, lr 1e-4
#   bash run_cgt_bracs.sh summary          # per-seed and mean +- SD (checkpoint with best VALIDATION wF1)
#
# SGT = CGT + MASE + SRE ([sgt]); each extends one CGT module (details in core/grapher.py and RERUN.md):
#   MASE  area + convexity embeddings join CGT's connectivity (degree) embedding, aggregated by CGT's LCA
#   SRE   per-head hop-distance bias added to CGT's distance-based connectivity bias in self-attention
# Chosen with MODEL (all but cgt need OUT_MORPH, see convert_morph below):
#   MODEL=cgt         CGT as released (= MODEL unset, MORPH unset)         -> runs_val_seed<s>
#   MODEL=sgt         CGT + MASE + SRE                                      -> runs_sgtfull_val_seed<s>
#   MODEL=cgt_mase    CGT + MASE  (= the earlier MORPH=real)    MODEL=cgt_sre   CGT + SRE   (ablations)
#   MODEL=sgt_mshuf   SGT, morphology bins permuted within each graph        (randomised morphology)
#   MODEL=sgt_pshuf   SGT, hop matrix permuted within each graph             (randomised structure)
#   MODEL=sgt_concat  CGT + SRE, raw area/convexity through one linear layer, FFN 259 (parameter-matched)
#   GPU=0 bash run_cgt_bracs.sh smoke      # every MODEL above: 1 epoch on 3 batches, gradients checked (~2 min)
#   MODEL=sgt GPU=0 bash run_cgt_bracs.sh train ;  MODEL=sgt bash run_cgt_bracs.sh summary
#   MODEL=sgt GPU=0 bash run_cgt_bracs.sh predict  # per-graph test predictions of finished runs (for compare)
#   bash run_cgt_bracs.sh compare          # paired tests + bootstrap; A, B = run globs, NAMES="CGT SGT"
# Earlier variant: MORPH=real adds the morphology embeddings to CGT; MORPH=shuffle and MORPH=concat
# are its controls; MORPH=none (default) is CGT exactly as released. MORPH != none needs the graphs with
# morphology attributes:  export OUT_MORPH=...;  bash run_cgt_bracs.sh convert_morph;  bash run_cgt_bracs.sh weights_morph
#   MORPH=real GPU=0 SEEDS="0 1 2" bash run_cgt_bracs.sh train      -> runs_sgt_val_seed<s>
#   MORPH=real bash run_cgt_bracs.sh summary
# VAL_SPLIT=val (default): checkpoints chosen on the validation split, as for SGT.
# VAL_SPLIT=test reproduces the released code, which chooses checkpoints on the test split.
set -euo pipefail
STEP="${1:?give a step: check | convert | weights | convert_morph | weights_morph | infer | smoke | train | summary | predict | compare}"
GPU="${GPU:-0}"; VAL_SPLIT="${VAL_SPLIT:-val}"; SEEDS="${SEEDS:-0 1 2 3 4}"; MORPH="${MORPH:-none}"
case "$MORPH" in
  none) TAG="" ;; real) TAG="sgt_" ;; shuffle) TAG="sgtshuf_" ;; concat) TAG="sgtcat_" ;;
  *) echo "MORPH must be none | real | shuffle | concat"; exit 1 ;;
esac
EXTRA=""; [ "$MORPH" = concat ] && EXTRA="--ffn_dim 259"      # parameter count matched to MORPH=real
EXTRA="--morph $MORPH $EXTRA"
MODEL="${MODEL:-}"
if [ -n "$MODEL" ]; then
  [ "$MORPH" != none ] && { echo "set MODEL or MORPH, not both"; exit 1; }
  case "$MODEL" in
    cgt)        TAG="" ;               EXTRA="" ;;
    sgt)        TAG="sgtfull_" ;       EXTRA="--sgt_mase --sgt_sre" ;;
    cgt_mase)   TAG="cgtmase_" ;       EXTRA="--sgt_mase" ;;
    cgt_sre)    TAG="cgtsre_" ;        EXTRA="--sgt_sre" ;;
    sgt_mshuf)  TAG="sgtmshuf_" ;      EXTRA="--sgt_mase --sgt_sre --mase_shuffle" ;;
    sgt_pshuf)  TAG="sgtpshuf_" ;      EXTRA="--sgt_mase --sgt_sre --sre_shuffle" ;;
    sgt_concat) TAG="sgtconcat_" ;     EXTRA="--sgt_sre --morph concat --ffn_dim 259" ;;
    *) echo "unknown MODEL $MODEL"; exit 1 ;;
  esac
fi
export CUDA_VISIBLE_DEVICES="$GPU"
unset PYTORCH_CUDA_ALLOC_CONF
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/core"

data_dir() {
  if [ "$MORPH" = none ] && { [ -z "$MODEL" ] || [ "$MODEL" = cgt ]; }; then echo "${OUT:?export OUT}"
  else echo "${OUT_MORPH:?export OUT_MORPH}"; fi
}

case "$STEP" in
  smoke)
    rc=0
    for M in cgt sgt cgt_mase cgt_sre sgt_mshuf sgt_pshuf sgt_concat; do
      log="/tmp/cgt_smoke_${USER:-u}_$M.log"
      if MODEL=$M MORPH=none bash "$HERE/run_cgt_bracs.sh" smoke1 >"$log" 2>&1; then
        printf '%-11s OK    %s\n' "$M" "$(grep -m1 '^parameters:' "$log" | cut -d'|' -f1)"
      else
        printf '%-11s FAIL  (log: %s)\n' "$M" "$log"; tail -n 15 "$log"; rc=1
      fi
    done
    echo "expected parameters: cgt 8,434,149 | cgt_mase 8,451,625 | cgt_sre 8,434,281 |"
    echo "                     sgt, sgt_mshuf, sgt_pshuf 8,451,757 | sgt_concat 8,454,345"
    exit $rc ;;
  smoke1)
    DATA="$(data_dir)"; d="/tmp/cgt_smoke_${USER:-u}_${MODEL:-cgt}"; rm -rf "$d"
    # shellcheck disable=SC2086
    python train.py --tg_path "$DATA/tissue_graphs/" --val_split "$VAL_SPLIT" --seed 0 --model_path "$d" \
                    --smoke_batches 3 $EXTRA
    grep -h . "$d"/logs/*/test_results.txt | head -3; rm -rf "$d" ;;
  check)
    python - <<'PY'
import sys, torch
print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
import dgl; print('dgl', dgl.__version__)
import histocartography; print('histocartography OK')
import sklearn, yaml, h5py; print('sklearn', sklearn.__version__)
import wrapper; print('algos.pyx compiled OK')
import grapher; print('CGT model imports OK')
PY
    ;;
  convert) : "${GRAPHS:?export GRAPHS}"; : "${OUT:?export OUT}"
    python convert_hact_graphs.py --src "$GRAPHS" --out "$OUT" ;;
  weights) : "${OUT:?export OUT}"
    python build_weights.py --tg_root "$OUT" ;;
  convert_morph) : "${GRAPHS:?export GRAPHS}"; : "${OUT_MORPH:?export OUT_MORPH}"
    python convert_hact_graphs.py --src "$GRAPHS" --out "$OUT_MORPH" --morph_bins ;;
  weights_morph) : "${OUT_MORPH:?export OUT_MORPH}"
    python build_weights.py --tg_root "$OUT_MORPH" ;;
  infer)   : "${OUT:?export OUT}"
    python inference.py --tg_path "$OUT/tissue_graphs/" --pretrained ./logs/model_best_val.pt ;;
  train)
    DATA="$(data_dir)"
    for s in $SEEDS; do
      d="$HERE/runs_${TAG}${VAL_SPLIT}_seed$s"
      if ls "$d"/logs/*/test_results.txt >/dev/null 2>&1; then echo "done: seed $s"; continue; fi
      echo "===== ${MODEL:-morph=$MORPH} seed $s (checkpoint selection on $VAL_SPLIT): $EXTRA ====="
      # shellcheck disable=SC2086
      python train.py --tg_path "$DATA/tissue_graphs/" --val_split "$VAL_SPLIT" --seed "$s" --model_path "$d" $EXTRA
    done ;;
  summary)
    python summarize_cgt_runs.py --runs "$HERE/runs_${TAG}${VAL_SPLIT}_seed*" ;;
  predict)
    # OUT_MORPH holds every node field, so it serves all models; the node features are the same as in OUT
    if [ -n "${OUT_MORPH:-}" ]; then DATA="$OUT_MORPH"; else DATA="$(data_dir)"; fi
    python predict_runs.py --tg_path "$DATA/tissue_graphs/" --runs "${RUNS:-$HERE/runs_${TAG}${VAL_SPLIT}_seed*}" ;;
  compare)
    # shellcheck disable=SC2086
    python compare_runs.py --a "${A:-$HERE/runs_${VAL_SPLIT}_seed*}" --b "${B:-$HERE/runs_sgtfull_${VAL_SPLIT}_seed*}" \
                           --names ${NAMES:-CGT SGT} ${CSV:+--csv "$CSV"} ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
