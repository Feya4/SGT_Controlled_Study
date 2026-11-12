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
#   TESTSEL=1 (train): also keep the best-on-TEST checkpoint each epoch (released-CGT rule) next to the validation one
#   SELECT=test (summary, compare): report the best-on-test results instead (optimistic; label them as CGT's rule)
# Earlier variant: MORPH=real adds the morphology embeddings to CGT; MORPH=shuffle and MORPH=concat
# are its controls; MORPH=none (default) is CGT exactly as released. MORPH != none needs the graphs with
# morphology attributes:  export OUT_MORPH=...;  bash run_cgt_bracs.sh convert_morph;  bash run_cgt_bracs.sh weights_morph
#   MORPH=real GPU=0 SEEDS="0 1 2" bash run_cgt_bracs.sh train      -> runs_sgt_val_seed<s>
#   MORPH=real bash run_cgt_bracs.sh summary
# VAL_SPLIT=val (default): checkpoints chosen on the validation split, as for SGT.
# VAL_SPLIT=test reproduces the released code, which chooses checkpoints on the test split.
#
# DATASET=bach: BACH microscopy (normal / benign / in situ / invasive), 5-fold cross-validation, graphs built
# exactly as for BRACS. Fold k: test = fold k, val = fold k+1, train = the other three. Run folders get
# bach_f<k>_ (e.g. runs_sgtfull_bach_f0_e100_val_seed0); compare pools the five folds per seed (--cv).
#   export BACH_IMAGES=.../BACH/BACH_data/images  BACH_GRAPHS=.../bach_graphs  BACH_OUT=.../bach_cv
#   BACH_PATIENTS: the organisers' patient file; found automatically as ICIAR2018_BACH_dataset_micro_patient* next
#   to BACH_IMAGES (folds then grouped by patient; without it, by class only)
#   DATASET=bach bash run_cgt_bracs.sh bach_patients        # shows what the patient file says (changes nothing)
#   DATASET=bach GPU=0 bash run_cgt_bracs.sh bach_graphs    # 400 tissue graphs (+ area, solidity)
#   DATASET=bach bash run_cgt_bracs.sh bach_folds           # 5 folds + connectivity weights
#   DATASET=bach EPOCHS=100 GPU=0 bash run_cgt_bracs.sh smoke
#   DATASET=bach EPOCHS=100 MODEL=sgt FOLD=0 SEEDS=0 GPU=0 bash run_cgt_bracs.sh train   (tools/run_bach.sh: all)
#   DATASET=bach EPOCHS=100 MODEL=sgt bash run_cgt_bracs.sh predict     # every fold
#   DATASET=bach EPOCHS=100 PERCLASS=1 bash run_cgt_bracs.sh compare    # CGT vs SGT, pooled folds
#
# DATASET=isic: ISIC 2018 Task 3 (MEL, NV, BCC, AKIEC, BKL, DF, VASC) on the fixed lesion-grouped split
# (train 7,047 / val 1,468 / test 1,500), graphs built with the BRACS builder (stain normalisation off by default:
# dermoscopy is not H&E stained; STAIN_NORM=1 applies it). Run folders get isic_ (e.g. runs_sgtfull_isic_e100_val_seed0).
#   export ISIC_IMAGES=.../sgt_impl_ISIC1/data/ISIC2018_Task3_Training_Input  ISIC_SPLIT=.../isic2018_lesion_split.csv
#   export ISIC_GRAPHS=.../isic_graphs  ISIC_OUT=.../isic_data
#   DATASET=isic TRIAL=2 GPU=0 bash run_cgt_bracs.sh isic_graphs     # trial: 2 images per class and split
#   DATASET=isic SHARD=0 NSHARDS=2 GPU=0 bash run_cgt_bracs.sh isic_graphs   (and SHARD=1 on GPU 1, in parallel)
#   DATASET=isic bash run_cgt_bracs.sh isic_split            # train/val/test folders + connectivity weights
#   DATASET=isic EPOCHS=100 GPU=0 bash run_cgt_bracs.sh smoke
#   EPOCHS=100 nohup bash tools/run_isic.sh > isic_jobs.log 2>&1 &          # every model x seed
#   DATASET=isic EPOCHS=100 MODEL=sgt bash run_cgt_bracs.sh predict ;  DATASET=isic EPOCHS=100 PERCLASS=1 bash run_cgt_bracs.sh compare
set -euo pipefail
STEP="${1:?give a step: check | convert | weights | convert_morph | weights_morph | infer | smoke | train | summary | predict | compare | bach_patients | bach_graphs | bach_folds | isic_graphs | isic_split}"
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
# INIT=<weights>: start from trained weights (e.g. CGT's released core/logs/model_best_val.pt) instead of random;
# MASE/SRE start at zero. Run folders get ft_ so fine-tuned and from-scratch runs never mix.
FTP=""; if [ -n "${INIT:-}" ]; then EXTRA="$EXTRA --init_from $INIT"; TAG="${TAG}ft_"; FTP="ft_"; fi
# CLASSES=3: train directly on the BRACS three-class task (benign N,PB,UDH / atypical FEA,ADH / malignant DCIS,IC).
# Run folders get c3_; predict/summary/compare follow. (TASK=bracs3 instead maps 7-class runs to the three groups.)
C3P=""; PARG=""
if [ "${CLASSES:-7}" = 3 ]; then
  EXTRA="$EXTRA --task bracs3"; TAG="${TAG}c3_"; C3P="c3_"; PARG="--task bracs3"
  if [ -n "${TASK:-}" ] && [ "$TASK" != bracs7 ]; then echo "CLASSES=3 runs are already three-class: leave TASK unset"; exit 1; fi
elif [ "${CLASSES:-7}" != 7 ]; then echo "CLASSES must be 7 or 3"; exit 1; fi
DATASET="${DATASET:-bracs}"; CVARG=""
if [ "$DATASET" = bach ]; then
  [ "${CLASSES:-7}" = 7 ] && [ -z "${TASK:-}" ] || { echo "CLASSES and TASK apply to BRACS only"; exit 1; }
  [ -z "${INIT:-}" ] || { echo "INIT (released BRACS weights) does not apply to BACH"; exit 1; }
  EXTRA="$EXTRA --task bach4"; PARG="--task bach4"; CVARG="--cv"
  TAG="${TAG}bach_f@_"; C3P="${C3P}bach_f@_"     # @ = the fold: one fold for train/predict, every fold (*) for compare
elif [ "$DATASET" = isic ]; then
  [ "${CLASSES:-7}" = 7 ] && [ -z "${TASK:-}" ] || { echo "CLASSES and TASK apply to BRACS only"; exit 1; }
  [ -z "${INIT:-}" ] || { echo "INIT (released BRACS weights) does not apply to ISIC"; exit 1; }
  EXTRA="$EXTRA --task isic7"; PARG="--task isic7"
  TAG="${TAG}isic_"; C3P="${C3P}isic_"
elif [ "$DATASET" != bracs ]; then echo "DATASET must be bracs, bach or isic"; exit 1; fi
FOLDS="${FOLD:-0 1 2 3 4}"
if [ "$DATASET" = bach ] && [ -z "${BACH_PATIENTS:-}" ] && [ -n "${BACH_IMAGES:-}" ]; then
  BACH_PATIENTS="$(ls -d "$BACH_IMAGES"/../ICIAR2018_BACH_dataset_micro_patient* "$BACH_IMAGES"/ICIAR2018_BACH_dataset_micro_patient* 2>/dev/null | head -1 || true)"
fi
# EPOCHS=100: shorter schedule than the 200 of CGT; run folders get e100_ so they never mix with 200-epoch runs
if [ -n "${EPOCHS:-}" ]; then EXTRA="$EXTRA --epochs $EPOCHS"; TAG="${TAG}e${EPOCHS}_"; EP="e${EPOCHS}_"; else EP=""; fi
# TESTSEL=1: also test after every epoch and keep the best-on-test checkpoint (released-CGT rule) next to the
# best-on-validation one; training itself is unchanged. SELECT=test: summary/compare read the best-on-test results.
[ -n "${TESTSEL:-}" ] && EXTRA="$EXTRA --test_every_epoch"
MARG=""; [ "${SELECT:-val}" = test ] && MARG="--metric best_test_weighted_f1_score"
# SCORE=val (compare): compare the runs' best validation wF1 instead of test scores (e.g. fine-tuned runs)
[ "${SCORE:-test}" = val ] && MARG="$MARG --score val --from_preds"
# TASK=bracs3 (compare): BRACS three-class task (benign N,PB,UDH / atypical FEA,ADH / malignant DCIS,IC), scored
# from the 7-class predictions (run the predict step first). Default: the 7-class task.
[ -n "${TASK:-}" ] && [ "$TASK" != bracs7 ] && MARG="$MARG --groups $TASK"
export CUDA_VISIBLE_DEVICES="$GPU"
unset PYTORCH_CUDA_ALLOC_CONF
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/core"

data_dir() {   # BACH: $1 = fold (its graphs carry the morphology fields, so they serve every model)
  if [ "$DATASET" = bach ]; then echo "${BACH_OUT:?export BACH_OUT}/fold$1"; return; fi
  if [ "$DATASET" = isic ]; then echo "${ISIC_OUT:?export ISIC_OUT}"; return; fi   # its graphs carry the morphology fields
  if [ "$MORPH" = none ] && { [ -z "$MODEL" ] || [ "$MODEL" = cgt ]; }; then echo "${OUT:?export OUT}"
  else echo "${OUT_MORPH:?export OUT_MORPH}"; fi
}

case "$STEP" in
  smoke)
    rc=0
    for M in cgt sgt cgt_mase cgt_sre sgt_mshuf sgt_pshuf sgt_concat; do
      log="/tmp/cgt_smoke_${USER:-u}_$M.log"
      if MODEL=$M MORPH=none FOLD="${FOLD:-0}" bash "$HERE/run_cgt_bracs.sh" smoke1 >"$log" 2>&1; then
        printf '%-11s OK    %s\n' "$M" "$(grep -m1 '^parameters:' "$log" | cut -d'|' -f1)"
      else
        printf '%-11s FAIL  (log: %s)\n' "$M" "$log"; tail -n 15 "$log"; rc=1
      fi
    done
    if [ "$DATASET" = isic ]; then
      echo "expected parameters (ISIC, 7 classes): cgt 8,434,149 | cgt_mase 8,451,625 | cgt_sre 8,434,281 |"
      echo "                                       sgt, sgt_mshuf, sgt_pshuf 8,451,757 | sgt_concat 8,454,345"
    elif [ "$DATASET" = bach ]; then
      echo "expected parameters (BACH, 4 classes): cgt 8,432,604 | cgt_mase 8,450,080 | cgt_sre 8,432,736 |"
      echo "                                       sgt, sgt_mshuf, sgt_pshuf 8,450,212 | sgt_concat 8,452,800"
    elif [ "${CLASSES:-7}" = 3 ]; then
      echo "expected parameters (3 classes): cgt 8,432,089 | cgt_mase 8,449,565 | cgt_sre 8,432,221 |"
      echo "                                 sgt, sgt_mshuf, sgt_pshuf 8,449,697 | sgt_concat 8,452,285"
    else
      echo "expected parameters: cgt 8,434,149 | cgt_mase 8,451,625 | cgt_sre 8,434,281 |"
      echo "                     sgt, sgt_mshuf, sgt_pshuf 8,451,757 | sgt_concat 8,454,345"
    fi
    exit $rc ;;
  smoke1)
    DATA="$(data_dir "${FOLD:-0}")"; d="/tmp/cgt_smoke_${USER:-u}_${MODEL:-cgt}"; rm -rf "$d"
    # shellcheck disable=SC2086
    python train.py --tg_path "$DATA/tissue_graphs/" --val_split "$VAL_SPLIT" --seed 0 --model_path "$d" \
                    --smoke_batches 3 $EXTRA
    cut -f1-3 "$d"/logs/*/test_results.txt; rm -rf "$d" ;;
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
    [ "$DATASET" = bach ] || FOLDS=-          # BRACS: one fixed split
    for k in $FOLDS; do
      DATA="$(data_dir "$k")"
      for s in $SEEDS; do
        d="$HERE/runs_${TAG//@/$k}${VAL_SPLIT}_seed$s"
        if ls "$d"/logs/*/test_results.txt >/dev/null 2>&1; then echo "done: seed $s"; continue; fi
        FL=""; [ "$DATASET" = bach ] && FL=" BACH fold $k"
        echo "===== ${MODEL:-morph=$MORPH} seed $s (checkpoint selection on $VAL_SPLIT)$FL: $EXTRA ====="
        # shellcheck disable=SC2086
        python train.py --tg_path "$DATA/tissue_graphs/" --val_split "$VAL_SPLIT" --seed "$s" --model_path "$d" $EXTRA
      done
    done ;;
  summary)
    # shellcheck disable=SC2086
    python summarize_cgt_runs.py --runs "$HERE/runs_${TAG//@/*}${VAL_SPLIT}_seed*" $MARG ;;
  predict)
    if [ "$DATASET" = bach ]; then          # every fold against its own test/val graphs
      for k in $FOLDS; do
        # shellcheck disable=SC2086
        python predict_runs.py --tg_path "$(data_dir "$k")/tissue_graphs/" --runs "$HERE/runs_${TAG//@/$k}${VAL_SPLIT}_seed*" --all $PARG
      done
      exit 0
    fi
    # OUT_MORPH holds every node field, so it serves all models; the node features are the same as in OUT
    if [ "$DATASET" = isic ]; then DATA="$(data_dir)"
    elif [ -n "${OUT_MORPH:-}" ]; then DATA="$OUT_MORPH"; else DATA="$(data_dir)"; fi
    # shellcheck disable=SC2086
    python predict_runs.py --tg_path "$DATA/tissue_graphs/" --runs "${RUNS:-$HERE/runs_${TAG}${VAL_SPLIT}_seed*}" --all $PARG ;;
  compare)
    # PERCLASS=1: also per-class F1 of both models
    # shellcheck disable=SC2086
    python compare_runs.py --a "${A:-$HERE/runs_${FTP}${C3P//@/*}${EP}${VAL_SPLIT}_seed*}" --b "${B:-$HERE/runs_sgtfull_${FTP}${C3P//@/*}${EP}${VAL_SPLIT}_seed*}" \
                           --names ${NAMES:-CGT SGT} ${CSV:+--csv "$CSV"} $MARG $CVARG ${PERCLASS:+--perclass} ;;
  bach_patients) : "${BACH_IMAGES:?export BACH_IMAGES}"; : "${BACH_PATIENTS:?no ICIAR2018_BACH_dataset_micro_patient* next to BACH_IMAGES: export BACH_PATIENTS}"
    python bach_patients.py --patients "$BACH_PATIENTS" --images "$BACH_IMAGES" ;;
  bach_graphs) : "${BACH_IMAGES:?export BACH_IMAGES}"; : "${BACH_GRAPHS:?export BACH_GRAPHS}"
    python build_tg_bach.py --images "$BACH_IMAGES" --out "$BACH_GRAPHS" ;;
  bach_folds) : "${BACH_GRAPHS:?export BACH_GRAPHS}"; : "${BACH_OUT:?export BACH_OUT}"
    if [ -n "${BACH_PATIENTS:-}" ]; then echo "patient file: $BACH_PATIENTS"; else echo "no patient file: folds by class only"; fi
    # shellcheck disable=SC2086
    python make_bach_folds.py --src "$BACH_GRAPHS" --out "$BACH_OUT" ${BACH_PATIENTS:+--patients "$BACH_PATIENTS"} ${OVERWRITE:+--overwrite}
    for k in 0 1 2 3 4; do python build_weights.py --tg_root "$BACH_OUT/fold$k"; done ;;
  isic_graphs) : "${ISIC_IMAGES:?export ISIC_IMAGES}"; : "${ISIC_SPLIT:?export ISIC_SPLIT}"; : "${ISIC_GRAPHS:?export ISIC_GRAPHS}"
    # shellcheck disable=SC2086
    python build_tg_isic.py --images "$ISIC_IMAGES" --split "$ISIC_SPLIT" --out "$ISIC_GRAPHS" ${TRIAL:+--trial "$TRIAL"} \
                            ${SHARD:+--shard "$SHARD"} ${NSHARDS:+--nshards "$NSHARDS"} ${STAIN_NORM:+--stain_norm} ${OVERWRITE:+--overwrite} ;;
  isic_split) : "${ISIC_GRAPHS:?export ISIC_GRAPHS}"; : "${ISIC_SPLIT:?export ISIC_SPLIT}"; : "${ISIC_OUT:?export ISIC_OUT}"
    # shellcheck disable=SC2086
    python make_isic_split.py --src "$ISIC_GRAPHS" --split "$ISIC_SPLIT" --out "$ISIC_OUT" ${ALLOW_MISSING:+--allow_missing} ${OVERWRITE:+--overwrite}
    python build_weights.py --tg_root "$ISIC_OUT" ${OVERWRITE:+--overwrite} ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
