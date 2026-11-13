# Reproducing the SGT controlled study

This guide is for the **corrected** source snapshot packaged for GitHub. It distinguishes verified metadata, an executable training workflow, and data/assets still required for an independent numerical reproduction.

## 1. Verify frozen manifests

Run `python tools/verify_release.py` from the repository root (Python >= 3.7). This checks the counts and group separation of the five frozen split files. These CSV/JSON files are **the original recovered assignments**, not regenerated from a new random seed. `splits/BACH/folds_info.json` contains an anonymized/portable description of the organizer's patient-data file path; it does not redistribute patient metadata.

When original experiment directories are accessible, run:

```bash
python tools/verify_release.py --runs /path/to/CGT_sgt2
```

Expected original primary runs: **25 BRACS + 125 BACH + 25 ISIC 2018 + 25 CAMELYON16 = 200**. The optional check compares test graph/image IDs to frozen test assignments; it does not reload model weights or measure scores. Run directories named `_e200_val_seed*` for BRACS are **supplementary experiments**, not the primary 100-epoch inventory.

## 2. Original software environment

Create the original conda environment where compatible:

```bash
conda env create -f SGT/environment.yml
conda activate cgt
bash run_study.sh check
```

`environment.yml` is a legacy CUDA-linked specification (Python 3.7, PyTorch 1.9.1+cu111, torchvision 0.10.1+cu111, DGL 0.4.3.post2, histocartography 0.2.0). Availability and compatibility on modern drivers are **not** established by this package. `SGT/core/wrapper.py` compiles `algos.pyx` at runtime; a working C compiler and Cython are required. The static audit does not establish a functional environment.

## 3. Acquire research inputs

Raw BRACS/BACH/ISIC/CAMELYON16 images are not included. Follow each dataset's terms and the manuscript's data-access statement. Keep original sample identifiers unchanged. Never derive a new data split from a seed when the frozen split already exists.

The breast/tissue-graph builder requires the **exact original Vahadane reference image** at `SGT/core/data/target.png`, or a path passed via `SGT_STAIN_TARGET`:

```bash
export SGT_STAIN_TARGET=/path/to/authorized/original/target.png
```

The project owner's original `CGT_sgt2.zip` contains `CGT/core/data/target.png`. **This package does not redistribute that image** pending a rights check. A different reference image changes graph construction and invalidates claims of exact preprocessing reproduction. ISIC can be built without stain normalization (the default), in which case the target is not needed.

## 4. Dataset-specific graph processing

The scripts below assume that authorized images and their fixed graph identifiers have been prepared. Supply full paths for your own system.

### BRACS

BRACS input: HACT/region-graph sources for the official WSI-separated 3,155/602/626 ROI split. Do not use earlier nonofficial graphs that overlapped partitions.

```bash
export GRAPHS=/data/BRACS/hact_source_graphs
export OUT=/data/BRACS/converted_cgt
export OUT_MORPH=/data/BRACS/converted_cgt_morph
bash run_study.sh convert
bash run_study.sh weights
bash run_study.sh convert_morph
bash run_study.sh weights_morph
DATASET=bracs CLASSES=3 MODEL=sgt EPOCHS=100 SEEDS=0 GPU=0 bash run_study.sh train
```

BRACS source graph construction in `SGT/core/build_tg.py` depends on stain-normalization reference and the original HACT pipeline. The package does **not** supply the HACT source graphs or a one-command image-to-HACT pipeline. Generating equivalent graphs from raw pixels remains a prerequisite.

### BACH

```bash
export BACH_IMAGES=/data/BACH/images
export BACH_PATIENTS=/data/BACH/ICIAR2018_BACH_dataset_micro_patient.xlsx
export BACH_GRAPHS=/data/BACH/graphs_raw
export BACH_OUT=/data/BACH/converted_cv
DATASET=bach bash run_study.sh bach_graphs
DATASET=bach bash run_study.sh bach_folds
DATASET=bach MODEL=sgt FOLD=0 EPOCHS=100 SEEDS=0 GPU=0 bash run_study.sh train
```

**Exact paper fold recovery:** The corrected `bach_folds` launcher uses `splits/BACH/folds.json` by default via `--folds_json`; it rejects missing or extra source image IDs. To generate a **different, non-paper** assignment, explicitly use `REGENERATE_BACH=1`. If patient metadata are supplied, the frozen folds are also checked for identified-patient overlap. Always verify the converted `BACH_OUT/fold*/tissue_graphs` IDs before a full training run. The 29 identified patients spanning 145 images are fold-disjoint; patient separation for 255 images remains unknown.

### ISIC 2018

```bash
export ISIC_IMAGES=/data/ISIC2018/images
export ISIC_SPLIT="$PWD/splits/ISIC2018/isic2018_lesion_split.csv"
export ISIC_GRAPHS=/data/ISIC2018/graphs_raw
export ISIC_OUT=/data/ISIC2018/converted
DATASET=isic bash run_study.sh isic_graphs
DATASET=isic bash run_study.sh isic_split
DATASET=isic MODEL=sgt EPOCHS=100 SEEDS=0 GPU=0 bash run_study.sh train
```

Keep lesion groups separated as in the frozen file. ISIC is a dermoscopy dataset; default graph construction does not apply the H&E stain-normalization target.

### CAMELYON16-derived patches

```bash
export CAM_ROOT=/data/CAMELYON16/prepared_level2_patch_npy
export CAM_SPLIT="$PWD/splits/CAMELYON16/split.csv"
export CAM_GRAPHS=/data/CAMELYON16/graphs_raw
export CAM_OUT=/data/CAMELYON16/converted
DATASET=camelyon bash run_study.sh cam_graphs
DATASET=camelyon bash run_study.sh cam_split
DATASET=camelyon MODEL=sgt EPOCHS=100 SEEDS=0 GPU=0 bash run_study.sh train
```

`CAM_ROOT` must contain prepared `.npy` dictionaries with the original `sample_id` names, not raw WSIs. The corrected launcher supports `cam_graphs`, `cam_split`, and the two-class `cam2` task. All 12,024 graphs are Level 2, and the held-out 2,247 patches belong to 3 slide groups. The original slide-to-patch extraction procedure and source WSIs are not deposited here, so end-to-end WSI-to-patch reproduction needs additional authorized data/preprocessing details.

## 5. Controlled training and evaluation

Five model presets: `MODEL=cgt`, `sgt`, `sgt_concat`, `sgt_mshuf`, `sgt_pshuf`. Default `SEEDS="0 1 2 3 4"`; BACH additionally requires `FOLD="0 1 2 3 4"` for the complete CV. Use `EPOCHS=100` for primary controlled experiments. Each model should use the same converted graphs, seeds, fold assignments, training recipe and selected validation rule.

```bash
DATASET=isic MODEL=sgt EPOCHS=100 SEEDS='0 1 2 3 4' bash run_study.sh train
DATASET=isic MODEL=sgt EPOCHS=100 bash run_study.sh predict
DATASET=isic EPOCHS=100 bash run_study.sh compare
```

Primary model weights: `model_best_val_weighted_f1_score.pt`; select the epoch using **validation** weighted F1, then test on the held-out split. The corrected source defaults to `val`. Historical diagnostic `VAL_SPLIT=test` or `TESTSEL=1` use requires explicit `ALLOW_TEST_SELECTION=1` and those results must be labelled **optimistic** and excluded from primary comparisons.

The helper scripts under `tools/` include historical schedulers with additional assumptions. `tools/run_bach.sh`, `tools/run_isic.sh`, and `tools/run_camelyon.sh` now locate the source relative to their own location and do not enable per-epoch test evaluation. Older analysis and orchestration helpers should not be used as the canonical clean setup.

## 6. Checkpoint distribution

A compact four-checkpoint package was assembled independently on the author's Ubuntu server (SGT full, seed 0 for BRACS/ISIC/CAMELYON16, and BACH fold 0 seed 0). It is **not included in this ZIP**. Do not cite a GitHub or Zenodo weight URL until the corresponding files are uploaded and checksummed. Four representative weights do not reproduce five-seed tables; full statistical reproduction requires full reruns or all per-seed predictions/configuration metadata.

## 7. What has / has not been verified

**Checked on the corrected source archive:** Python syntax for 23 modules, shell syntax for launchers, fixed split sizes and group leakage, CAMELYON16 task wiring, presence of the original CGT config file, and no implicit test-selected checkpoint rule.

**Not verified here:** end-to-end code execution on the research GPU stack, graph equivalence, recomputed five-seed metrics, original best epochs, equivalence of all 200 checkpoints to their predictions, and availability of dataset/provider-specific preprocessing assets. Complete these checks on the research server before freezing the Zenodo DOI.
