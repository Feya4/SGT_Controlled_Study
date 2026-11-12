# Do Explicit Morphological and Spatial Encodings Help Graph Transformers in Breast Histopathology? A Controlled Study with SGT

**Feidu Akmel · Xun Gong · Ashraf Osman Ibrahim · Gedamu Alemu · Abdalla Hadabi · Abdenacer Naouri**

*Corresponding author: Xun Gong.*  

[![Study](https://img.shields.io/badge/Study-Controlled%20Graph%20Transformer%20Evaluation-blue)](#study-overview) [![Datasets](https://img.shields.io/badge/Datasets-4-blue)](#datasets-and-evaluation-protocols) [![Reproducibility](https://img.shields.io/badge/Checkpoint%20selection-Validation%20weighted%20F1-green)](#checkpoint-selection-and-evaluation-integrity)

## Study overview

This repository accompanies **“Do Explicit Morphological and Spatial Encodings Help Graph Transformers in Breast Histopathology? A Controlled Study with SGT.”** It documents the implementation, controlled comparisons, data-partition protocols, and reproducibility materials used to investigate whether explicit **morphological** and **spatial** representations improve graph-transformer-based histopathology classification beyond a comparable baseline.

The study examines the **Structural-Morphology-Aware Graph Transformer (SGT)**, combining a morphology-aware structural encoding (**MASE**) with a spatial relationship encoding (**SRE**). The central research question is *not* merely whether a more complex model achieves a higher score: it is whether any observed difference can be attributed to informative morphological and spatial encodings rather than added parameters, input dimensionality, or the training and evaluation procedure.


> **Release scope:** The research-server experiment inventory comprises **200 primary model × seed × fold runs** across four datasets. The lightweight checkpoint package prepared for sharing contains **four representative validation-selected checkpoints (one per dataset)** and the **five recovered split files**. These representative weights are **not** a substitute for the complete 200-run evaluation or retraining all seeds and folds. 

## Study design and controlled comparisons

| Variant | Purpose | Controlled question |
|---|---|---|
| **Baseline (CGT)** | Graph-transformer reference without the additional SGT modules | How does the underlying graph-transformer formulation perform? |
| **SGT (MASE + SRE)** | Full proposed model | Do explicit morphology and spatial encodings improve performance? |
| **Concatenation control** | Incorporates morphological attributes through a parameter-comparable concatenation-based pathway | Is the effect attributable to the structured encoder or simply to extra input features / capacity? |
| **Morphology-permuted control** | Disrupts the correspondence between morphological attributes and graph nodes | Does meaningful morphology–node alignment matter? |
| **Spatial/structure-permuted control** | Disrupts the spatial/structural information supplied to the relationship encoder | Does meaningful spatial organization matter? |

In the experimental code, names may appear as `cgt`, `sgtfull`, `sgtconcat`, `sgtmshuf`, and `sgtpshuf`. Refer to the dataset-specific launcher or run metadata for the exact configuration used in each reported comparison. **The perturbation controls are not interchangeable with removing the corresponding modules.**

## Datasets and evaluation protocols

| Dataset | Task and split unit | Verified protocol | Primary runs |
|---|---|---|---:|
| **BRACS** | Breast histopathology; whole-slide-image (WSI)-disjoint partitions | **3,155 train / 602 validation / 626 test ROIs** from **193 / 68 / 59 WSIs**, respectively. The controlled `c3` runs use a three-class grouping of the original diagnostic categories. | 25 |
| **BACH (ICIAR 2018, Part A)** | Four-class breast histology; five-fold cross-validation | **400 images**; test fold sizes **83, 65, 73, 94, 85**. Each image appears in one test fold per seed. | 125 |
| **ISIC 2018** | Skin-lesion image classification; lesion/group-disjoint split | **7,047 train / 1,468 validation / 1,500 test images**, with no lesion/group overlap across partitions. | 25 |
| **CAMELYON16-derived patch graphs** | Metastasis-related patch classification; slide-disjoint split | **7,956 train / 1,821 validation / 2,247 test patch graphs**; the test set comes from **3 slide groups**. The verified experiment uses **Level-2** graphs. | 25 |
| **Total** | | Five model variants; five random seeds, plus five BACH folds | **200** |

**Scope clarification:** BRACS and BACH are the breast-histopathology datasets central to the paper's question. ISIC 2018 and CAMELYON16-derived experiments provide additional controlled evaluations across different image/graph settings; they should not be characterized as proof of cross-domain generalization unless supported by a matched transfer experiment.

### BACH patient-metadata limitation

The recovered `folds_info.json` records `grouped_by_patient: true`, **145 images with identified patients**, **29 identified patients**, and **zero identified patients assigned to multiple folds**. Fold-to-prediction comparison also confirmed 400 distinct test images and no duplicate test membership. However, patient identifiers are **not verified for the other 255 images**. Therefore, **patient-level independence for all 400 images has not been established**. This limitation must be retained in the manuscript and any interpretation of the BACH evaluation.

## Repository layout

The following is the **intended public release layout**. Some items are optional or may reside in an external archive; do not assume that all paths exist in a newly created repository.

```text
SGT_Controlled_Study/
├── README.md
├── SGT/                       # Graph-building, training, evaluation code
│   ├── core/
│   ├── tools/
│   └── environment.yml
├── splits/                    # Actual, verified experimental partitions
│   ├── BRACS/bracs_split.csv
│   ├── BACH/folds.json
│   ├── BACH/folds_info.json
│   ├── ISIC2018/isic2018_lesion_split.csv
│   └── CAMELYON16/split.csv
├── results/                   # Dataset-level summaries / run metadata, if released
├── predictions/               # Optional per-run prediction CSVs
└── checkpoints/               # Optional representative weights or archive links
```

**Important:** An earlier GitHub-style staging ZIP contained *placeholder* split directories. It is not an authoritative source of split membership. Replace those placeholders with the five actual split files from the research server's `SGT_final_4_checkpoints/splits/` before publishing. Keep the candidate/inferred split manifests separate from the verified files.

## Reproducing the evaluation

### 1. Obtain code and dependencies

```bash
# After this repository has been created and populated:
git clone https://github.com/Feya4/SGT_Controlled_Study.git
cd SGT_Controlled_Study

# If the SGT source tree and environment file have been included:
conda env create -f SGT/environment.yml
conda activate cgt
```

The environment configuration in the supplied source archive targets a **legacy Python/PyTorch/DGL software stack**. Installation may require compatible CUDA drivers, platform-specific adjustments, and access to original graph-generation dependencies. Record all deviations from the archived environment. Do not silently replace dependency versions when claiming an exact rerun.

### 2. Obtain the datasets and graph inputs

Acquire each dataset from its authorized provider and observe its access, citation, privacy, and redistribution terms. Prepare graph inputs using the corresponding preprocessing code and configuration. **Raw medical images and patient metadata are not included in this repository unless their redistribution is explicitly permitted.**

Preserve original image/ROI identifiers during graph construction. Match the generated graph IDs to the split manifest *before* training. The split files encode the **experiment actually evaluated**; regenerating a new split, even with the same seed, is not equivalent to reproducing the original partitions.

### 3. Apply the original partitions

- **BRACS:** `splits/BRACS/bracs_split.csv` uses ROI and slide identifiers, class labels, and `train`/`val`/`test` assignments. Ensure no slide is shared across partitions.
- **BACH:** `splits/BACH/folds.json` maps image identifiers to folds 0–4. For each test fold *k*, the experimental protocol uses the next fold as validation and the remaining folds for training. Retain the patient-metadata qualification above.
- **ISIC 2018:** `splits/ISIC2018/isic2018_lesion_split.csv` includes image/lesion grouping and partition assignments. Keep all images in a lesion group within the same partition.
- **CAMELYON16:** `splits/CAMELYON16/split.csv` identifies Level-2 patch graphs and the slide-group-separated train, validation, and test sets. **Do not substitute an older 14,609-patch candidate split** for the experimentally verified **12,024-patch** manifest.

### 4. Select and evaluate checkpoints correctly

The checkpoint rule for the controlled analysis is **best validation weighted F1**, represented in the run directories by:

```text
model_best_val_weighted_f1_score.pt
```

Select the checkpoint using **validation data only**, then evaluate it on the held-out test data. Never select an epoch by the highest test F1 and subsequently report that test F1 as an unbiased estimate. Any test-selected checkpoint from exploratory analyses must be identified separately and excluded from the primary validation-selected comparison.

The representative four-checkpoint package uses the following deterministic choices:

| Dataset | Representative source run | Saved name in compact package |
|---|---|---|
| BRACS | `runs_sgtfull_c3_e100_val_seed0` | `checkpoints/BRACS/best_validation.pt` |
| BACH | `runs_sgtfull_bach_f0_e100_val_seed0` | `checkpoints/BACH/best_validation.pt` |
| ISIC 2018 | `runs_sgtfull_isic_e100_val_seed0` | `checkpoints/ISIC2018/best_validation.pt` |
| CAMELYON16 | `runs_sgtfull_cam_e100_val_seed0` | `checkpoints/CAMELYON16/best_validation.pt` |

These are **representative seed-0 checkpoints**, with **BACH fold 0** selected as a representative fold; they were **not chosen as the globally highest-scoring seeds**. Loading or evaluating a checkpoint requires the exact graph construction, labels, model flags, and architecture that produced it. A single BACH fold-0 checkpoint does **not** recreate the pooled five-fold evaluation.

### 5. Reproduce the multi-run comparisons

The primary controlled inventory has five variants × five seeds for BRACS, ISIC 2018, and CAMELYON16, and five variants × five folds × five seeds for BACH: **200 runs** in total. The exported research-server archive additionally detected **10 separate BRACS 200-epoch runs** (`sgtfull` and `sgtconcat`, seeds 0–4); those are **supplementary experiments**, not part of this 200-run inventory. Keep their training budgets and selection protocols distinct.

For exact reruns, use the dataset-specific scripts under `SGT/` and the configuration recorded in each original run. The source archive includes launchers and supporting files such as `run_cgt_bracs.sh`, `tools/run_bach.sh`, `tools/run_isic.sh`, and `tools/run_camelyon.sh`; consult their help/comments and configure dataset paths before executing. **Commands should not be presented as one-click turnkey reproduction until tested on a clean environment.**

Report seed-level observations and, when comparing variants, **paired seed/fold differences** with uncertainty intervals. The test set is used for final evaluation, not for selecting models, tuning preprocessing, or deciding which experiments to report. For CAMELYON16, image/patch-level uncertainty must not be confused with independent-slide uncertainty because the test set contains only three slide groups.

## Verification record

The following checks were performed on the research-server artifacts during preparation of this release:

- **BRACS:** 25 primary prediction files matched the test identifiers of the recovered WSI-disjoint split; no missing, extra, or duplicate IDs were reported.
- **BACH:** `folds.json` mapped 400 images to five non-overlapping test folds; all five **SGT seed-0** test-prediction files matched the assignments exactly. The additional BACH run files were inventoried, but a fold-by-fold identity check across **all 125 runs** is not documented here.
- **ISIC 2018:** 25 primary prediction files matched the recovered lesion-disjoint test split; no missing, extra, or duplicate IDs were reported.
- **CAMELYON16:** 25 primary prediction files matched all **2,247** test graph IDs in the recovered slide-disjoint split; no missing, extra, duplicate, or ground-truth label inconsistencies were reported.
- **Checkpoints:** 200 primary best-validation checkpoint files were collected into an intermediate **6.4 GB** archive; four selected representative checkpoints plus five split manifests were copied into a separate **131 MB** compact package.

These are **artifact-consistency checks**, not independent verification that every result can be reproduced bit-for-bit from a clean installation. The original training commits, exact selected epochs, data provenance, preprocessing, and final summary statistics require ongoing archival auditing.

## Interpretation and limitations

This work is designed to isolate the effect of morphology and spatial encoding from confounding changes in model capacity and evaluation. A favorable point estimate alone does not establish that a module is useful: interpretation should consider paired uncertainty, variation across seeds and datasets, and the effects of the corresponding permutation and parameter-matched controls. In particular, **non-significant comparisons should not be described as established improvements**.

Further limitations include the incomplete BACH patient-identity mapping, the limited number of independent CAMELYON16 test-slide groups, differences between datasets and classification settings, and the fact that a representative-checkpoint archive cannot substitute for all five seeds or all BACH folds. Statements of generalization beyond the evaluated data and protocols should be appropriately qualified.

## Citation

If you use this code, please cite the associated study once its final bibliographic record is available:

> Akmel, F., Gong, X., Ibrahim, A. O., Alemu, G., Hadabi, A., & Naouri, A. **Do Explicit Morphological and Spatial Encodings Help Graph Transformers in Breast Histopathology? A Controlled Study with SGT.** *Manuscript; journal, year, DOI and bibliographic metadata to be added after publication.*

The original SGT project is available at **https://github.com/Feya4/SGT**. Cite source datasets and upstream software independently as required by their licenses and documentation.

## Availability, licensing, and contact

- **Original SGT repository:** https://github.com/Feya4/SGT.
- **Reproducibility materials:** Source code, split manifests, run summaries, and representative checkpoints should be identified by versioned release assets or permanent archive links when published. Do not interpret unpopulated paths as already available downloads.
- **License:** Refer to the actual `LICENSE` file in the published repository; source-code licenses do **not** automatically grant permission to redistribute medical images.
- **Correspondence:** Xun Gong (contact details to be added after confirmation).

---

*Reproducibility principle: preserve the evaluated partitions, choose checkpoints by validation performance, report all prespecified controlled comparisons, and distinguish verified artifacts from still-unreleased materials.*
