# Do Explicit Morphological and Spatial Encodings Help Graph Transformers in Breast Histopathology? A Controlled Study with SGT

**Feidu Akmel¹˒², Xun Gong¹˒²*, Ashraf Osman Ibrahim³, Gedamu Alemu⁴, Abdalla Hadabi², Abdenacer Naouri²**  
*Corresponding author: Xun Gong · xgong@swjtu.edu.cn*

¹ Yibin Research Institute, Southwest Jiaotong University, China  
² School of Computing and Artificial Intelligence, Southwest Jiaotong University, China  
³ Universiti Teknologi PETRONAS, Malaysia  
⁴ Khalifa University of Science and Technology, United Arab Emirates

## About this repository

This repository is the **controlled-study implementation** accompanying the manuscript listed above. It evaluates whether adding explicit tissue-region morphology (**MASE**) and graph-hop-distance relationships (**SRE**) to the connectivity-aware graph transformer (**CGT**) provides a measurable classification benefit under matched experimental protocols.

**Research interpretation:** On the BRACS three-class task the tested encodings yielded no demonstrated performance benefit relative to the controlled baselines. On BACH, the full SGT scored below CGT and the morphology-permutation control. These results are not presented as evidence of superiority for SGT.

The public source release supports **BRACS, BACH, ISIC 2018, and CAMELYON16-derived Level-2 patch graphs**. Experimental graphs and raw medical images are **not** included; they must be obtained from their authorized sources or reconstructed under the stated preprocessing protocol. The current release contains **no actual `.pt` checkpoint weights or the 200 original saved prediction files**. The author prepared a separate four-model representative-checkpoint package on a research server, but it must be uploaded and verified before it can be cited as an available download.

## Quick start — verify without installing ML dependencies

```bash
git clone https://github.com/Feya4/SGT_Controlled_Study.git
cd SGT_Controlled_Study
python tools/verify_release.py
```

The stdlib-only audit checks manifest counts, slide/lesion grouping, CAMELYON16 magnification, BACH folds, Python source syntax, and presence of the CAMELYON16 training integration. It **does not** train models, test GPU compatibility, verify medical-image preprocessing outputs, or prove that numerical paper results have been reproduced. To recheck sample identifiers from the author's original run directories (when available):

```bash
python tools/verify_release.py --runs /path/to/original/CGT_sgt2
```

The audit checks test-prediction identifiers, not model weights, checkpoint epoch provenance, or training determinism.

## Controlled model variants

| Name | `MODEL` | Description |
|---|---|---|
| Connectivity-aware graph transformer | `cgt` | Baseline without additional morphology/hop encodings |
| Full SGT | `sgt` | CGT + MASE + SRE |
| Parameter-matched concatenation | `sgt_concat` | Raw morphology through an alternative input path, SRE retained |
| Randomized morphology | `sgt_mshuf` | Node-morphology correspondence permuted within each graph |
| Randomized structure | `sgt_pshuf` | SRE-specific hop information randomized while the base connectivity remains intact |

Experiments use seeds **0, 1, 2, 3, 4** with the shared graph-input, model-training and validation-selection protocol described in `REPRODUCIBILITY.md`. The 200 primary runs comprise 25 BRACS, 125 BACH, 25 ISIC 2018, and 25 CAMELYON16 model × seed × fold runs.

## Verified frozen data splits

| Dataset | Frozen manifest | Train | Validation | Test | Grouping rule |
|---|---|---:|---:|---:|---|
| BRACS | `splits/BRACS/bracs_split.csv` | 3,155 ROIs | 602 ROIs | 626 ROIs | WSI-disjoint |
| BACH | `splits/BACH/folds.json` | 5-fold CV | next fold | held-out fold | 400 unique images; 145 images with known patient IDs |
| ISIC 2018 | `splits/ISIC2018/isic2018_lesion_split.csv` | 7,047 | 1,468 | 1,500 | lesion/group-disjoint |
| CAMELYON16 | `splits/CAMELYON16/split.csv` | 7,956 | 1,821 | 2,247 | slide-group-disjoint, all Level 2 |

**BACH qualification:** The organizer patient metadata identify 29 patients associated with 145 images; none of these identified patients is distributed across folds. The remaining **255 images do not have verified patient identifiers**, so patient-level independence for the entire dataset cannot be asserted. Fold sizes: 83, 65, 73, 94 and 85.

For CAMELYON16, **do not use an older, different 14,609-patch manifest**. The controlled Level-2 experiment is based on the **12,024-patch** split above.

## Repository layout

```text
SGT_Controlled_Study/
├── README.md
├── REPRODUCIBILITY.md          # Dataset-specific setup and commands
├── tools/verify_release.py     # Standard-library preflight and optional prediction-ID audit
├── run_study.sh                # Entry point for the core training/evaluation launcher
├── SGT/
│   ├── core/                   # Graph processing, model, training, inference, tests
│   │   ├── config/bracs_grapher_7_classes_pna.yml
│   │   └── data/README.md      # Required reference image: NOT redistributed
│   ├── run_cgt_bracs.sh
│   └── environment.yml
├── tools/                      # Schedulers and helper scripts
├── splits/                     # Five frozen experimental split files
├── results/                    # Preserved comparison-terminal outputs (not raw predictions)
├── checkpoints/                # Documentation / legacy supplemental scripts; no weights
└── source_versions/            # Archived provenance source bundle
```

## Installation and example execution

The archived environment, `SGT/environment.yml`, specifies a **legacy Python 3.7 / PyTorch 1.9 / DGL 0.4 software stack**, with CUDA-specific wheels. Modern systems may need a compatible container or manual installation. Do not claim exact reproducibility after substituting dependency versions without validating model outputs.

```bash
conda env create -f SGT/environment.yml
conda activate cgt
DATASET=bracs bash run_study.sh check
```

Once your authorized graphs have been prepared and converted to the expected DGL input structure:

```bash
# Example: three-class BRACS; supply actual graph roots as required by the launcher
export OUT_MORPH=/path/to/converted/bracs_graphs_with_morphology
export OUT=/path/to/converted/bracs_graphs
DATASET=bracs CLASSES=3 MODEL=sgt EPOCHS=100 SEEDS=0 GPU=0 \
  bash run_study.sh train

# The five-fold BACH setup similarly requires BACH_OUT/fold0 ... fold4.
# ISIC 2018 requires ISIC_OUT; CAMELYON16 requires CAM_OUT.
```

The training launcher now defaults to **checkpoint selection on validation weighted F1**, and performs held-out test evaluation after training rather than at every epoch. Historical test-selected runs require an **explicit separate diagnostic override** and must not be reported as unbiased primary results.

See **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** for required input directories, all four datasets' conversion commands, stain-normalization source, model presets, and release limitations.

### Model checkpoints

Due to repository file-size constraints, pretrained model checkpoints are distributed separately through Zenodo. The checkpoint archive contains four representative SGT models, one each for BRACS, BACH (fold 0), ISIC 2018, and CAMELYON16. All checkpoints were selected using validation weighted F1 rather than test-set performance.

The complete training configurations and experimental protocols are documented in this repository. The four representative checkpoints do not constitute the full set of models used in the five-seed controlled study.



## Transparency and limitations

- The frozen partitions and controlled variant source files are provided, but original datasets, all preprocessed DGL graphs and the original stain-normalization target image are **not redistributed**.
- The required target image is present in the owner's original `CGT_sgt2.zip` under `CGT/core/data/target.png`. Its redistribution rights must be confirmed; if authorized, restore the exact image locally or set `SGT_STAIN_TARGET` as documented.
- Five-seed quantitative results must be regenerated or supported by deposited per-run outputs; the included comparison-terminal logs do not independently establish a full clean rerun.
- BACH's unidentified patients and CAMELYON16's three slide groups restrict clinical and group-level interpretation.
- This project is **research software**, not a validated clinical diagnostic product.

## Citation and archive status

Please cite the manuscript title and all authors above. **Do not invent a journal volume, DOI, publication status or Zenodo accession.** A versioned Zenodo record and citation metadata can be added after the software audit and deposition are complete.

The implementation incorporates and modifies CGT and associated graph-processing software; see the source provenance and upstream notices before selecting or modifying licensing terms.
