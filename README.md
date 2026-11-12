# SGT: four-dataset reproducibility release — preparation package

This is a GitHub-style **staging package**, organized to match the public [Feya4/SGT](https://github.com/Feya4/SGT) conventions (`results/`, `splits/`, training code, and reproducibility utilities). It **does not yet contain actual finalized split CSVs, all 200 final-run predictions, or trained model checkpoints**. The missing files must be exported from the research server. Never present this staging package as a fully verified reproducibility release.

## Datasets and split units

| Dataset | Partition / evaluation protocol | Expected run groups |
|---|---|---|
| BRACS | WSI-independent train / validation / test; 193 / 68 / 59 WSIs and 3,155 / 602 / 626 ROIs | 5 models × 5 seeds = 25 |
| BACH | Patient-grouped five-fold CV, five paired seeds; pooled out-of-fold test predictions | 5 models × 5 folds × 5 seeds = 125 |
| ISIC 2018 | Lesion-level independent train / validation / test (near-duplicates grouped) | 5 models × 5 seeds = 25 |
| CAMELYON16-derived | Slide-group-independent split; test set 2,247 patches from 3 slide groups | 5 models × 5 seeds = 25 |

Those counts summarize the experiment design described by the researcher. Dataset split manifests must be checked against the actual final run.

## Layout

- `SGT/` — consolidated code and original updates, see `MANIFEST.json` for provenance.
- `splits/{BRACS,BACH,ISIC2018,CAMELYON16}/` — **currently placeholders; replace with actual split manifests**.
- `results/` — supplied terminal outputs, plus per-dataset guidance.
- `predictions/` — export best-validation test prediction CSVs per run.
- `checkpoints/` — optional large selected model weights; archive on Zenodo if too large for GitHub.
- `tools/export_experiment_artifacts.py` — read-only exporter for running server.
- `RELEASE_CHECKLIST.md` — publication audit.

## Export the actual artifacts from the server

Transfer `tools/export_experiment_artifacts.py` to `/workdir1.8t/fei27/CGT/GNNMD/CGT_sgt2` and execute:

```bash
python export_experiment_artifacts.py --root "$PWD" --output "$PWD/SGT_experiment_artifacts"
# Copy selected checkpoints too if practical:
python export_experiment_artifacts.py --root "$PWD" --output "$PWD/SGT_experiment_artifacts_with_weights" --include-checkpoints
```

