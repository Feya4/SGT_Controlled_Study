# Historical source versions — SGT Controlled Study

**Project:** Do Explicit Morphological and Spatial Encodings Help Graph Transformers in Breast Histopathology? A Controlled Study with SGT  
**Repository:** https://github.com/Feya4/SGT_Controlled_Study  
**Role of this directory:** historical provenance of code and experiment scripts, **not** the authoritative executable release.

## What this archive contains

| Folder | Role | Notes |
| --- | --- | --- |
| `base/CGT/` | Original CGT implementation included in the supplied source archive | Contains baseline sources, example configuration, environment and license. Review upstream licensing/attribution before reuse. |
| `bracs_update/` | BRACS-specific changes to training, prediction, comparison and launch scripts | Historical update; not a self-contained environment. |
| `bach_update/sgt2_update_bach/` | BACH training/fold utilities, patient mapping, graph construction and run scripts | Frozen five-fold assignment files are **not** included here; obtain them from repository `splits/BACH/`. |
| `isic_update/sgt2_update_isic/` | ISIC 2018 graph building, split generation and controlled-run scripts | Historical implementation; the exact evaluated lesion split belongs in `splits/ISIC2018/`. |
| `camelyon_update/` | CAMELYON16 graph construction, split utilities, launch scripts and available experiment logs | The 10 files in the separately supplied `SGT_CAMELYON16_scripts_and_logs.zip` were confirmed **byte-identical** and are retained here once, not duplicated in a second directory. |

## How this archive was reorganized

1. Retained **all 63 non-directory files** from the supplied `source_versions.zip`, preserving their original relative locations and their bytes.
2. Compared all 10 entries in `SGT_CAMELYON16_scripts_and_logs.zip` against `camelyon_update/` using SHA-256; every file matched exactly.
3. Did **not** append the redundant CAMELYON16 archive or create duplicate copies of those 10 files.
4. Added this README, `FILE_INVENTORY.csv`, `SHA256SUMS.txt` and `AUDIT_REPORT.md` for provenance and verification.
5. Kept similarly named scripts from **different historical versions** intentionally. These are not redundant release copies: their version-specific behavior may differ, and merging them would lose provenance.

## Reproducibility scope and limits

**For current reproduction, start with the repository's maintained SGT implementation and its main README.** These historical sources document how the controlled experiments evolved; they are not an independently validated, clean-install pipeline. This archive does not supply training images, precomputed tissue graphs, patient metadata spreadsheets, fixed dataset split manifests, checkpoints, seed-level predictions or a numerical rerun of the paper.

In particular, the current reported controlled experiments should use the *official verified split files* (BRACS WSI-disjoint, BACH frozen five-fold assignments, ISIC lesion-disjoint, CAMELYON16 slide-disjoint Level-2). Do **not** regenerate or replace those frozen manifests using historical split-generation scripts without verifying exact sample identifiers.

The saved experiment logs in `camelyon_update/` are historical context; they are not evidence that the package was rerun, and they must not be confused with the current five-seed 2,247-test-patch controlled protocol.

**Checkpoint selection:** prefer best **validation** weighted F1, without using test results for choosing epochs. Some original/historical scripts and logs are not guaranteed to implement that policy; refer to the maintained repository and independently audit command flags.

**Portability/privacy warning:** some unchanged historical source files and logs contain absolute Ubuntu server paths (`/workdir...`); these are *not* portable and may disclose details of the original environment. Review them before making this archive public. No secrets audit can guarantee the absence of all confidential material.

**Licensing:** The archive contains an original CGT `LICENSE`; check the terms and third-party attributions before publishing or re-licensing the derived source. This packaging process does not change source licenses.

## Audit artifacts

- `FILE_INVENTORY.csv`: one row per original source/log file, relative path, byte size, SHA-256, category, and original archive.
- `SHA256SUMS.txt`: per-file hashes in the packaged directory, to verify source integrity after download.
- `AUDIT_REPORT.md`: archive comparison, static checks and known limitations.

Example verification from the root of the extracted repository:

```bash
cd source_versions
sha256sum -c SHA256SUMS.txt
```

This should verify the **63 original files**. There is no claim here that model training, inference or data reconstruction has been executed.
