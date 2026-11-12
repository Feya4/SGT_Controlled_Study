# Final release gate (do not claim completed until checked)

- [ ] BRACS exact WSI+ROI split manifest and class counts confirmed
- [ ] BACH 5 patient-grouped folds and patient IDs verified (record missing IDs explicitly)
- [ ] ISIC 2018 lesion/near-duplicate grouping manifests confirmed
- [ ] CAMELYON16 slide-group partition manifests confirmed
- [ ] All 25 BRACS predictions + seed/run configuration files
- [ ] All 125 BACH predictions + fold and seed identifiers
- [ ] All 25 ISIC predictions and checkpoint-selection metadata
- [ ] All 25 CAMELYON predictions and checkpoint-selection metadata
- [ ] Selected best-validation model checkpoints inventoried (Zenodo for large weights)
- [ ] Environment versions, pretrained encoder provenance and available data preparation instructions recorded
- [ ] Parameter counts and all manuscript tables recomputed from archived predictions
- [ ] Checkpoint selection uses validation, not best test epoch
- [ ] No raw patient data, credentials, private server paths or copyrighted original image datasets committed
- [ ] Archive SHA256 checksums; tag git release; archive exact release on Zenodo with DOI
