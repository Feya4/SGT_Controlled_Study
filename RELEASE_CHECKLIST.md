# Pre-Zenodo release checklist (must be completed by project owner)

- [x] Recover five authentic split manifests for BRACS, BACH, ISIC 2018, CAMELYON16.
- [x] Check split counts and no overlap for slide/lesion groups; BACH 400 unique fold assignments.
- [x] Restore missing original CGT config YAML (from user-provided original CGT archive).
- [x] Integrate `cam2` in training, prediction, comparisons and main launcher.
- [x] Default to validation-only selection and block accidental test-selected training.
- [x] Make main launchers relative to source root; remove obsolete placeholder files.
- [x] Provide `tools/verify_release.py` static/metadata audit (not a GPU training test).
- [ ] Acquire the original stain-normalization target locally with permission and verify its hash.
- [ ] Confirm all environment dependencies install on a clean research machine; compile and import the Cython extension.
- [ ] On the research server, verify graphs against frozen splits and run a smoke test for each dataset/variant.
- [ ] Re-evaluate at least one representative validation-selected checkpoint per dataset and confirm recorded predictions.
- [ ] Verify the 200 original run configurations/checkpoint provenance and aggregate source tables (or disclose any exceptions).
- [ ] Upload original prediction CSVs/configurations/logs as permitted to the archival record.
- [ ] Upload and checksum the four representative SGT weights (if permitted); confirm the archive download link.
- [ ] Confirm licenses and redistribution rights for CGT-derived source, normalization target and data derivatives.
- [ ] Freeze GitHub release and Zenodo version, then insert the actual DOI into the manuscript and response letter.
