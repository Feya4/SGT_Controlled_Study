# SGT / CGT-derived implementation

This directory contains the controlled-study implementation for **Do Explicit Morphological and Spatial Encodings Help Graph Transformers in Breast Histopathology? A Controlled Study with SGT**.

See the [repository README](../README.md) and [reproducibility instructions](../REPRODUCIBILITY.md) for the corrected four-dataset workflow. The source builds on the connectivity-aware graph transformer (CGT) and retains its upstream license in `LICENSE`. The `INSTALL.sh` script does **not** overwrite a separate research directory.

The code expects data/graph inputs in separate paths supplied through environment variables. The original stain-normalization target image is not bundled; see `core/data/README.md`.
