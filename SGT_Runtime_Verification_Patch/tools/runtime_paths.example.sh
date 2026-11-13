#!/usr/bin/env bash
# Copy to tools/runtime_paths.local.sh, edit for your server, then run:
#   source tools/runtime_paths.local.sh
# This file is an EXAMPLE: no data locations are guessed.
# The source paths may be outside the GitHub checkout.

# BRACS: already converted graph data for the baseline and SGT variants.
export OUT="/path/to/bracs_cgt_dataset"
export OUT_MORPH="/path/to/bracs_sgt_dataset"

# BACH: five folds, each with tissue_graphs/{train,val,test}.
export BACH_OUT="/path/to/bach_cv"

# ISIC 2018: lesion-disjoint dataset prepared for the trainer.
export ISIC_OUT="/path/to/isic_dataset"

# CAMELYON16: Level-2, slide-disjoint dataset prepared for the trainer.
export CAM_OUT="/path/to/camelyon_dataset"

# GPU index if CUDA is available.
export GPU=0
