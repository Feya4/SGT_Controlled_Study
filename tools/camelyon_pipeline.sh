#!/usr/bin/env bash
# Portable CAMELYON Level-2 workflow; does NOT start a multi-GPU training scheduler.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CAM_ROOT:?Set CAM_ROOT to prepared Level-2 .npy patch directory}"
CAM_SPLIT="${CAM_SPLIT:-$ROOT/splits/CAMELYON16/split.csv}"
: "${CAM_GRAPHS:?Set CAM_GRAPHS to output path for raw graph files}"
: "${CAM_OUT:?Set CAM_OUT to output path for converted graph files}"
export CAM_ROOT CAM_SPLIT CAM_GRAPHS CAM_OUT
cd "$ROOT"
python tools/verify_release.py
DATASET=camelyon bash run_study.sh cam_graphs
DATASET=camelyon bash run_study.sh cam_split
DATASET=camelyon MODEL=sgt EPOCHS="${EPOCHS:-100}" SEEDS="${SEEDS:-0}" GPU="${GPU:-0}" bash run_study.sh smoke
printf '%s\n' 'CAMELYON preprocessing and smoke checks complete. Training is NOT started automatically.'
printf '%s\n' 'To train: DATASET=camelyon MODEL=sgt EPOCHS=100 SEEDS=0 bash run_study.sh train'
