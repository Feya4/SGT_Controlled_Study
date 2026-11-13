#!/usr/bin/env bash
# Lightweight manual training trigger for one CAMELYON model/seed (no automatic 25-run queue).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CAM_OUT:?Set CAM_OUT to prepared CAMELYON graph root containing tissue_graphs/{train,val,test}}"
DATASET=camelyon MODEL="${MODEL:-sgt}" EPOCHS="${EPOCHS:-100}" SEEDS="${SEEDS:-0}" GPU="${GPU:-0}" bash run_study.sh train
