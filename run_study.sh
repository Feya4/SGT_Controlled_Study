#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 1 ]; then echo "Usage: DATASET=bracs|bach|isic|camelyon MODEL=sgt|cgt|sgt_concat|sgt_mshuf|sgt_pshuf EPOCHS=100 bash run_study.sh check|smoke|train|predict|compare"; exit 2; fi
cd "$ROOT/SGT"
exec bash run_cgt_bracs.sh "$@"
