#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "This release is self-contained; INSTALL.sh no longer overwrites files in a separate CGT_sgt2 directory."
echo "To create the legacy environment: conda env create -f "$HERE/environment.yml""
echo "Then run: python "$HERE/../tools/verify_release.py""
echo "See REPRODUCIBILITY.md before training."
