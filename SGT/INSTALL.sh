#!/usr/bin/env bash
# Install the ISIC additions into CGT_sgt2 by copy + rename (never rewritten in place, so running scripts are safe).
# The files it replaces are kept in CGT_sgt2/_backup_before_isic_<date>/.
#   bash sgt2_update_isic/INSTALL.sh
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${DST:-/workdir1.8t/fei27/CGT/GNNMD/CGT_sgt2}"
FILES="run_cgt_bracs.sh tools/run_isic.sh core/build_tg_isic.py core/make_isic_split.py core/train.py
       core/predict_runs.py core/compare_runs.py RERUN_ISIC.md"
[ -d "$DST/core" ] || { echo "no CGT_sgt2 code in $DST"; exit 1; }
[ -f "$DST/core/build_tg.py" ] && [ -f "$DST/core/build_tg_bach.py" ] || { echo "$DST lacks build_tg.py / build_tg_bach.py"; exit 1; }
for f in $FILES; do [ -f "$PKG/$f" ] || { echo "missing $PKG/$f"; exit 1; }; done
BK="$DST/_backup_before_isic_$(date +%m%d%H%M)"
mkdir -p "$BK"
for f in $FILES; do
  if [ -f "$DST/$f" ]; then mkdir -p "$BK/$(dirname "$f")"; cp -p "$DST/$f" "$BK/$f"; fi
  mkdir -p "$DST/$(dirname "$f")"
  cp "$PKG/$f" "$DST/$f.new" && mv -f "$DST/$f.new" "$DST/$f"
  echo "installed $f"
done
chmod +x "$DST/run_cgt_bracs.sh" "$DST/tools/run_isic.sh"
echo "replaced files kept in $BK"
