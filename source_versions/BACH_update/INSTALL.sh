#!/usr/bin/env bash
# Copies this update into CGT_sgt2. Each file is written next to the old one and renamed over it, so scripts that
# are running now (BRACS queues) keep their old copy and are not disturbed.
set -e
S2=${S2:-/workdir1.8t/fei27/CGT/GNNMD/CGT_sgt2}
U="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$U"
for f in core/*.py run_cgt_bracs.sh tools/* RERUN.md; do
  cp "$f" "$S2/$f.new" && mv -f "$S2/$f.new" "$S2/$f"; echo "installed $f"
done
cp tools/sgt_progress.sh tools/sgt_best.sh tools/sgt_perclass.py ~/ && echo "updated ~/sgt_progress.sh ~/sgt_best.sh ~/sgt_perclass.py"
