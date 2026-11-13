# Historical patch utility. CAMELYON16 changes have ALREADY BEEN integrated in this released source.
# Do not rerun this against the release; it is retained only for provenance.
"""[sgt] add the CAMELYON16-derived task (cam2, 2 classes) to train.py, predict_runs.py, compare_runs.py, run_cgt_bracs.sh.
All four files are patched or none; each is written as <file>.new and then renamed (running jobs keep the old file).
Running it again is harmless. Usage, from CGT_sgt2:  python patch_camelyon.py
"""
import os
import sys

E = {
 "core/train.py": [
  ("'bach4': None, 'isic7': None}", "'bach4': None, 'isic7': None, 'cam2': None}"),
  ("'bach4': 4, 'isic7': 7}", "'bach4': 4, 'isic7': 7, 'cam2': 2}"),
  ("choices=['bracs7', 'bracs3', 'bach4', 'isic7'],", "choices=['bracs7', 'bracs3', 'bach4', 'isic7', 'cam2'],"),
 ],
 "core/predict_runs.py": [
  ("{'bracs7': 7, 'bracs3': 3, 'bach4': 4, 'isic7': 7}[args.task]", "{'bracs7': 7, 'bracs3': 3, 'bach4': 4, 'isic7': 7, 'cam2': 2}[args.task]"),
  ("choices=['bracs7', 'bracs3', 'bach4', 'isic7'], default='bracs7'", "choices=['bracs7', 'bracs3', 'bach4', 'isic7', 'cam2'], default='bracs7'"),
 ],
 "core/compare_runs.py": [
  ("TASK_CLASS_NAMES = {'isic7': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']}",
   "TASK_CLASS_NAMES = {'isic7': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'], 'cam2': ['Normal', 'Tumor']}"),
 ],
 "run_cgt_bracs.sh": [
  ("bach_patients | bach_graphs | bach_folds | isic_graphs | isic_split}",
   "bach_patients | bach_graphs | bach_folds | isic_graphs | isic_split | cam_graphs | cam_split}"),
  ("""  TAG="${TAG}isic_"; C3P="${C3P}isic_"
elif [ "$DATASET" != bracs ]; then echo "DATASET must be bracs, bach or isic"; exit 1; fi""",
   """  TAG="${TAG}isic_"; C3P="${C3P}isic_"
elif [ "$DATASET" = camelyon ]; then          # [sgt] CAMELYON16-derived patches, slide-disjoint Level-2 split
  [ "${CLASSES:-7}" = 7 ] && [ -z "${TASK:-}" ] || { echo "CLASSES and TASK apply to BRACS only"; exit 1; }
  [ -z "${INIT:-}" ] || { echo "INIT (released BRACS weights) does not apply to CAMELYON"; exit 1; }
  EXTRA="$EXTRA --task cam2"; PARG="--task cam2"
  TAG="${TAG}cam_"; C3P="${C3P}cam_"
elif [ "$DATASET" != bracs ]; then echo "DATASET must be bracs, bach, isic or camelyon"; exit 1; fi"""),
  ("""  if [ "$DATASET" = isic ]; then echo "${ISIC_OUT:?export ISIC_OUT}"; return; fi""",
   """  if [ "$DATASET" = isic ]; then echo "${ISIC_OUT:?export ISIC_OUT}"; return; fi
  if [ "$DATASET" = camelyon ]; then echo "${CAM_OUT:?export CAM_OUT}"; return; fi"""),
  ("""    if [ "$DATASET" = isic ]; then
      echo "expected parameters (ISIC, 7 classes)""",
   """    if [ "$DATASET" = camelyon ]; then
      echo "expected parameters (CAMELYON, 2 classes): cgt 8,431,574 | cgt_mase 8,449,050 | cgt_sre 8,431,706 |"
      echo "                                           sgt, sgt_mshuf, sgt_pshuf 8,449,182 | sgt_concat 8,451,770"
    elif [ "$DATASET" = isic ]; then
      echo "expected parameters (ISIC, 7 classes)"""),
  ("""    if [ "$DATASET" = isic ]; then DATA="$(data_dir)\"""",
   """    if [ "$DATASET" = isic ] || [ "$DATASET" = camelyon ]; then DATA="$(data_dir)\""""),
  ("""  *) echo "unknown step $STEP"; exit 1 ;;""",
   """  cam_graphs) : "${CAM_ROOT:?export CAM_ROOT}"; : "${CAM_SPLIT:?export CAM_SPLIT}"; : "${CAM_GRAPHS:?export CAM_GRAPHS}"
    # shellcheck disable=SC2086
    python build_tg_camelyon.py --root "$CAM_ROOT" --split "$CAM_SPLIT" --out "$CAM_GRAPHS" ${TRIAL:+--trial "$TRIAL"} \\
                                ${SHARD:+--shard "$SHARD"} ${NSHARDS:+--nshards "$NSHARDS"} ${OVERWRITE:+--overwrite} ;;
  cam_split) : "${CAM_GRAPHS:?export CAM_GRAPHS}"; : "${CAM_SPLIT:?export CAM_SPLIT}"; : "${CAM_OUT:?export CAM_OUT}"
    # shellcheck disable=SC2086
    python make_camelyon_split.py --src "$CAM_GRAPHS" --split "$CAM_SPLIT" --out "$CAM_OUT" ${ALLOW_MISSING:+--allow_missing} ${OVERWRITE:+--overwrite}
    python build_weights.py --tg_root "$CAM_OUT" ${OVERWRITE:+--overwrite} ;;
  *) echo "unknown step $STEP"; exit 1 ;;"""),
 ],
}

new = {}
for f, edits in E.items():
    t = open(f).read()
    if "cam2" in t or "camelyon" in t:
        print(f"{f}: already patched")
        continue
    for old, rep in edits:
        if t.count(old) != 1:
            sys.exit(f"patch does not fit {f} (nothing changed): {old[:70]!r}")
        t = t.replace(old, rep)
    new[f] = t
for f, t in new.items():
    open(f + ".new", "w").write(t)
    os.chmod(f + ".new", os.stat(f).st_mode)
for f in new:
    os.replace(f + ".new", f)
print("patched:", ", ".join(new) if new else "nothing to do")
