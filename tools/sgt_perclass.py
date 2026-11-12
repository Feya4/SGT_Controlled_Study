#!/usr/bin/env python
"""Per-class F1 on the BRACS test set, per seed and mean +- SD, for both checkpoint rules.
   python ~/sgt_perclass.py                     # SGT  (runs_sgtfull_e200_val_seed*)
   python ~/sgt_perclass.py e200                # CGT  (runs_e200_val_seed*)
   python ~/sgt_perclass.py sgtfull_c3_e200     # SGT trained on the three-class task, etc.
BACH (cross-validation): per-class F1 over the pooled folds comes from the compare step:
   DATASET=bach EPOCHS=100 PERCLASS=1 bash run_cgt_bracs.sh compare
"""
import glob
import os
import statistics as st
import sys

NAMES = {7: ['N', 'PB', 'UDH', 'ADH', 'FEA', 'DCIS', 'IC'], 3: ['benign', 'atypical', 'malignant'],
         4: ['normal', 'benign', 'in situ', 'invasive']}
BASE = os.environ.get('S2', '/workdir1.8t/fei27/CGT/GNNMD/CGT_sgt2')
tag = sys.argv[1] if len(sys.argv) > 1 else 'sgtfull_e200'
if 'bach' in tag:
    sys.exit('BACH runs are five folds per seed: use  DATASET=bach EPOCHS=100 PERCLASS=1 bash run_cgt_bracs.sh compare')
for rule, metric in (('checkpoint chosen on VALIDATION', 'best_val_weighted_f1_score'),
                     ('checkpoint chosen on TEST (CGT rule)', 'best_test_weighted_f1_score')):
    runs = {}
    pattern = os.path.join(BASE, 'runs_{}_val_seed*'.format(tag), 'logs', '*', 'classification_report_{}.txt'.format(metric))
    for f in sorted(glob.glob(pattern)):
        seed = f.split('_val_seed')[1].split(os.sep)[0]
        f1, sup = {}, {}
        for line in open(f):
            p = line.split()
            if len(p) == 5 and p[0].isdigit():
                f1[int(p[0])], sup[int(p[0])] = 100 * float(p[3]), int(p[4])
        if len(f1) in NAMES:
            runs.setdefault(seed, (f1, sup))
    print('\n=== runs_{}  |  {}  |  finished seeds: {}'.format(tag, rule, ' '.join(sorted(runs)) or 'none'))
    if not runs:
        continue
    seeds = sorted(runs)
    names = NAMES[len(runs[seeds[0]][0])]
    print('{:9s} {:>7s} '.format('class', 'images') + ' '.join('{:>7s}'.format('seed' + s) for s in seeds) + '   mean +- SD')
    for k, name in enumerate(names):
        v = [runs[s][0][k] for s in seeds]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print('{:9s} {:7d} '.format(name, runs[seeds[0]][1][k]) + ' '.join('{:7.2f}'.format(x) for x in v)
              + '   {:5.2f} +- {:4.2f}'.format(st.mean(v), sd))
