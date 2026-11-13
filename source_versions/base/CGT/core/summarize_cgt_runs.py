#!/usr/bin/env python3
"""
[rerun-addition] Summarise CGT training runs: one line per seed and mean +- SD for the checkpoint with the best
VALIDATION weighted F1 (the rule used for SGT), read from <runs>/*/logs/*/test_results.txt.

    python summarize_cgt_runs.py --runs ../runs_val_seed0 ../runs_val_seed1 ...   (or a glob: '../runs_val_seed*')
"""
import argparse
import glob
import statistics as st
from pathlib import Path


def parse(line):
    parts = line.strip().split('\t')
    rec = {'metric': parts[0]}
    for kv in parts[1:]:
        k, v = kv.split('=', 1)
        rec[k] = v
    return rec


def main(a):
    dirs = sorted({d for pat in a.runs for d in glob.glob(pat)})
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob('logs/*/test_results.txt')):
            for line in open(f):
                if line.startswith('best_val_weighted_f1_score') and 'macro_f1=' in line:
                    r = parse(line)
                    r['dir'] = str(f.parent)
                    rows.append(r)
    if not rows:
        print('no finished runs found in', dirs)
        return
    print('{:>5} {:>14} {:>9} {:>9} {:>9} {:>9} {:>9}  {}'.format('seed', 'model', 'val_split', 'val_wF1', 'test_wF1',
                                                                     'test_mF1', 'test_acc', 'best epoch'))
    for r in rows:
        label = r.get('model') or {'real': 'mase', 'shuffle': 'mase+mshuf', 'concat': 'concat'}.get(
            r.get('morph', 'none'), 'cgt')                      # runs written before the model field existed
        print('{:>5} {:>14} {:>9} {:9.2f} {:9.2f} {:9.2f} {:9.2f}  {}'.format(
            r['seed'], label, r['val_split'], 100 * float(r['best_val_wf1']), 100 * float(r['weighted_f1']),
            100 * float(r['macro_f1']), 100 * float(r['accuracy']), r['best_val_wf1_epoch']))

    def ms(key):
        v = [100 * float(r[key]) for r in rows]
        return '{:.2f} +- {:.2f}'.format(st.mean(v), st.stdev(v)) if len(v) > 1 else '{:.2f}'.format(v[0])
    print('mean +- SD over {} runs: val wF1 {} | test wF1 {} | test mF1 {} | test acc {}'.format(
        len(rows), ms('best_val_wf1'), ms('weighted_f1'), ms('macro_f1'), ms('accuracy')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    main(p.parse_args())
