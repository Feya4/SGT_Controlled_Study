#!/usr/bin/env python3
"""
[rerun-addition] Summarise CGT training runs: one line per seed and mean +- SD for the checkpoint with the best
VALIDATION weighted F1 (the rule used for SGT), read from <runs>/*/logs/*/test_results.txt.

    python summarize_cgt_runs.py --runs ../runs_val_seed0 ../runs_val_seed1 ...   (or a glob: '../runs_val_seed*')
"""
import argparse
import glob
import re
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
                if line.split('\t', 1)[0] == a.metric and 'macro_f1=' in line:
                    r = parse(line)
                    r['dir'] = str(f.parent)
                    m = re.search(r'_f(\d+)_', Path(d).name)     # BACH cross-validation fold
                    r['fold'] = m.group(1) if m else ''
                    rows.append(r)
    if not rows:
        print('no finished runs with {} found in'.format(a.metric), dirs)
        return
    test_rule = a.metric.startswith('best_test')
    print('checkpoint rule: {}'.format(
        'best TEST weighted F1 over epochs (released-CGT rule; optimistic, report as such)' if test_rule
        else 'best validation weighted F1, test evaluated once'))
    cv = any(r['fold'] for r in rows)
    if cv:
        rows.sort(key=lambda r: (int(r['seed']) if r['seed'].isdigit() else 0, int(r['fold'] or 0)))
        print('cross-validation runs: one line per fold and seed; the mean below is over all of them '
              '(pooled out-of-fold scores: compare_runs.py --cv)')
    print(('{:>4} '.format('fold') if cv else '') + '{:>5} {:>14} {:>9} {:>9} {:>9} {:>9} {:>9}  {}'.format('seed', 'model', 'val_split', 'val_wF1', 'test_wF1',
                                                                     'test_mF1', 'test_acc', 'best epoch'))
    for r in rows:
        label = r.get('model') or {'real': 'mase', 'shuffle': 'mase+mshuf', 'concat': 'concat'}.get(
            r.get('morph', 'none'), 'cgt')                      # runs written before the model field existed
        print(('{:>4} '.format(r['fold']) if cv else '') + '{:>5} {:>14} {:>9} {:9.2f} {:9.2f} {:9.2f} {:9.2f}  {}'.format(
            r['seed'], label, r['val_split'], 100 * float(r['best_val_wf1']), 100 * float(r['weighted_f1']),
            100 * float(r['macro_f1']), 100 * float(r['accuracy']),
            r.get('best_test_wf1_epoch', '?') if test_rule else r['best_val_wf1_epoch']))

    def ms(key):
        v = [100 * float(r[key]) for r in rows]
        return '{:.2f} +- {:.2f}'.format(st.mean(v), st.stdev(v)) if len(v) > 1 else '{:.2f}'.format(v[0])
    print('mean +- SD over {} runs: val wF1 {} | test wF1 {} | test mF1 {} | test acc {}'.format(
        len(rows), ms('best_val_wf1'), ms('weighted_f1'), ms('macro_f1'), ms('accuracy')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--metric', default='best_val_weighted_f1_score',
                   help="results line to read; 'best_test_weighted_f1_score' = released-CGT rule "
                        "(runs trained with --test_every_epoch)")
    main(p.parse_args())
