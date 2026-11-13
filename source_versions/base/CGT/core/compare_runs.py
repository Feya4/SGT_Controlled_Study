#!/usr/bin/env python3
"""[sgt] Paired comparison of two models trained with the same seeds (editor point 3).

    python compare_runs.py --a '../runs_val_seed*' --b '../runs_sgtfull_val_seed*' --names CGT SGT

For each model the run of every seed is read (checkpoint chosen on validation weighted F1). Reported:
  * per-seed test weighted F1, macro F1, accuracy and the mean +- SD over seeds
  * seed-paired difference B - A: mean, 95% t interval, paired t-test (df = seeds - 1), Cohen's d_z
  * paired bootstrap over test graphs (needs test_predictions.csv from predict_runs.py): the same resample
    of graphs is applied to every seed of both models; statistic = mean over seeds of B minus mean over
    seeds of A; percentile 95% interval and two-sided p.
Only seeds present for both models are used. --csv writes the per-seed table.
"""
import argparse
import glob
import math
from pathlib import Path

import numpy as np
from scipy import stats

N_CLASSES = 7
METRICS = (('weighted_f1', 'wF1'), ('macro_f1', 'mF1'), ('accuracy', 'acc'))


def parse(line):
    parts = line.rstrip('\n').split('\t')
    return parts[0], dict(kv.split('=', 1) for kv in parts[1:] if '=' in kv)


def load(pattern, metric='best_val_weighted_f1_score'):
    runs = {}
    for d in sorted({d for pat in pattern for d in glob.glob(pat)}):
        for f in sorted(Path(d).glob('logs/*/test_results.txt')):
            for line in open(str(f)):
                name, r = parse(line)
                if name != metric:
                    continue
                seed = r.get('seed')
                if seed in runs:
                    raise SystemExit('seed {} appears twice under {} ({}); move the old run aside'.format(seed, pattern, f))
                r['_dir'] = f.parent
                runs[seed] = r
    return runs


def read_preds(run_dir):
    f = Path(run_dir) / 'test_predictions.csv'
    if not f.exists():
        return None
    rows = [l.rstrip('\n').split(',') for l in open(str(f))][1:]
    return [r[0] for r in rows], np.array([int(r[1]) for r in rows]), np.array([int(r[2]) for r in rows])


def f1_from_confusion(conf):
    """conf: (B, C, C) counts [true, pred] -> weighted F1, macro F1, accuracy (sklearn conventions)."""
    tp = np.diagonal(conf, axis1=1, axis2=2).astype(float)
    support = conf.sum(2).astype(float)
    predicted = conf.sum(1).astype(float)
    denom = support + predicted
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    total = support.sum(1)
    wf1 = (f1 * support).sum(1) / total
    present = denom > 0                                    # sklearn: labels in y_true or y_pred
    mf1 = (f1 * present).sum(1) / present.sum(1)
    acc = tp.sum(1) / total
    return wf1, mf1, acc


def bootstrap(preds_a, preds_b, n_boot, seed):
    names, y = preds_a[0][0], preds_a[0][1]
    for g, yy, _ in preds_a + preds_b:
        if g != names or not np.array_equal(yy, y):
            raise SystemExit('test graphs or labels differ between runs; predict all runs from the same --tg_path')
    n = len(y)
    rs = np.random.RandomState(seed)
    w = rs.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float32)   # (B, n) resample counts

    def stat(group):
        out = np.zeros((3, n_boot))
        full = np.zeros(3)
        for _, yy, pp in group:
            onehot = np.zeros((n, N_CLASSES * N_CLASSES), dtype=np.float32)
            onehot[np.arange(n), yy * N_CLASSES + pp] = 1
            conf = (w @ onehot).reshape(n_boot, N_CLASSES, N_CLASSES)
            out += np.stack(f1_from_confusion(conf))
            full += np.array([v[0] for v in f1_from_confusion(onehot.sum(0).reshape(1, N_CLASSES, N_CLASSES))])
        return out / len(group), full / len(group)

    sa, fa = stat(preds_a)
    sb, fb = stat(preds_b)
    diff = sb - sa
    res = []
    for k in range(3):
        lo, hi = np.percentile(diff[k], [2.5, 97.5])
        p = min(1.0, 2 * min((diff[k] <= 0).mean(), (diff[k] >= 0).mean()))
        res.append((fb[k] - fa[k], lo, hi, p))
    return res


def main(a):
    A, B = load(a.a), load(a.b)
    if not A or not B:
        raise SystemExit('no finished runs for {}'.format(a.a if not A else a.b))
    seeds = sorted(set(A) & set(B), key=lambda s: int(s) if s.isdigit() else s)
    only = sorted(set(A) ^ set(B))
    na, nb = a.names
    print('{} = {}\n{} = {}\npaired seeds: {}{}'.format(na, a.a, nb, a.b, ' '.join(seeds),
          '  (unpaired, ignored: {})'.format(' '.join(only)) if only else ''))
    print('\n{:>5} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7}'.format('seed', na + ' wF1', 'mF1', 'acc', nb + ' wF1', 'mF1', 'acc'))
    rows = []
    for s in seeds:
        va = [100 * float(A[s][m]) for m, _ in METRICS]
        vb = [100 * float(B[s][m]) for m, _ in METRICS]
        rows.append((s, va, vb))
        print('{:>5} | {:7.2f} {:7.2f} {:7.2f} | {:7.2f} {:7.2f} {:7.2f}'.format(s, *(va + vb)))
    va = np.array([r[1] for r in rows])
    vb = np.array([r[2] for r in rows])
    n = len(seeds)
    sd = (lambda x: x.std(0, ddof=1)) if n > 1 else (lambda x: np.zeros(x.shape[1]))
    print('{:>5} | {} | {}'.format('mean', ' '.join('{:7.2f}'.format(v) for v in va.mean(0)),
                                  ' '.join('{:7.2f}'.format(v) for v in vb.mean(0))))
    print('{:>5} | {} | {}'.format('SD', ' '.join('{:7.2f}'.format(v) for v in sd(va)),
                                  ' '.join('{:7.2f}'.format(v) for v in sd(vb))))

    print('\nseed-paired difference {} - {} (percentage points), df = {}'.format(nb, na, n - 1))
    for k, (_, lab) in enumerate(METRICS):
        d = vb[:, k] - va[:, k]
        if n < 2:
            print('  {:4s} {:+.2f} (one seed: no test)'.format(lab, d.mean()))
            continue
        sdd = d.std(ddof=1)
        half = stats.t.ppf(0.975, n - 1) * sdd / math.sqrt(n)
        t, p = stats.ttest_rel(vb[:, k], va[:, k])
        dz = d.mean() / sdd if sdd > 0 else float('nan')
        print('  {:4s} {:+.2f}  95% CI [{:+.2f}, {:+.2f}]  t({}) = {:.2f}  p = {:.4f}  d_z = {:.2f}  ({} of {} seeds better)'
              .format(lab, d.mean(), d.mean() - half, d.mean() + half, n - 1, t, p, dz, int((d > 0).sum()), n))

    pa = [read_preds(A[s]['_dir']) for s in seeds]
    pb = [read_preds(B[s]['_dir']) for s in seeds]
    if any(p is None for p in pa + pb):
        print('\npaired bootstrap: skipped, test_predictions.csv missing (run: bash run_cgt_bracs.sh predict)')
    else:
        print('\npaired bootstrap over {} test graphs ({:,} resamples, same resample for all {} runs)'.format(
            len(pa[0][1]), a.n_boot, 2 * n))
        for (diff, lo, hi, p), (_, lab) in zip(bootstrap(pa, pb, a.n_boot, a.boot_seed), METRICS):
            print('  {:4s} {:+.2f}  95% CI [{:+.2f}, {:+.2f}]  p = {:.4f}'.format(lab, 100 * diff, 100 * lo, 100 * hi, p))

    if a.csv:
        with open(a.csv, 'w') as f:
            f.write('seed,' + ','.join('{}_{}'.format(na, l) for _, l in METRICS) + ','
                    + ','.join('{}_{}'.format(nb, l) for _, l in METRICS) + '\n')
            for s, x, y in rows:
                f.write(s + ',' + ','.join('{:.4f}'.format(v) for v in x + y) + '\n')
        print('\nwrote', a.csv)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', nargs='+', required=True, help="baseline run folders or globs")
    ap.add_argument('--b', nargs='+', required=True, help="model run folders or globs")
    ap.add_argument('--names', nargs=2, default=['A', 'B'])
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--boot_seed', type=int, default=0)
    ap.add_argument('--csv', default=None)
    main(ap.parse_args())
