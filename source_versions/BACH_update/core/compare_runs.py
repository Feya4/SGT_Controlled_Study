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

--cv (BACH, k-fold cross-validation; run folders carry _f<k>_): for every seed the out-of-fold test predictions
of all folds are pooled, so each image is scored once by a model that never saw it; the seed-paired test and
the bootstrap (over the pooled images) are then computed exactly as above. Mean +- SD over folds is shown too.
--perclass adds per-class F1 (mean +- SD over seeds) of both models.
"""
import argparse
import glob
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

N_CLASSES = 7
METRICS = (('weighted_f1', 'wF1'), ('macro_f1', 'mF1'), ('accuracy', 'acc'))
# BRACS three-class task defined by the dataset authors; classes are in the order N, PB, UDH, ADH, FEA, DCIS, IC
GROUPS = {'bracs3': (np.array([0, 0, 0, 1, 1, 2, 2]), 'benign = N, PB, UDH | atypical = ADH, FEA | malignant = DCIS, IC')}
CLASS_NAMES = {7: ['N', 'PB', 'UDH', 'ADH', 'FEA', 'DCIS', 'IC'], 3: ['benign', 'atypical', 'malignant'],
               4: ['normal', 'benign', 'in situ', 'invasive']}      # 4 = BACH
FOLD_RE = re.compile(r'_f(\d+)_')


def fold_of(run_dir):
    m = FOLD_RE.search(Path(run_dir).name)
    return int(m.group(1)) if m else None


def parse(line):
    parts = line.rstrip('\n').split('\t')
    return parts[0], dict(kv.split('=', 1) for kv in parts[1:] if '=' in kv)


def load(pattern, metric='best_val_weighted_f1_score', cv=False):
    """{seed: results} or, with cv, {(fold, seed): results}"""
    runs = {}
    for d in sorted({d for pat in pattern for d in glob.glob(pat)}):
        fold = fold_of(d)
        if cv and fold is None:
            raise SystemExit('--cv: run folder {} has no _f<k>_ fold tag'.format(d))
        if not cv and fold is not None:
            raise SystemExit('{} is one fold of a cross-validation: compare with --cv (DATASET=bach)'.format(d))
        for f in sorted(Path(d).glob('logs/*/test_results.txt')):
            for line in open(str(f)):
                name, r = parse(line)
                if name != metric:
                    continue
                seed = r.get('seed')
                key = (fold, seed) if cv else seed
                if key in runs:
                    raise SystemExit('{} appears twice under {} ({}); move the old run aside'.format(
                        'fold {} seed {}'.format(*key) if cv else 'seed ' + seed, pattern, f))
                r['_dir'] = f.parent
                runs[key] = r
    return runs


def read_preds(run_dir, fname='test_predictions.csv', mapping=None):
    f = Path(run_dir) / fname
    if not f.exists():
        return None
    lines = [l.rstrip('\n').split(',') for l in open(str(f))]
    rows = lines[1:]
    y, p = np.array([int(r[1]) for r in rows]), np.array([int(r[2]) for r in rows])
    if mapping is not None and sum(c.startswith('logit_') for c in lines[0]) != len(mapping):
        raise SystemExit('{} is not a 7-class model; TASK=bracs3 maps 7-class runs only '
                         '(runs trained on 3 classes are compared without it)'.format(f))
    if mapping is not None:                     # 7-class label and prediction -> its group
        y, p = mapping[y], mapping[p]
    return [r[0] for r in rows], y, p


def n_logits(run_dir, fname):
    f = Path(run_dir) / fname
    return sum(c.startswith('logit_') for c in open(str(f)).readline().strip().split(',')) if f.exists() else None


def pooled_preds(R, seed, folds, fname, mapping=None):
    """out-of-fold predictions of one seed: all folds concatenated, ordered by image name"""
    parts = [read_preds(R[(k, seed)]['_dir'], fname, mapping) for k in folds]
    if any(p is None for p in parts):
        return None
    names = [n for p in parts for n in p[0]]
    if len(set(names)) != len(names):
        raise SystemExit('an image is in the {} of two folds (seed {}): the folds overlap'.format(fname, seed))
    y, p = np.concatenate([q[1] for q in parts]), np.concatenate([q[2] for q in parts])
    order = np.argsort(names)
    return [names[i] for i in order], y[order], p[order]


def per_class_f1(pred, k):
    _, y, p = pred
    out = []
    for c in range(k):
        tp = float(((y == c) & (p == c)).sum()); d = float((y == c).sum() + (p == c).sum())
        out.append(100 * 2 * tp / d if d else 0.0)
    return out


def scores_from_preds(pred, k):
    _, y, p = pred
    conf = np.zeros((1, k, k))
    np.add.at(conf[0], (y, p), 1)
    return [100 * float(v[0]) for v in f1_from_confusion(conf)]


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


def bootstrap(preds_a, preds_b, n_boot, seed, k=N_CLASSES):
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
            onehot = np.zeros((n, k * k), dtype=np.float32)
            onehot[np.arange(n), yy * k + pp] = 1
            conf = (w @ onehot).reshape(n_boot, k, k)
            out += np.stack(f1_from_confusion(conf))
            full += np.array([v[0] for v in f1_from_confusion(onehot.sum(0).reshape(1, k, k))])
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
    A, B = load(a.a, a.metric, a.cv), load(a.b, a.metric, a.cv)
    test_rule = a.metric.startswith('best_test')
    print('checkpoint rule: {}'.format(
        'best TEST weighted F1 over epochs (released-CGT rule; optimistic, report as such)' if test_rule
        else 'best validation weighted F1, test evaluated once'))
    if not A or not B:
        raise SystemExit('no finished runs with {} for {}'.format(a.metric, a.a if not A else a.b))
    skey = lambda s: int(s) if s.isdigit() else s
    if a.cv:
        folds = sorted({k for k, _ in A} | {k for k, _ in B})
        full = lambda R: {s for _, s in R if all((k, s) in R for k in folds)}
        seeds = sorted(full(A) & full(B), key=skey)
        only = sorted(({s for _, s in A} | {s for _, s in B}) - set(seeds), key=skey)
    else:
        seeds = sorted(set(A) & set(B), key=skey)
        only = sorted(set(A) ^ set(B))
    if not seeds:
        raise SystemExit('no seed is finished for {}both models'.format('every fold of ' if a.cv else ''))
    na, nb = a.names
    use_preds = a.from_preds or a.groups != 'none' or a.cv or a.perclass
    split = 'val' if a.score == 'val' else 'test'
    pred_file = '{}_predictions{}.csv'.format(split, '_testrule' if test_rule and split == 'test' else '')
    first = A[(folds[0], seeds[0]) if a.cv else seeds[0]]['_dir']
    mapping, n_cls = (GROUPS[a.groups][0], 3) if a.groups != 'none' else (None, n_logits(first, pred_file) or N_CLASSES)

    def preds(R, s, fname):
        return pooled_preds(R, s, folds, fname, mapping) if a.cv else read_preds(R[s]['_dir'], fname, mapping)
    if a.cv:
        print('cross-validation: {} folds; per seed the out-of-fold {} predictions of all folds are pooled '
              '(every image scored once)'.format(len(folds), split))
    if a.groups != 'none':
        print('task: BRACS 3-class ({}), 7-class predictions mapped to their group'.format(GROUPS[a.groups][1]))
    if use_preds:
        metrics = tuple((None, '{} {}'.format(split, l)) for l in ('wF1', 'mF1', 'acc'))
        print('scores computed from {} of each run'.format(pred_file))
    else:
        metrics = (('best_val_wf1', 'val wF1'),) if a.score == 'val' else METRICS
    if a.score == 'val':
        print('score compared: VALIDATION scores of each run (no test data involved)')
    print('{} = {}\n{} = {}\npaired seeds: {}{}'.format(na, a.a, nb, a.b, ' '.join(seeds),
          '  ({}, ignored: {})'.format('not finished for every fold of both models' if a.cv else 'unpaired',
                                        ' '.join(only)) if only else ''))
    W = max([9] + [len(n) + 1 + len(l) for n in (na, nb) for _, l in metrics])
    cols = lambda name: ' '.join('{:>{w}}'.format(name + ' ' + l, w=W) for _, l in metrics)
    nums = lambda vals: ' '.join('{:{w}.2f}'.format(v, w=W) for v in vals)
    print('\n{:>5} | {} | {}'.format('seed', cols(na), cols(nb)))
    rows = []
    for s in seeds:
        if use_preds:
            pa_s, pb_s = preds(A, s, pred_file), preds(B, s, pred_file)
            if pa_s is None or pb_s is None:
                raise SystemExit('{} missing for seed {} (run: bash run_cgt_bracs.sh predict for both models)'.format(
                    pred_file, s))
            va, vb = scores_from_preds(pa_s, n_cls), scores_from_preds(pb_s, n_cls)
        else:
            va = [100 * float(A[s][m]) for m, _ in metrics]
            vb = [100 * float(B[s][m]) for m, _ in metrics]
        rows.append((s, va, vb))
        print('{:>5} | {} | {}'.format(s, nums(va), nums(vb)))
    va = np.array([r[1] for r in rows])
    vb = np.array([r[2] for r in rows])
    n = len(seeds)
    sd = (lambda x: x.std(0, ddof=1)) if n > 1 else (lambda x: np.zeros(x.shape[1]))
    print('{:>5} | {} | {}'.format('mean', nums(va.mean(0)), nums(vb.mean(0))))
    print('{:>5} | {} | {}'.format('SD', nums(sd(va)), nums(sd(vb))))
    if a.cv:                                   # the usual cross-validation summary: mean +- SD over folds
        fa, fb = [], []
        for k in folds:
            fa.append(np.mean([scores_from_preds(read_preds(A[(k, s)]['_dir'], pred_file, mapping), n_cls) for s in seeds], 0))
            fb.append(np.mean([scores_from_preds(read_preds(B[(k, s)]['_dir'], pred_file, mapping), n_cls) for s in seeds], 0))
        fa, fb = np.array(fa), np.array(fb)
        print('\nper fold (mean over the {} seeds)'.format(len(seeds)))
        print('{:>5} | {} | {}'.format('fold', cols(na), cols(nb)))
        for k, x, y in zip(folds, fa, fb):
            print('{:>5} | {} | {}'.format(k, nums(x), nums(y)))
        print('{:>5} | {} | {}'.format('mean', nums(fa.mean(0)), nums(fb.mean(0))))
        print('{:>5} | {} | {}'.format('SD', nums(fa.std(0, ddof=1) if len(folds) > 1 else 0 * fa[0]),
                                       nums(fb.std(0, ddof=1) if len(folds) > 1 else 0 * fb[0])))

    print('\nseed-paired difference {} - {} (percentage points), df = {}'.format(nb, na, n - 1))
    for k, (_, lab) in enumerate(metrics):
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

    pa = [preds(A, s, 'test_predictions.csv') for s in seeds]
    pb = [preds(B, s, 'test_predictions.csv') for s in seeds]
    if a.score == 'val':
        pass
    elif test_rule:
        print('\npaired bootstrap over test graphs: not computed, the checkpoints were chosen on these test graphs')
    elif any(p is None for p in pa + pb):
        print('\npaired bootstrap: skipped, test_predictions.csv missing (run: bash run_cgt_bracs.sh predict)')
    else:
        print('\npaired bootstrap over {} {}test graphs ({:,} resamples, same resample for all {} {})'.format(
            len(pa[0][1]), 'pooled out-of-fold ' if a.cv else '', a.n_boot, 2 * n,
            'seeds x models' if a.cv else 'runs'))
        for (diff, lo, hi, p), (_, lab) in zip(bootstrap(pa, pb, a.n_boot, a.boot_seed, n_cls), METRICS):
            print('  {:4s} {:+.2f}  95% CI [{:+.2f}, {:+.2f}]  p = {:.4f}'.format(lab, 100 * diff, 100 * lo, 100 * hi, p))

    if a.perclass:
        names = CLASS_NAMES.get(n_cls, [str(c) for c in range(n_cls)])
        print('\nper-class F1 on {} ({}), mean +- SD over {} seeds'.format(pred_file, 'pooled folds' if a.cv else 'test set', n))
        ca = np.array([per_class_f1(preds(A, s, pred_file), n_cls) for s in seeds])
        cb = np.array([per_class_f1(preds(B, s, pred_file), n_cls) for s in seeds])
        y0 = preds(A, seeds[0], pred_file)[1]
        print('{:10s} {:>6s} | {:>15s} | {:>15s} | {:>7s}'.format('class', 'images', na, nb, 'diff'))
        for c, nm in enumerate(names):
            print('{:10s} {:6d} | {:6.2f} +- {:5.2f} | {:6.2f} +- {:5.2f} | {:+7.2f}'.format(
                nm, int((y0 == c).sum()), ca[:, c].mean(), sd(ca)[c], cb[:, c].mean(), sd(cb)[c],
                cb[:, c].mean() - ca[:, c].mean()))

    if a.csv:
        with open(a.csv, 'w') as f:
            f.write('seed,' + ','.join('{}_{}'.format(na, l) for _, l in metrics) + ','
                    + ','.join('{}_{}'.format(nb, l) for _, l in metrics) + '\n')
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
    ap.add_argument('--groups', choices=['none'] + sorted(GROUPS), default='none',
                    help="'bracs3': BRACS three-class task (benign / atypical / malignant) from the 7-class predictions")
    ap.add_argument('--from_preds', action='store_true',
                    help='compute the scores from the saved prediction files (gives macro F1 and accuracy on validation too)')
    ap.add_argument('--score', choices=['test', 'val'], default='test',
                    help="'val' compares the best validation weighted F1 of each run (e.g. fine-tuned runs)")
    ap.add_argument('--cv', action='store_true',
                    help='k-fold cross-validation runs (_f<k>_ in the folder name, e.g. BACH): pool the folds per seed')
    ap.add_argument('--perclass', action='store_true', help='per-class F1 of both models')
    ap.add_argument('--metric', default='best_val_weighted_f1_score',
                    help="'best_test_weighted_f1_score' = released-CGT rule (runs trained with --test_every_epoch)")
    main(ap.parse_args())
