#!/usr/bin/env python3
"""[sgt] K-fold cross-validation datasets for BACH in CGT's input format.

Every image goes to one of K folds, stratified by class and, when a patient file is given, grouped by patient
(images of one patient never end up in two folds). Fold k: test = fold k, val = fold (k+1) mod K, train = the
other folds, so every image is tested exactly once and used once for checkpoint selection.
Per fold the graphs are converted exactly as BRACS (convert_hact_graphs.py): node features = 512-d ResNet
features + centroid / (width, height); MASE attributes standardised with that fold's training statistics
(log area -> 16 bins over [-3, 3]; solidity -> 16 bins over [0, 1]).

Input   <src>/all/<stem>.bin and <src>/image_sizes.json     (from build_tg_bach.py)
        --patients: optional, the organisers' ICIAR2018_BACH_dataset_micro_patient file (.csv or .xlsx); images
                    it does not identify come from other patients (organisers) and are kept as their own group
Output  <out>/fold<k>/tissue_graphs/{train,val,test}/<stem>.bin, <out>/folds.json
Then    python build_weights.py --tg_root <out>/fold<k>     (run_cgt_bracs.sh bach_folds does both)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import dgl
from dgl.data.utils import load_graphs, save_graphs
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from convert_hact_graphs import bin16, centroid_order
from bach_patients import read_patients          # any layout of the organisers' patient file (.csv / .xlsx)

CLASSES = ['normal', 'benign', 'in situ', 'invasive']


def main(a):
    src, out = Path(a.src), Path(a.out)
    files = sorted((src / 'all').glob('*.bin'))
    if not files:
        raise SystemExit('no graphs in {}'.format(src / 'all'))
    sizes = json.load(open(str(src / 'image_sizes.json')))
    stems = [f.stem for f in files]
    labels = np.array([int(load_graphs(str(f))[1]['label'].view(-1)[0]) for f in files])
    if a.patients:
        pat = read_patients(a.patients, set(stems))
        groups = np.array([pat.get(s, 'unknown_' + s) for s in stems])   # unidentified: own group
        print('patient file: {} of {} images identified ({} patients); {} unidentified images kept as their own '
              'group; {} groups in all'.format(len(pat), len(stems), len(set(pat.values())), len(stems) - len(pat),
                                               len(set(groups))))
        splitter = StratifiedGroupKFold(n_splits=a.k, shuffle=True, random_state=a.seed)
        split_iter = splitter.split(np.zeros(len(stems)), labels, groups)
    else:
        print('WARNING: no patient file - folds are stratified by class only; images of one patient may appear '
              'in different folds (state this in the paper).')
        splitter = StratifiedKFold(n_splits=a.k, shuffle=True, random_state=a.seed)
        split_iter = splitter.split(np.zeros(len(stems)), labels)
    fold_of = np.empty(len(stems), dtype=int)
    for k, (_, test_idx) in enumerate(split_iter):
        fold_of[test_idx] = k
    out.mkdir(parents=True, exist_ok=True)
    assignment = {s: int(f) for s, f in zip(stems, fold_of)}
    old_dirs = sorted(out.glob('fold[0-9]*'))
    old = json.load(open(str(out / 'folds.json'))) if (out / 'folds.json').exists() else None
    if old_dirs and old != assignment:          # e.g. a patient file added later: existing splits are stale
        if not a.overwrite:
            raise SystemExit('{} holds folds made with a different assignment; rerun with --overwrite '
                             '(deletes {}/fold*, graphs and weights, and every run trained on them is invalid)'.format(out, out))
        for d in old_dirs:
            shutil.rmtree(str(d))
        print('removed the old folds:', ' '.join(d.name for d in old_dirs))
    json.dump(assignment, open(str(out / 'folds.json'), 'w'), indent=1)
    json.dump({'patient_file': a.patients, 'grouped_by_patient': bool(a.patients), 'k': a.k, 'seed': a.seed,
               'images_identified': len(pat) if a.patients else 0,
               'patients_identified': len(set(pat.values())) if a.patients else 0,
               'patients_split_across_folds': int(sum(len(set(fold_of[groups == g])) > 1 for g in set(groups)))
               if a.patients else None},
              open(str(out / 'folds_info.json'), 'w'), indent=1)
    order = centroid_order(files, sizes)
    graphs = {f.stem: load_graphs(str(f))[0][0] for f in files}
    for k in range(a.k):
        role = {}
        for s, f in zip(stems, fold_of):
            role[s] = 'test' if f == k else ('val' if f == (k + 1) % a.k else 'train')
        train_logs = torch.cat([torch.log(graphs[s].ndata['area'].float().clamp(min=1.0))
                                for s in stems if role[s] == 'train'])
        mu, sd = float(train_logs.mean()), float(train_logs.std())
        counts = {r: [0] * len(CLASSES) for r in ('train', 'val', 'test')}
        for s, y in zip(stems, labels):
            g, r = graphs[s], role[s]
            counts[r][y] += 1
            dst = out / 'fold{}'.format(k) / 'tissue_graphs' / r / (s + '.bin')
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and not a.overwrite:
                continue
            feat = g.ndata['feat'].float()
            feat = feat[:, :512] if feat.shape[1] == 514 else feat
            if feat.shape[1] != 512:
                raise ValueError('{}: expected 512-d node features, got {}'.format(s, feat.shape[1]))
            c = g.ndata['centroid'].float()[:, list(order)]
            w, h = sizes[s]
            loc = (c / torch.tensor([float(w), float(h)])).clamp(0.0, 1.0)
            new = dgl.DGLGraph()
            new.add_nodes(g.number_of_nodes())
            u, v = g.edges()
            new.add_edges(u, v)
            new.ndata['feat'] = torch.cat([feat, loc], dim=1)
            new.ndata['centroid'] = c
            area_z = (torch.log(g.ndata['area'].float().clamp(min=1.0)) - mu) / sd
            conv = g.ndata['solidity'].float()
            new.ndata['area_z'] = area_z
            new.ndata['conv'] = conv
            new.ndata['area_bin'] = bin16(area_z, -3.0, 3.0)
            new.ndata['conv_bin'] = bin16(conv, 0.0, 1.0)
            save_graphs(str(dst), [new], {'label': torch.tensor([int(y)])})
        print('fold {}: log-area mean {:.4f} sd {:.4f} | '.format(k, mu, sd) + ' | '.join(
            '{} {} ({})'.format(r, sum(counts[r]), '/'.join(str(x) for x in counts[r])) for r in ('train', 'val', 'test')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='output of build_tg_bach.py')
    ap.add_argument('--out', required=True, help='folder for fold<k>/ datasets')
    ap.add_argument('--patients', default=None, help='optional table image -> patient for patient-wise folds')
    ap.add_argument('-k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0, help='seed of the fold assignment')
    ap.add_argument('--overwrite', action='store_true')
    main(ap.parse_args())
