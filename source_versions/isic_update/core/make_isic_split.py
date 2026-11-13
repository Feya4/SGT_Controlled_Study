#!/usr/bin/env python3
"""[sgt] ISIC 2018 in CGT's input format, on the fixed lesion-grouped split (as BRACS: one train/val/test split).

The split file assigns every image to train / val / test with all images of a lesion (and of near-duplicate
lesion groups) in the same set. Graphs are converted exactly as for BRACS and BACH (convert_hact_graphs.py,
make_bach_folds.py): node features = 512-d ResNet features + centroid / (width, height); MASE attributes
standardised with the TRAINING nodes' statistics (log area -> 16 bins over [-3, 3]; solidity -> 16 bins over [0, 1]).

Input   <src>/all/<image>.bin and <src>/image_sizes*.json      (from build_tg_isic.py)
        --split isic2018_lesion_split.csv
Output  <out>/tissue_graphs/{train,val,test}/<image>.bin, <out>/area_stats.json
Then    python build_weights.py --tg_root <out>                (run_cgt_bracs.sh isic_split does both)
--allow_missing converts whatever graphs exist (for the --trial check of build_tg_isic.py); without it every
image of the split must have its graph.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import dgl
from dgl.data.utils import load_graphs, save_graphs

from convert_hact_graphs import bin16, centroid_order

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
SPLITS = ('train', 'val', 'test')


def main(a):
    src, out = Path(a.src), Path(a.out)
    rows = [(r['image'].strip(), int(r['label']), r['split'].strip())
            for r in csv.DictReader(open(a.split, encoding='utf-8-sig'))]
    if {s for _, _, s in rows} - set(SPLITS):
        raise SystemExit('unexpected split names in {}: {}'.format(a.split, sorted({s for _, _, s in rows})))
    sizes = {}
    for f in sorted(src.glob('image_sizes*.json')):
        sizes.update(json.load(open(str(f))))
    have = {f.stem: f for f in (src / 'all').glob('*.bin')}
    missing = [im for im, _, _ in rows if im not in have or im not in sizes]
    if missing and not a.allow_missing:
        raise SystemExit('{} of {} images have no graph (or no size), e.g. {}; finish build_tg_isic.py first '
                         '(or --allow_missing for a trial)'.format(len(missing), len(rows), missing[:3]))
    rows = [r for r in rows if r[0] in have and r[0] in sizes]
    print('{} images with graphs{}'.format(len(rows), ' ({} missing, skipped)'.format(len(missing)) if missing else ''))
    for im, y, _ in rows:                      # the stored label must be the split file's label
        stored = int(load_graphs(str(have[im]))[1]['label'].view(-1)[0]) if a.check_labels else y
        if stored != y:
            raise SystemExit('{}: graph label {} but split file says {}'.format(im, stored, y))
    order = centroid_order([have[im] for im, _, _ in rows], sizes)
    logs = [torch.log(load_graphs(str(have[im]))[0][0].ndata['area'].float().clamp(min=1.0))
            for im, _, s in rows if s == 'train']
    v = torch.cat(logs)
    mu, sd = float(v.mean()), float(v.std())
    out.mkdir(parents=True, exist_ok=True)
    json.dump({'mean_log_area': mu, 'std_log_area': sd, 'train_graphs': len(logs)},
              open(str(out / 'area_stats.json'), 'w'), indent=2)
    print('log-area statistics of the training nodes: mean {:.4f}  std {:.4f}'.format(mu, sd))
    counts = {s: [0] * len(CLASSES) for s in SPLITS}
    n_new = n_clip = 0
    for im, y, s in rows:
        counts[s][y] += 1
        dst = out / 'tissue_graphs' / s / (im + '.bin')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not a.overwrite:
            continue
        g = load_graphs(str(have[im]))[0][0]
        feat = g.ndata['feat'].float()
        feat = feat[:, :512] if feat.shape[1] == 514 else feat
        if feat.shape[1] != 512:
            raise ValueError('{}: expected 512-d node features, got {}'.format(im, feat.shape[1]))
        c = g.ndata['centroid'].float()[:, list(order)]
        w, h = sizes[im]
        loc = c / torch.tensor([float(w), float(h)])
        if bool((loc < 0).any() or (loc > 1).any()):
            n_clip += 1
            loc = loc.clamp(0.0, 1.0)
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
        save_graphs(str(dst), [new], {'label': torch.tensor([y])})
        n_new += 1
    for s in SPLITS:
        print('{:5s}: {:5d} graphs | {}'.format(s, sum(counts[s]), '  '.join(
            '{}={}'.format(c, n) for c, n in zip(CLASSES, counts[s]))))
    print('{} graphs written to {} | graphs with centroids clipped to the image: {}'.format(
        n_new, out / 'tissue_graphs', n_clip))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='output of build_tg_isic.py')
    ap.add_argument('--split', required=True, help='isic2018_lesion_split.csv')
    ap.add_argument('--out', required=True, help='dataset root: <out>/tissue_graphs/<split>/')
    ap.add_argument('--allow_missing', action='store_true', help='convert only the graphs that exist (trial)')
    ap.add_argument('--no_check_labels', dest='check_labels', action='store_false',
                    help='skip re-reading every graph to compare its label with the split file')
    ap.add_argument('--overwrite', action='store_true')
    main(ap.parse_args())
