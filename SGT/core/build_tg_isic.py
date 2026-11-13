#!/usr/bin/env python3
"""[sgt] Region graphs for the ISIC 2018 Task 3 dermoscopy images, built with the BRACS builder.

Same builder as for BRACS and BACH (build_tg.TgBuilding, as called by build_tg_bach.py): colour-merged
superpixels, ResNet-34 region features (512-d), region adjacency graph; per region also its area and solidity
(convexity) for MASE, from the same superpixel map, with the same node-order check as for BACH.
One deliberate difference: the Vahadane H&E stain normalisation to the BRACS target image is NOT applied by
default, because dermoscopy images are not H&E stained and their colour is itself diagnostic. --stain_norm
applies it exactly as for BRACS and BACH.

Input   --images  folder with the 10,015 ISIC_xxxxxxx.jpg of ISIC 2018 Task 3 (training set = HAM10000)
        --split   isic2018_lesion_split.csv: image, lesion_id, group, label, class, split;
                  label 0..6 = MEL, NV, BCC, AKIEC, BKL, DF, VASC (checked against the class column)
Output  <out>/all/<image>.bin   DGL graph: feat (512 + location), centroid, area, solidity; label 0..6
        <out>/image_sizes.json  {image: [width, height]}   (image_sizes_shard<k>.json with --nshards > 1)
Options --trial K     only K images per (class, split): a quick end-to-end check (7 x 3 x K images)
        --shard k --nshards n   build every n-th image, starting at k (n builders in parallel)

    python build_tg_isic.py --images .../ISIC2018_Task3_Training_Input --split .../isic2018_lesion_split.csv \\
                            --out /path/to/isic_graphs
"""
import argparse
import csv
import json
import os
import time
from glob import glob

import numpy as np
import torch
from dgl.data.utils import save_graphs
from PIL import Image
from tqdm import tqdm

from build_tg import TgBuilding
from build_tg_bach import region_attributes      # area, solidity, centroid per region, in node order

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


def read_split(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    bad = [r['image'] for r in rows if CLASSES[int(r['label'])] != r['class'].strip()]
    if bad:
        raise SystemExit('label and class columns disagree for {} images, e.g. {}'.format(len(bad), bad[:3]))
    return [(r['image'].strip(), int(r['label']), r['split'].strip()) for r in rows]


def main(a):
    rows = sorted(read_split(a.split))
    print('{}: {} images | '.format(os.path.basename(a.split), len(rows)) + '  '.join(
        '{}={}'.format(c, sum(1 for _, y, _ in rows if y == k)) for k, c in enumerate(CLASSES)))
    if a.trial:
        keep, seen = [], {}
        for r in rows:
            key = (r[1], r[2])
            if seen.get(key, 0) < a.trial:
                keep.append(r)
                seen[key] = seen.get(key, 0) + 1
        rows = keep
        print('trial: {} images ({} per class and split where available)'.format(len(rows), a.trial))
    rows = rows[a.shard::a.nshards]
    missing = [im for im, _, _ in rows if not os.path.exists(os.path.join(a.images, im + '.jpg'))]
    if missing:
        raise SystemExit('{} images missing in {}, e.g. {}'.format(len(missing), a.images, missing[:3]))
    out_dir = os.path.join(a.out, 'all')
    os.makedirs(out_dir, exist_ok=True)
    sizes_path = os.path.join(a.out, 'image_sizes.json' if a.nshards == 1 else 'image_sizes_shard{}.json'.format(a.shard))
    sizes = json.load(open(sizes_path)) if os.path.exists(sizes_path) else {}
    builder = TgBuilding(normalize=a.stain_norm)
    print('stain normalisation: {}'.format('Vahadane to the BRACS target (as BRACS/BACH)' if a.stain_norm
                                           else 'off (dermoscopy is not H&E stained)'))
    failed, stats = [], []
    for im, label, _ in tqdm(rows, desc='ISIC graphs{}'.format(' shard {}/{}'.format(a.shard, a.nshards)
                                                                if a.nshards > 1 else '')):
        dst = os.path.join(out_dir, im + '.bin')
        image = np.array(Image.open(os.path.join(a.images, im + '.jpg')).convert('RGB'))
        sizes[im] = [int(image.shape[1]), int(image.shape[0])]          # width, height
        if os.path.exists(dst) and not a.overwrite:
            continue
        t0 = time.time()
        if a.stain_norm:
            try:
                image = builder.normalizer.process(image)
            except Exception:
                print('Warning: {} failed during stain normalization (kept unnormalised, as for BRACS).'.format(im))
        try:
            graph, superpixels = builder._build_tg(image)
            area, solidity, cent = region_attributes(superpixels)
            if area.shape[0] != graph.number_of_nodes():
                raise ValueError('{} regions but {} graph nodes'.format(area.shape[0], graph.number_of_nodes()))
            gc = graph.ndata['centroid'].float()
            if not (torch.allclose(gc, cent, atol=1.0) or torch.allclose(gc, cent[:, [1, 0]], atol=1.0)):
                raise ValueError('region order differs from graph node order')
            graph.ndata['area'] = area
            graph.ndata['solidity'] = solidity
            save_graphs(dst, [graph], {'label': torch.tensor([label])})
            stats.append((graph.number_of_nodes(), time.time() - t0, float((np.asarray(superpixels) > 0).mean())))
            if len(stats) % 200 == 0:                # keep the sizes file current during long runs
                json.dump(sizes, open(sizes_path, 'w'))
        except Exception as e:
            print('Warning: {} failed during graph generation: {}'.format(im, e))
            failed.append(im)
    json.dump(sizes, open(sizes_path, 'w'))
    n_ok = len(glob(os.path.join(out_dir, '*.bin')))
    print('{} images in this run, {} graphs written now, {} graphs in {}, {} failed'.format(
        len(rows), len(stats), n_ok, out_dir, len(failed)))
    for im in failed:
        print('  failed:', im)
    if stats:
        n, s, cov = (np.array(v) for v in zip(*stats))
        print('nodes per graph: median {:.0f}, min {}, max {} | seconds per image: median {:.2f} | '
              'pixels covered by regions: min {:.3f}, median {:.3f}'.format(
                  np.median(n), int(n.min()), int(n.max()), np.median(s), cov.min(), np.median(cov)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='folder with ISIC_xxxxxxx.jpg')
    ap.add_argument('--split', required=True, help='isic2018_lesion_split.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--trial', type=int, default=0, help='only K images per (class, split)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--stain_norm', action='store_true', help='Vahadane H&E normalisation as for BRACS/BACH')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if not 0 <= a.shard < a.nshards:
        raise SystemExit('--shard must be in 0..nshards-1')
    main(a)
