#!/usr/bin/env python3
"""[sgt] Region graphs for the CAMELYON16-derived patches (Level 2, slide-disjoint split), built with the BRACS builder.

Same builder as for BRACS, BACH and ISIC (build_tg.TgBuilding): Vahadane H&E stain normalisation to the BRACS target,
colour-merged superpixels, ResNet-34 region features (512-d), region adjacency graph; per region also its area and
solidity (convexity) for MASE, with the node-order check of build_tg_bach.region_attributes.
Graph size: with the BRACS superpixel settings (size 500, colour-merging threshold 0.05) a 256 x 256 patch gives a
median of 3 regions. Superpixel size 62 and threshold 0.025 (the defaults here) give a median of 53 nodes on 24
training patches from 8 slide groups (BRACS: 50); all other settings are those of BRACS. The settings are stored in
<out>/builder_settings.json, and a later run into the same --out must use the same ones (no mixed graphs).

Input   --root   folder searched (recursively) for <sample_id>.npy, each a dict with 'image' (256 x 256 x 3, uint8)
        --split  split.csv: sample_id, label (Normal / Tumor), level, group, split
Output  <out>/all/<sample_id>.bin   DGL graph: feat, centroid, area, solidity; label 0 = Normal, 1 = Tumor
        <out>/image_sizes.json      {sample_id: [width, height]}   (image_sizes_shard<k>.json with --nshards > 1)
Options --trial K   only K patches per (label, split);   --shard k --nshards n   every n-th patch, starting at k
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
from tqdm import tqdm

from build_tg import TgBuilding
from build_tg_bach import region_attributes      # area, solidity, centroid per region, in node order

CLASSES = ['Normal', 'Tumor']


def read_split(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    bad = [r['sample_id'] for r in rows if r['label'].strip() not in CLASSES]
    if bad:
        raise SystemExit('unknown labels for {} patches, e.g. {}'.format(len(bad), bad[:3]))
    return [(r['sample_id'].strip(), CLASSES.index(r['label'].strip()), r['split'].strip()) for r in rows]


def index_patches(root, need):
    """first copy of every needed <sample_id>.npy below root (copies were checked to be identical)"""
    where = {}
    for f in sorted(glob(os.path.join(root, '**', '*.npy'), recursive=True)):
        s = os.path.basename(f)[:-4]
        if s in need and s not in where:
            where[s] = f
    return where


def superpixel_extractor(builder):
    ext = [v for v in vars(builder).values() if 'superpixel' in type(v).__name__.lower()]
    if len(ext) != 1:
        raise SystemExit('expected one superpixel extractor in TgBuilding, found {}'.format(len(ext)))
    return ext[0]


def main(a):
    rows = sorted(read_split(a.split))
    print('{}: {} patches | Normal={} Tumor={}'.format(os.path.basename(a.split), len(rows),
          sum(1 for _, y, _ in rows if y == 0), sum(1 for _, y, _ in rows if y == 1)))
    if a.trial:
        keep, seen = [], {}
        for r in rows:
            key = (r[1], r[2])
            if seen.get(key, 0) < a.trial:
                keep.append(r)
                seen[key] = seen.get(key, 0) + 1
        rows = keep
        print('trial: {} patches ({} per label and split)'.format(len(rows), a.trial))
    rows = rows[a.shard::a.nshards]
    where = index_patches(a.root, {r[0] for r in rows})
    missing = [s for s, _, _ in rows if s not in where]
    if missing:
        raise SystemExit('{} patches not found below {}, e.g. {}'.format(len(missing), a.root, missing[:3]))
    out_dir = os.path.join(a.out, 'all')
    os.makedirs(out_dir, exist_ok=True)
    builder = TgBuilding()
    ext = superpixel_extractor(builder)
    bracs = {'superpixel_size': ext.superpixel_size, 'merge_threshold': ext.threshold}
    ext.superpixel_size, ext.threshold = a.superpixel_size, a.merge_threshold
    settings = {'superpixel_size': a.superpixel_size, 'merge_threshold': a.merge_threshold,
                'stain_norm': not a.no_stain_norm, 'bracs_settings': bracs}
    sp = os.path.join(a.out, 'builder_settings.json')
    keys = ('superpixel_size', 'merge_threshold', 'stain_norm')
    if os.path.exists(sp):
        old = json.load(open(sp))
        if any(old.get(k) != settings[k] for k in keys) and not a.overwrite:
            raise SystemExit('{} holds graphs built with {}; use another --out (or --overwrite)'.format(
                a.out, {k: old.get(k) for k in keys}))
    json.dump(settings, open(sp, 'w'), indent=1)
    print('superpixels: size {} (BRACS {}), colour-merging threshold {} (BRACS {})'.format(
        a.superpixel_size, bracs['superpixel_size'], a.merge_threshold, bracs['merge_threshold']))
    print('stain normalisation: {}'.format('off' if a.no_stain_norm else 'Vahadane to the BRACS target (as BRACS/BACH)'))
    sizes_path = os.path.join(a.out, 'image_sizes.json' if a.nshards == 1 else 'image_sizes_shard{}.json'.format(a.shard))
    sizes = json.load(open(sizes_path)) if os.path.exists(sizes_path) else {}
    failed, stats = [], []
    for s, label, _ in tqdm(rows, desc='CAMELYON graphs{}'.format(
            ' shard {}/{}'.format(a.shard, a.nshards) if a.nshards > 1 else '')):
        dst = os.path.join(out_dir, s + '.bin')
        image = np.load(where[s], allow_pickle=True).item()['image']
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            failed.append(s)
            print('Warning: {} has image {} {}'.format(s, image.dtype, image.shape))
            continue
        sizes[s] = [int(image.shape[1]), int(image.shape[0])]          # width, height
        if os.path.exists(dst) and not a.overwrite:
            continue
        t0 = time.time()
        if not a.no_stain_norm:
            try:
                image = builder.normalizer.process(image)
            except Exception:
                print('Warning: {} failed during stain normalization (kept unnormalised, as for BRACS).'.format(s))
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
            if len(stats) % 500 == 0:
                json.dump(sizes, open(sizes_path, 'w'))
        except Exception as e:
            print('Warning: {} failed during graph generation: {}'.format(s, e))
            failed.append(s)
    json.dump(sizes, open(sizes_path, 'w'))
    n_ok = len(glob(os.path.join(out_dir, '*.bin')))
    print('{} patches in this run, {} graphs written now, {} graphs in {}, {} failed'.format(
        len(rows), len(stats), n_ok, out_dir, len(failed)))
    for s in failed:
        print('  failed:', s)
    if stats:
        n, t, cov = (np.array(v) for v in zip(*stats))
        print('nodes per graph: median {:.0f}, min {}, max {} | seconds per patch: median {:.2f} | '
              'pixels covered by regions: min {:.3f}, median {:.3f}'.format(
                  np.median(n), int(n.min()), int(n.max()), np.median(t), cov.min(), np.median(cov)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='folder searched recursively for <sample_id>.npy')
    ap.add_argument('--split', required=True, help='split.csv of the slide-disjoint Level-2 split')
    ap.add_argument('--out', required=True)
    ap.add_argument('--trial', type=int, default=0, help='only K patches per (label, split)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--superpixel_size', type=int, default=62, help='BRACS: 500')
    ap.add_argument('--merge_threshold', type=float, default=0.025, help='colour-merging threshold; BRACS: 0.05')
    ap.add_argument('--no_stain_norm', action='store_true', help='skip the Vahadane normalisation (default: applied)')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if not 0 <= a.shard < a.nshards:
        raise SystemExit('--shard must be in 0..nshards-1')
    main(a)
