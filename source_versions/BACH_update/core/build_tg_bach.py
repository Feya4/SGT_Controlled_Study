#!/usr/bin/env python3
"""[sgt] Tissue graphs for the BACH microscopy images, built exactly as for BRACS.

Uses the BRACS builder (build_tg.TgBuilding) unchanged: Vahadane stain normalisation to the BRACS target image,
colour-merged superpixels, ResNet-34 region features (512-d), region adjacency graph. In addition stores, per
region, its area and solidity (convexity), needed by MASE, computed from the same superpixel map.

Input   <images>/{Normal,Benign,InSitu,Invasive}/*.tif        (BACH_data/images)
Output  <out>/all/<stem>.bin   DGL graph: feat (512 + location), centroid, area, solidity; label 0..3
        <out>/image_sizes.json {stem: [width, height]}
Labels  0 normal, 1 benign, 2 in situ carcinoma, 3 invasive carcinoma

    python build_tg_bach.py --images /workdir1.8t/fei27/CGT/GNNMD/BACH/BACH_data/images --out /path/to/bach_graphs
"""
import argparse
import json
import os
from glob import glob

import numpy as np
import torch
from dgl.data.utils import load_graphs, save_graphs
from PIL import Image
from skimage.measure import regionprops
from tqdm import tqdm

from build_tg import TgBuilding

CLASSES = ['Normal', 'Benign', 'InSitu', 'Invasive']


def region_attributes(superpixels):
    """area and solidity of every region, in label order (= node order of the region adjacency graph)"""
    props = sorted(regionprops(superpixels), key=lambda r: r.label)
    area = torch.tensor([float(r.area) for r in props])
    solidity = torch.tensor([float(r.solidity) for r in props])
    cent = torch.tensor([list(r.centroid) for r in props], dtype=torch.float32)   # (row, col) = (y, x)
    return area, solidity, cent


def main(a):
    files = []
    for k, c in enumerate(CLASSES):
        fs = sorted(glob(os.path.join(a.images, c, '*.tif')) + glob(os.path.join(a.images, c, '*.tiff')))
        if not fs:
            raise SystemExit('no images in {}'.format(os.path.join(a.images, c)))
        files += [(f, k) for f in fs]
    gt_path = os.path.join(a.images, 'microscopy_ground_truth.csv')
    if os.path.exists(gt_path):                    # official labels must agree with the class folders
        gt = dict([v.strip() for v in l.split(',')[:2]] for l in open(gt_path, encoding='utf-8-sig') if ',' in l)
        bad = [(os.path.basename(f), CLASSES[k], gt.get(os.path.basename(f))) for f, k in files
               if gt.get(os.path.basename(f)) != CLASSES[k]]
        if bad:
            raise SystemExit('folder label differs from microscopy_ground_truth.csv for {} images, e.g. {}'.format(len(bad), bad[:3]))
        print('labels checked against microscopy_ground_truth.csv: all {} agree'.format(len(files)))
    out_dir = os.path.join(a.out, 'all')
    os.makedirs(out_dir, exist_ok=True)
    sizes_path = os.path.join(a.out, 'image_sizes.json')
    sizes = json.load(open(sizes_path)) if os.path.exists(sizes_path) else {}
    builder = TgBuilding()
    failed, n_new = [], 0
    for f, label in tqdm(files, desc='BACH tissue graphs'):
        stem = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(out_dir, stem + '.bin')
        image = np.array(Image.open(f).convert('RGB'))
        sizes[stem] = [int(image.shape[1]), int(image.shape[0])]          # width, height
        if os.path.exists(dst) and not a.overwrite:
            continue
        try:
            image = builder.normalizer.process(image)
        except Exception:
            print('Warning: {} failed during stain normalization (kept unnormalised, as for BRACS).'.format(f))
        try:
            graph, superpixels = builder._build_tg(image)
            area, solidity, cent = region_attributes(superpixels)
            if area.shape[0] != graph.number_of_nodes():
                raise ValueError('{} regions but {} graph nodes'.format(area.shape[0], graph.number_of_nodes()))
            gc = graph.ndata['centroid'].float()
            # node order check: our region centroids must match the graph's, as (y, x) or (x, y)
            if not (torch.allclose(gc, cent, atol=1.0) or torch.allclose(gc, cent[:, [1, 0]], atol=1.0)):
                raise ValueError('region order differs from graph node order')
            graph.ndata['area'] = area
            graph.ndata['solidity'] = solidity
            save_graphs(dst, [graph], {'label': torch.tensor([label])})
            n_new += 1
        except Exception as e:
            print('Warning: {} failed during tissue graph generation: {}'.format(f, e))
            failed.append(f)
    json.dump(sizes, open(sizes_path, 'w'))
    n_ok = len(glob(os.path.join(out_dir, '*.bin')))
    print('{} images, {} graphs written now, {} graphs in {}, {} failed'.format(len(files), n_new, n_ok, out_dir, len(failed)))
    for f in failed:
        print('  failed:', f)
    if n_ok:
        g = load_graphs(os.path.join(out_dir, sorted(os.listdir(out_dir))[0]))[0][0]
        print('example graph: {} nodes, node fields {}'.format(
            g.number_of_nodes(), {k: tuple(v.shape) for k, v in g.ndata.items()}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='BACH_data/images with Normal/Benign/InSitu/Invasive')
    ap.add_argument('--out', required=True)
    ap.add_argument('--overwrite', action='store_true')
    main(ap.parse_args())
