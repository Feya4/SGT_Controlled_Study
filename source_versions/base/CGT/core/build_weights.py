#!/usr/bin/env python3
"""
[rerun-addition] Build the CGT "graph weight" files (weights/<split>/<name>.pt).

The released code loads, for every tissue graph, an N x N integer matrix `w`
that is fed to nn.Embedding(512, num_heads) as the connectivity bias, but the
script that creates it is not in the repository (the authors share the files
on request).

The paper (TMI 2024, Eq. 9) defines the bias from the Euclidean distance between
node centroids. The released checkpoint (logs/model_best_val.pt) shows that only
embedding rows 1..138 were ever trained, i.e. w took values 0..137. This matches
    w_ij = floor(100 * || c_i - c_j ||_2)
with c the image-size-normalised centroids in [0,1]^2 (max = 100*sqrt(2) = 141),
which histocartography stores as the last two node features (add_loc_feats=True).
This is a reconstruction, not the authors' script; the defaults can be changed
with --scale / --rounding if the official files turn out to differ.

Usage (from core/):
    python build_weights.py --tg_root /path/to/cgt_data
    # expects /path/to/cgt_data/tissue_graphs/{train,val,test}/*.bin
    # writes  /path/to/cgt_data/weights/{train,val,test}/*.pt
"""
import os
import argparse
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from dgl.data.utils import load_graphs

FEAT_KEY = 'feat'          # histocartography GNN_NODE_FEAT_IN
MAX_W = 510                # pad_2d_w adds +1, embedding has 512 rows (0 = padding)


def compute_w(norm_centroids, scale=100.0, rounding='floor'):
    """norm_centroids: (N, 2) tensor in [0, 1]. Returns (N, N) LongTensor."""
    c = norm_centroids.double()
    d = torch.cdist(c, c, compute_mode='donot_use_mm_for_euclid_dist') * scale
    d = torch.round(d) if rounding == 'round' else torch.floor(d)
    return d.clamp_(0, MAX_W).long()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tg_root', required=True,
                   help='folder that contains tissue_graphs/<split>/*.bin')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--scale', type=float, default=100.0)
    p.add_argument('--rounding', choices=['floor', 'round'], default='floor')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    root = os.path.abspath(args.tg_root.rstrip('/'))
    if os.path.basename(root) == 'tissue_graphs':
        root = os.path.dirname(root)

    for split in args.splits:
        tg_dir = os.path.join(root, 'tissue_graphs', split)
        out_dir = os.path.join(root, 'weights', split)
        fnames = sorted(glob(os.path.join(tg_dir, '*.bin')))
        if not fnames:
            print('[skip] no graphs in {}'.format(tg_dir))
            continue
        os.makedirs(out_dir, exist_ok=True)
        max_vals, n_nodes = [], []
        for fn in tqdm(fnames, desc='weights/{}'.format(split)):
            out = os.path.join(out_dir, os.path.basename(fn).replace('.bin', '.pt'))
            if os.path.isfile(out) and not args.overwrite:
                continue
            g = load_graphs(fn)[0][0]
            feat = g.ndata[FEAT_KEY]
            if feat.shape[1] != 514:
                raise ValueError('{}: expected 514-d node features (512 ResNet34 + 2 normalised '
                                 'centroids), got {}'.format(fn, feat.shape[1]))
            w = compute_w(feat[:, -2:], args.scale, args.rounding)
            torch.save(w, out)
            max_vals.append(int(w.max()))
            n_nodes.append(int(w.shape[0]))
        if max_vals:
            print('{}: {} files | nodes median {} max {} | w max overall {} (checkpoint used 0..137)'.format(
                split, len(max_vals), int(np.median(n_nodes)), max(n_nodes), max(max_vals)))


if __name__ == '__main__':
    main()
