#!/usr/bin/env python3
"""
[rerun-addition] Convert existing HACT tissue graphs (as used by SGT) into the input format of CGT, so that CGT
runs on exactly the same graphs, split and labels without rebuilding them from the images.

Input  <src>/{train,val,test}_official/BRACS_*.bin   DGL graphs with node fields feat (512-d ResNet-34)
                                                     and centroid (x, y in pixels at 40x)
       <src>/roi_image_sizes.json                   {roi_stem: [width, height]}
Output <out>/tissue_graphs/{train,val,test}/BRACS_*.bin
       same nodes and edges; feat = [feat (512) || centroid_x / width || centroid_y / height] (514-d, as
       histocartography's RAGGraphBuilder(add_loc_feats=True) used by CGT's build_tg.py); label {'label': idx}
       with idx in [N, PB, UDH, ADH, FEA, DCIS, IC] read from the file name (the order used by CGT).

    python convert_hact_graphs.py --src /path/to/data/tissue_graphs_official_morph --out /path/to/cgt_data
    python convert_hact_graphs.py ... --limit 20      # quick trial on 20 graphs per split
    python convert_hact_graphs.py ... --morph_bins    # [sgt] also store the morphology attributes:
        area_z   (log area - mean) / std, statistics of all training nodes (written to <out>/area_stats.json)
        conv     convexity (solidity) in [0, 1]
        area_bin, conv_bin   16 uniform bins of area_z over [-3, 3] and of conv over [0, 1]
"""
import argparse
import json
from pathlib import Path

import torch
import dgl
from dgl.data.utils import load_graphs, save_graphs

CLASSES = ['N', 'PB', 'UDH', 'ADH', 'FEA', 'DCIS', 'IC']


def centroid_order(files, sizes, n=200):
    """Return (0, 1) if centroids are (x, y) w.r.t. (width, height), (1, 0) if they are (y, x)."""
    fit_xy = fit_yx = 0
    for f in files[:n]:
        c = load_graphs(str(f))[0][0].ndata['centroid'].float()
        w, h = sizes[f.stem]
        fit_xy += int(bool((c[:, 0] <= w + 1).all() and (c[:, 1] <= h + 1).all()))
        fit_yx += int(bool((c[:, 0] <= h + 1).all() and (c[:, 1] <= w + 1).all()))
    print('centroid check on {} graphs: fits (x, y) {}  fits (y, x) {}'.format(min(n, len(files)), fit_xy, fit_yx))
    if fit_xy >= fit_yx:
        return (0, 1)
    print('-> centroids are stored as (y, x); they are swapped to (x, y)')
    return (1, 0)


def bin16(v, lo, hi):
    t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
    return (t * 16).long().clamp(max=15)


def area_stats(src, limit):
    files = sorted((src / 'train_official').glob('BRACS_*.bin'))
    if limit:
        files = files[:limit]
    logs = []
    for f in files:
        g = load_graphs(str(f))[0][0]
        if 'area' not in g.ndata or 'solidity' not in g.ndata:
            raise ValueError('{} has no area/solidity: use the *_morph graphs'.format(f.name))
        logs.append(torch.log(g.ndata['area'].float().clamp(min=1.0)))
    v = torch.cat(logs)
    return float(v.mean()), float(v.std())


def main(a):
    src, out = Path(a.src), Path(a.out)
    sizes = json.load(open(src / 'roi_image_sizes.json'))
    order = None
    if a.morph_bins:
        out.mkdir(parents=True, exist_ok=True)
        mu, sd = area_stats(src, a.limit)
        json.dump({'mean_log_area': mu, 'std_log_area': sd}, open(out / 'area_stats.json', 'w'), indent=2)
        print('log-area statistics of the training nodes: mean {:.4f}  std {:.4f}'.format(mu, sd))
    for split in ('train', 'val', 'test'):
        files = sorted((src / '{}_official'.format(split)).glob('BRACS_*.bin'))
        if a.limit:
            files = files[:a.limit]
        if not files:
            raise FileNotFoundError('no graphs in {}'.format(src / '{}_official'.format(split)))
        if order is None:
            order = centroid_order(files, sizes)
        dst_dir = out / 'tissue_graphs' / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        n_new = n_clip = 0
        counts = [0] * len(CLASSES)
        for f in files:
            label = CLASSES.index(f.stem.split('_')[2])
            counts[label] += 1
            dst = dst_dir / f.name
            if dst.exists() and not a.overwrite:
                continue
            gs, ld = load_graphs(str(f))
            g = gs[0]
            if 'label' in ld and int(ld['label'].view(-1)[0]) != label:
                raise ValueError('{}: stored label {} != label from file name {}'.format(
                    f.name, int(ld['label'].view(-1)[0]), label))
            feat = g.ndata['feat'].float()
            if feat.shape[1] == 514:          # location features already present: recompute them identically
                feat = feat[:, :512]
            elif feat.shape[1] != 512:
                raise ValueError('{}: expected 512-d node features, got {}'.format(f.name, feat.shape[1]))
            c = g.ndata['centroid'].float()[:, list(order)]
            w, h = sizes[f.stem]
            loc = c / torch.tensor([float(w), float(h)])
            if bool((loc < 0).any() or (loc > 1).any()):
                n_clip += 1
                loc = loc.clamp(0.0, 1.0)
            new = dgl.DGLGraph()
            new.add_nodes(g.number_of_nodes())
            s, d = g.edges()
            new.add_edges(s, d)
            new.ndata['feat'] = torch.cat([feat, loc], dim=1)
            new.ndata['centroid'] = c
            if a.morph_bins:
                area_z = (torch.log(g.ndata['area'].float().clamp(min=1.0)) - mu) / sd
                conv = g.ndata['solidity'].float()
                new.ndata['area_z'] = area_z
                new.ndata['conv'] = conv
                new.ndata['area_bin'] = bin16(area_z, -3.0, 3.0)
                new.ndata['conv_bin'] = bin16(conv, 0.0, 1.0)
            save_graphs(str(dst), [new], {'label': torch.tensor([label])})
            n_new += 1
        print('{:5s}: {} graphs ({} written) -> {} | {} | graphs with centroids clipped to the image: {}'.format(
            split, len(files), n_new, dst_dir, '  '.join('{}={}'.format(k, v) for k, v in zip(CLASSES, counts)),
            n_clip))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='folder with {train,val,test}_official and roi_image_sizes.json')
    p.add_argument('--out', required=True, help='output root; graphs go to <out>/tissue_graphs/<split>/')
    p.add_argument('--limit', type=int, default=0, help='convert only the first N graphs per split (trial)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--morph_bins', action='store_true', help='[sgt] also store the morphology attributes')
    main(p.parse_args())
