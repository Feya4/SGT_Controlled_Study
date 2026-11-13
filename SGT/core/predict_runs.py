#!/usr/bin/env python3
"""[sgt] Per-graph test predictions of finished runs, for the paired bootstrap in compare_runs.py.

For each run folder this loads the checkpoint selected on validation weighted F1
(model_best_val_weighted_f1_score.pt), rebuilds the model from the checkpoint itself (which [sgt] modules it
holds, FFN width) and from the flags recorded in test_results.txt, and writes test_predictions.csv
(graph, label, pred, logit_0..logit_6) next to test_results.txt. It also works for runs started before this
script existed, and it skips runs that already have the file unless --force is given.

    python predict_runs.py --tg_path "$OUT_MORPH/tissue_graphs/" --runs '../runs_sgtfull_val_seed*'
The graphs in --tg_path only need the node fields the model uses: OUT_MORPH works for every model,
OUT for CGT alone.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from grapher import Grapher
from dataloader import make_data_loader, set_label_map
from train_utils import parse_device

DEVICE = parse_device()
NODE_DIM = 514
CKPT = 'model_best_val_weighted_f1_score.pt'


def parse_results(path, metric='best_val_weighted_f1_score'):
    for line in open(path):
        parts = line.rstrip('\n').split('\t')
        if parts and parts[0] == metric:
            return dict(kv.split('=', 1) for kv in parts[1:] if '=' in kv)
    return {}


def pred_name(rule, split):
    """test_predictions.csv (validation-selected checkpoint on test, the original file),
    test_predictions_testrule.csv (test-selected checkpoint), val_predictions.csv, ..."""
    return '{}_predictions{}.csv'.format(split, '' if rule == 'val' else '_testrule')


def flags_from_checkpoint(sd, res):
    """Which [sgt] modules the checkpoint holds; the shuffle controls come from test_results.txt."""
    model = res.get('model', 'cgt').split('+')
    return dict(morph='concat' if 'morph_linear.weight' in sd else 'none',
                sgt_mase='area_encoder.weight' in sd, sgt_sre='sre_hop_encoder.weight' in sd,
                mase_shuffle='mshuf' in model or res.get('morph') == 'shuffle', sre_shuffle='pshuf' in model,
                ffn_dim=int(sd['layers.0.ffn.layer1.weight'].shape[0]),
                num_classes=int(sd['downstream_out_proj.weight'].shape[0]))


def build(args, config, flags):
    return Grapher(
        gnn_params=config['gnn_params'], classification_params=config['classification_params'],
        node_dim=NODE_DIM, n_layers=args.n_layers, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
        attention_dropout_rate=args.attention_dropout_rate, dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.intput_dropout_rate, weight_decay=args.weight_decay,
        dataset_name='BRACS', warmup_updates=args.warmup_updates, tot_updates=args.tot_updates,
        peak_lr=args.peak_lr, end_lr=args.end_lr, edge_type=args.edge_type, multi_hop_max_dist=20,
        flag=args.flag, flag_m=args.flag_m, flag_step_size=args.flag_step_size, **flags,
    ).to(DEVICE)


def main(args):
    with open(args.config_fpath) as f:
        config = yaml.safe_load(f)
    if args.all:   # everything the comparisons use: both checkpoint rules on test, and validation
        jobs = [('val', 'test'), ('test', 'test'), ('val', 'val')]
    else:
        jobs = [(args.checkpoint, args.split)]
    if args.task == 'bracs3':                       # models trained on the three-class task
        set_label_map([0, 0, 0, 1, 1, 2, 2])
    loaders = {}
    for _, split in jobs:
        if split not in loaders:
            ld = make_data_loader(tg_path=os.path.join(args.tg_path, split), batch_size=args.batch_size,
                                  load_in_ram=False, shuffle=False)
            loaders[split] = (ld, [Path(f).stem for f in ld.dataset.tg_fnames])   # sorted = loader order
    dirs = sorted({d for pat in args.runs for d in glob.glob(pat)})
    if not dirs:
        raise SystemExit('no run folders match {}'.format(args.runs))
    for d in dirs:
        for res_file in sorted(Path(d).glob('logs/*/test_results.txt')):
            run = res_file.parent
            res = parse_results(res_file)
            model, flags = None, None
            for rule, split in jobs:
                out = run / pred_name(rule, split)
                ckpt = run / 'model_best_{}_weighted_f1_score.pt'.format(rule)
                if out.exists() and not args.force:
                    print('exists  ', out)
                    continue
                if not ckpt.exists():
                    if not (args.all and rule == 'test'):   # runs without --test_every_epoch have no test-rule file
                        print('no checkpoint {}, skipped'.format(ckpt))
                    continue
                sd = torch.load(str(ckpt), map_location=DEVICE)
                n_out = int(sd['downstream_out_proj.weight'].shape[0])
                if n_out != {'bracs7': 7, 'bracs3': 3, 'bach4': 4, 'isic7': 7, 'cam2': 2}[args.task]:
                    raise SystemExit('{} predicts {} classes but --task is {}'.format(ckpt, n_out, args.task))
                if model is None:
                    flags = flags_from_checkpoint(sd, res)
                    model = build(args, config, flags)
                model.load_state_dict(sd)
                model.eval()
                seed = int(res['seed']) if res.get('seed', 'None') not in ('None', '') else 0
                torch.manual_seed(seed)                     # the shuffle controls permute at test time as well
                loader, names = loaders[split]
                logits, labels = [], []
                with torch.no_grad():
                    for batch in loader:
                        logits.append(model(*batch[:-1]).cpu())
                        labels.append(batch[-1].cpu())
                logits = torch.cat(logits).numpy()
                labels = torch.cat(labels).numpy()
                preds = logits.argmax(1)
                with open(str(out), 'w') as f:
                    f.write('graph,label,pred,' + ','.join('logit_{}'.format(k) for k in range(logits.shape[1])) + '\n')
                    for n, y, p, l in zip(names, labels, preds, logits):
                        f.write('{},{},{},{}\n'.format(n, int(y), int(p), ','.join('{:.5f}'.format(v) for v in l)))
                wf1 = f1_score(labels, preds, average='weighted')
                note = ''
                ref = parse_results(res_file, 'best_{}_weighted_f1_score'.format(rule)) if split == 'test' else {}
                shuffled = flags['mase_shuffle'] or flags['sre_shuffle'] or flags['morph'] == 'shuffle'
                if 'weighted_f1' in ref and not shuffled and abs(wf1 - float(ref['weighted_f1'])) > 2e-4:
                    note = '  <-- differs from test_results.txt ({})'.format(ref['weighted_f1'])
                print('{}  seed={} | {} wF1 {:.4f} mF1 {:.4f} acc {:.4f}{}'.format(
                    out, res.get('seed'), split, wf1, f1_score(labels, preds, average='macro'),
                    accuracy_score(labels, preds), note))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tg_path', required=True, help='.../tissue_graphs/ (OUT_MORPH works for every model)')
    ap.add_argument('--runs', nargs='+', required=True, help="run folders or globs, e.g. '../runs_val_seed*'")
    ap.add_argument('--split', default='test')
    ap.add_argument('--checkpoint', choices=['val', 'test'], default='val',
                    help="which checkpoint: chosen on validation (default) or on test (CGT rule)")
    ap.add_argument('--task', choices=['bracs7', 'bracs3', 'bach4', 'isic7', 'cam2'], default='bracs7',
                    help='label set of the runs: bracs3 for models trained on the three-class task')
    ap.add_argument('--all', action='store_true',
                    help='write test predictions of both checkpoints and validation predictions in one pass')
    ap.add_argument('--config_fpath', default='./config/bracs_grapher_7_classes_pna.yml')
    ap.add_argument('-b', '--batch_size', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    ap = Grapher.add_model_specific_args(ap)
    main(ap.parse_args())
