#!/usr/bin/env python3
"""
Script for training CGT
"""
import torch
import os
import re
import uuid
import yaml
from tqdm import tqdm
try:
    import mlflow
    import mlflow.pytorch
except ImportError:  # [rerun-fix] mlflow is only needed with --logger mlflow
    mlflow = None
import numpy as np
import pandas as pd
import shutil
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

from histocartography.ml import CellGraphModel, TissueGraphModel, HACTModel
from grapher import Grapher

from dataloader import make_data_loader


import matplotlib
matplotlib.use('Agg')  # [rerun-fix] never open GUI windows (plt.show() blocked under ToDesk)
import matplotlib.pyplot as plt
# [rerun-fix] unused: import pytorch_lightning as pl


from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
from train_utils import parse_device


# [rerun-fix] removed hard-coded CUDA_VISIBLE_DEVICES='0,1,2,3'; choose the GPU from the shell instead.
DEVICE = parse_device()

NODE_DIM = 514


def model_label(args):
    """[sgt] e.g. 'cgt', 'mase', 'mase+sre', 'mase+sre+mshuf', 'sre+concat' (morph real/shuffle = MASE)."""
    mase = args.sgt_mase or args.morph in ('real', 'shuffle')
    mshuf = args.mase_shuffle or args.morph == 'shuffle'
    parts = [k for k, v in (('mase', mase), ('sre', args.sgt_sre), ('concat', args.morph == 'concat'),
                            ('mshuf', mshuf), ('pshuf', args.sre_shuffle)) if v]
    return '+'.join(parts) or 'cgt'


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.',
        default=None,
        required=False
    )
    
    parser.add_argument(
        '--tg_path',
        type=str,
        help='path to tissue graphs.',
        default='../../dataset/hact-net-data/tissue_graphs/',
        required=False
    )
    
    parser.add_argument(
        '--assign_mat_path',
        type=str,
        help='path to the assignment matrices.',
        default=None,
        required=False
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='./config/bracs_grapher_7_classes_pna.yml',
        required=False
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='path to where the model is saved.',
        default='',
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='batch size.',
        default=4,
        required=False
    )
    parser.add_argument(
        '--epochs', type=int, help='epochs.', default=200, required=False
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='learning rate.',
        default=1e-4,
        required=False
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='path to where the output data are saved (currently only for the interpretability).',
        default='../../data/graphs',
        required=False
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        help='path to saved model',
        default=None,
        required=False
    )
    parser.add_argument(
        '--logger',
        type=str,
        help='Logger type. Options are "mlflow" or "none"',
        required=False,
        default='none'
    )
    # [rerun-fix] the original selected checkpoints on the *test* split
    # (val_dataloader read tissue_graphs/test). Default keeps that behaviour so
    # the run matches the released code; use --val_split val for a clean protocol.
    parser.add_argument(
        '--val_split',
        type=str,
        choices=['val', 'test'],
        default='test',
        help='split used for checkpoint selection (original code: test).'
    )
    parser.add_argument('--seed', type=int, default=None, help='random seed (original: none).')
    parser.add_argument('--init_from', type=str, default=None,
                        help='[sgt] start from these weights, e.g. the released CGT checkpoint core/logs/model_best_val.pt. '
                             'Modules the file does not contain (MASE, SRE) start at zero, so the starting model is '
                             'exactly the checkpoint. Default: random initialisation, as in CGT.')
    parser.add_argument('--test_every_epoch', action='store_true',
                        help='[sgt] also evaluate the test split after every epoch and keep the checkpoint with the '
                             'best test weighted F1 (the checkpoint rule of the released CGT code). Reported as '
                             'best_test_weighted_f1_score next to the validation-selected results, which are unchanged.')
    parser.add_argument('--smoke_batches', type=int, default=0,
                        help='[sgt] check run: 1 epoch on the first N batches of each split, then exit.')
    parser.add_argument('--sgt_mase', action='store_true',
                        help='[sgt] MASE: area and convexity embeddings join CGT connectivity embedding (input + LCA)')
    parser.add_argument('--sgt_sre', action='store_true',
                        help='[sgt] SRE: per-head hop-distance bias next to CGT distance bias in self-attention')
    parser.add_argument('--mase_shuffle', action='store_true', help='[sgt] control: MASE bins permuted within each graph')
    parser.add_argument('--sre_shuffle', action='store_true', help='[sgt] control: SRE hop matrix permuted within each graph')
    parser.add_argument('--morph', type=str, default='none', choices=['none', 'real', 'shuffle', 'concat'],
                        help="[sgt] 'concat' = control for MASE (raw area/convexity through one linear layer, use "
                             "--ffn_dim 259); 'real' / 'shuffle' = old names of --sgt_mase / --sgt_mase --mase_shuffle. "
                             "MASE and the controls need graphs converted with --morph_bins.")

    parser = Grapher.add_model_specific_args(parser)

    return parser.parse_args()


def main(args):
    """
    Train CGT.
    Args:
        args (Namespace): parsed arguments.
    """

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.val_split == 'test':
        print('NOTE: checkpoint selection uses the TEST split (as in the released code). '
              'Pass --val_split val for selection on the validation split.')

    if args.config_fpath is not None:
        with open(args.config_fpath, 'r') as f:
            config = yaml.safe_load(f)

    if args.logger == 'mlflow':
        if mlflow is None:
            raise ImportError('--logger mlflow requested but mlflow is not installed (pip install mlflow).')
        mlflow.log_params({
            'batch_size': args.batch_size
        })
        df = pd.json_normalize(config)
        rep = {"graph_building.": "", "model_params.": "", "gnn_params.": ""}  
        for i, j in rep.items():
            df.columns = df.columns.str.replace(i, j)
        flatten_config = df.to_dict(orient='records')[0]
        for key, val in flatten_config.items():
            mlflow.log_params({key: str(val)})

    
    model_path = os.path.join(args.model_path, "logs", str(uuid.uuid4()))
    os.makedirs(model_path, exist_ok=True)
    writer = SummaryWriter(model_path)

    
    train_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'train') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'train') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'train') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    val_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, args.val_split) if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, args.val_split) if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, args.val_split) if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    test_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'test') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'test') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    # [sgt] per-epoch test evaluation (CGT checkpoint rule). Unshuffled, so it draws no random numbers and the
    # training run is the same as without --test_every_epoch.
    test_eval_dataloader = None
    if args.test_every_epoch:
        test_eval_dataloader = make_data_loader(
            cg_path=os.path.join(args.cg_path, 'test') if args.cg_path is not None else None,
            tg_path=os.path.join(args.tg_path, 'test') if args.tg_path is not None else None,
            assign_mat_path=os.path.join(args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
            batch_size=args.batch_size,
            load_in_ram=args.in_ram,
            shuffle=False,
        )
    if args.smoke_batches > 0:  # [sgt] quick end-to-end check of a configuration
        import itertools
        args.epochs = 1
        train_dataloader = list(itertools.islice(train_dataloader, args.smoke_batches))
        val_dataloader = list(itertools.islice(val_dataloader, args.smoke_batches))
        test_dataloader = list(itertools.islice(test_dataloader, args.smoke_batches))
        if test_eval_dataloader is not None:
            test_eval_dataloader = list(itertools.islice(test_eval_dataloader, args.smoke_batches))

    
    if 'bracs_grapher' in args.config_fpath:
        model = Grapher(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name='BRACS',
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=20,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
            num_classes=7,
            morph=args.morph,
            sgt_mase=args.sgt_mase,
            sgt_sre=args.sgt_sre,
            mase_shuffle=args.mase_shuffle,
            sre_shuffle=args.sre_shuffle,
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('parameters: {:,} | model={} morph={} sgt_mase={} sgt_sre={} mase_shuffle={} sre_shuffle={} '
              'ffn_dim={}'.format(n_params, model_label(args), args.morph, args.sgt_mase, args.sgt_sre,
                                  args.mase_shuffle, args.sre_shuffle, args.ffn_dim))
    else:
        raise ValueError('Model type not recognized. Options are: TG, CG or HACT.')


    if args.init_from:  # [sgt] fine-tuning from trained weights; added modules start at zero (= the checkpoint)
        sd = torch.load(args.init_from, map_location=DEVICE)
        own = model.state_dict()
        unexpected = [k for k in sd if k not in own]
        missing = [k for k in own if k not in sd]
        # a wider FFN (parameter-matched concatenation control, --ffn_dim 259) keeps the checkpoint's units; the
        # extra units get random input weights and zero output weights, so the starting model is unchanged
        widened = []
        for k in list(sd):
            if k in own and tuple(sd[k].shape) != tuple(own[k].shape) and re.match(r'layers\.\d+\.ffn\.layer[12]\.', k):
                src, dst = sd[k], own[k].clone()
                if k.endswith('layer1.weight') and dst.shape[1] == src.shape[1] and dst.shape[0] > src.shape[0]:
                    dst.normal_(0.0, 0.02); dst[:src.shape[0]] = src
                elif k.endswith('layer1.bias') and dst.shape[0] > src.shape[0]:
                    dst.zero_(); dst[:src.shape[0]] = src
                elif k.endswith('layer2.weight') and dst.shape[0] == src.shape[0] and dst.shape[1] > src.shape[1]:
                    dst.zero_(); dst[:, :src.shape[1]] = src
                else:
                    continue
                sd[k] = dst
                widened.append(k)
        mismatch = [k for k in sd if k in own and tuple(sd[k].shape) != tuple(own[k].shape)]
        allowed = ('area_encoder.', 'conv_encoder.', 'sre_hop_encoder.', 'morph_linear.')
        if unexpected or mismatch or [k for k in missing if not k.startswith(allowed)]:
            raise ValueError('{} does not fit this model: unexpected {} | shape mismatch {} | missing {}'.format(
                args.init_from, unexpected, mismatch, missing))
        if widened:
            print('widened FFN to {} units in {} tensors; extra units start with zero output weights'.format(
                args.ffn_dim, len(widened)))
        model.load_state_dict(sd, strict=False)
        params = dict(model.named_parameters())
        with torch.no_grad():
            for k in missing:
                params[k].zero_()
        print('initialised from {} | added modules set to zero: {}'.format(args.init_from, missing or 'none'))

    '''
    Compute params and flops
    '''
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameter: {param}")

    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    '''define loss function'''
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.init_from:  # [sgt] score of the starting weights, before any training
        def _wf1(loader):
            model.eval()
            lg, lb = [], []
            for batch in loader:
                with torch.no_grad():
                    lg.append(model(*batch[:-1]))
                lb.append(batch[-1])
            return f1_score(torch.cat(lb).cpu().numpy(), torch.argmax(torch.cat(lg).cpu(), dim=1).numpy(),
                            average='weighted')
        msg = 'before training: {} wF1 {:.4f}'.format(args.val_split, _wf1(val_dataloader))
        if test_eval_dataloader is not None:
            msg += ' | test wF1 {:.4f}'.format(_wf1(test_eval_dataloader))
        print(msg)

    
    step = 0
    best_val_loss = 10e5
    best_val_accuracy = 0.
    best_val_weighted_f1_score = 0.
    best_val_wf1_epoch = -1
    best_test_wf1, best_test_wf1_epoch, all_test_weighted_f1_score = -1.0, -1, []  # [sgt] CGT checkpoint rule

    
    all_train_loss = []
    all_train_accuracy = []
    all_train_weighted_f1_score = []

    all_val_loss = []
    all_val_accuracy = []
    all_val_weighted_f1_score = []

    for epoch in range(args.epochs):
        
        
        model.train()
        epoch_loss = 0
        all_train_logits = []
        all_train_labels = []
        for batch in tqdm(train_dataloader, desc='Epoch training {}'.format(epoch), unit='batch'):
            
            labels = batch[-1]
            data = batch[:-1]

            logits = model(*data)

            all_train_logits.append(logits)
            all_train_labels.append(labels)

            
            loss = loss_fn(logits, labels)
            

            epoch_loss += loss.cpu().detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            if args.smoke_batches > 0 and step == 0:  # [sgt] every added module must receive a gradient
                assert torch.isfinite(loss).item(), 'non-finite loss'
                for name, prm in model.named_parameters():
                    if name.startswith(('area_encoder', 'conv_encoder', 'sre_', 'morph_')) and name.endswith('weight'):
                        gn = float('nan') if prm.grad is None else prm.grad.norm().item()
                        print('  grad {:28s} {:.3e}'.format(name, gn))
                        assert prm.grad is not None and np.isfinite(gn), 'no/invalid gradient for ' + name
            optimizer.step()

            
            if args.logger == 'mlflow':
                mlflow.log_metric('train_loss', loss.item(), step=step)

            
            step += 1

        
        train_loss = epoch_loss/len(train_dataloader)
        all_train_loss.append(train_loss)
        
        all_train_logits = torch.cat(all_train_logits).cpu()
        all_train_preds = torch.argmax(all_train_logits, dim=1).detach().numpy()
        all_train_labels = torch.cat(all_train_labels).cpu().detach().numpy()

        epoch_train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        epoch_train_weighted_f1_score = f1_score(all_train_labels, all_train_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('train_loss', train_loss, step=step)
            mlflow.log_metric('train_accuracy', epoch_train_accuracy, step=step)
            mlflow.log_metric('train_weighted_f1_score', epoch_train_weighted_f1_score, step=step)
        all_train_accuracy.append(epoch_train_accuracy)
        all_train_weighted_f1_score.append(epoch_train_weighted_f1_score)
        print('train loss: {} | train weighted F1: {} | train accuracy: {}'.format(train_loss, epoch_train_weighted_f1_score, epoch_train_accuracy))

        '''Log of each Epoch'''
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/Weighted_F1_score', epoch_train_weighted_f1_score, epoch)
        writer.add_scalar('Training/Accuracy', epoch_train_accuracy, epoch)
        writer.add_scalar('Training/Lr', optimizer.param_groups[0]['lr'], epoch)

        if scheduler is not None:
            scheduler.step()

        
        model.eval()
        all_val_logits = []
        all_val_labels = []
        for batch in tqdm(val_dataloader, desc='Epoch validation {}'.format(epoch), unit='batch'):
            labels = batch[-1]
            data = batch[:-1]
            with torch.no_grad():
                logits = model(*data)
            all_val_logits.append(logits)
            all_val_labels.append(labels)

        all_val_logits = torch.cat(all_val_logits).cpu()
        all_val_preds = torch.argmax(all_val_logits, dim=1)
        all_val_labels = torch.cat(all_val_labels).cpu()

        
        with torch.no_grad():
            loss = loss_fn(all_val_logits, all_val_labels).item()
            all_val_loss.append(loss)
        if args.logger == 'mlflow':
            mlflow.log_metric('val_loss', loss, step=step)
        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_loss.pt'))

        
        all_val_preds = all_val_preds.detach().numpy()
        all_val_labels = all_val_labels.detach().numpy()
        accuracy = accuracy_score(all_val_labels, all_val_preds)
        if args.logger == 'mlflow':
            mlflow.log_metric('val_accuracy', accuracy, step=step)
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_accuracy.pt'))

        
        weighted_f1_score = f1_score(all_val_labels, all_val_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('val_weighted_f1_score', weighted_f1_score, step=step)
        if weighted_f1_score > best_val_weighted_f1_score:
            best_val_weighted_f1_score = weighted_f1_score
            best_val_wf1_epoch = epoch  # [rerun-fix] recorded for the results file
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_weighted_f1_score.pt'))

        all_val_weighted_f1_score.append(weighted_f1_score)
        all_val_accuracy.append(accuracy)

        print('val loss: {} | val weighted F1: {} | val accuracy: {}'.format(loss, weighted_f1_score, accuracy))

        if test_eval_dataloader is not None:  # [sgt] CGT checkpoint rule, kept next to the validation rule
            t_logits, t_labels = [], []
            for batch in tqdm(test_eval_dataloader, desc='Epoch test {}'.format(epoch), unit='batch'):
                with torch.no_grad():
                    t_logits.append(model(*batch[:-1]))
                t_labels.append(batch[-1])
            t_preds = torch.argmax(torch.cat(t_logits).cpu(), dim=1).numpy()
            t_wf1 = f1_score(torch.cat(t_labels).cpu().numpy(), t_preds, average='weighted')
            all_test_weighted_f1_score.append(t_wf1)
            if t_wf1 > best_test_wf1:
                best_test_wf1, best_test_wf1_epoch = t_wf1, epoch
                torch.save(model.state_dict(), os.path.join(model_path, 'model_best_test_weighted_f1_score.pt'))
            print('test wF1 (CGT rule): {:.4f} | best so far {:.4f} at epoch {}'.format(t_wf1, best_test_wf1,
                                                                                       best_test_wf1_epoch))

        '''Log of each Epoch'''
        writer.add_scalar('Val/Loss', loss, epoch)
        writer.add_scalar('Val/Weighted_F1_score', weighted_f1_score, epoch)
        writer.add_scalar('Val/Accuracy', accuracy, epoch)

    '''
    All train loops are finished.
    '''
    
    # [rerun-fix] plt.show() -> savefig, so an attached display cannot block the run.
    plt.figure()
    plt.title('Loss vs. epochs')
    plt.plot(range(args.epochs), all_train_loss, label='Training loss')
    plt.plot(range(args.epochs), all_val_loss, label='Val loss')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'loss_curves.png'), dpi=150)
    plt.close()

    plt.figure()
    plt.title('Metrics vs. epochs')
    plt.plot(range(args.epochs), all_train_weighted_f1_score, label='Training F1')
    plt.plot(range(args.epochs), all_train_accuracy, label='Training accuracy')
    plt.plot(range(args.epochs), all_val_weighted_f1_score, label='Val F1')
    plt.plot(range(args.epochs), all_val_accuracy, label='Val accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'metric_curves.png'), dpi=150)
    plt.close()
    history = {
        'epoch': list(range(args.epochs)),
        'train_loss': all_train_loss, 'train_acc': all_train_accuracy, 'train_wf1': all_train_weighted_f1_score,
        'val_loss': all_val_loss, 'val_acc': all_val_accuracy, 'val_wf1': all_val_weighted_f1_score,
    }
    if all_test_weighted_f1_score:
        history['test_wf1'] = all_test_weighted_f1_score
    pd.DataFrame(history).to_csv(os.path.join(model_path, 'history.csv'), index=False)

    
    model.eval()

    best_F1 = 0
    metrics = ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']
    if args.test_every_epoch:
        metrics.append('best_test_weighted_f1_score')   # [sgt] CGT checkpoint rule
    for metric in metrics:
        
        print('\n*** Start testing w/ {} model ***'.format(metric))

        model_name = [f for f in os.listdir(model_path) if f.endswith(".pt") and metric in f][0]
        model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

        all_test_logits = []
        all_test_labels = []
        for batch in tqdm(test_dataloader, desc='Testing: {}'.format(metric), unit='batch'):
            labels = batch[-1]
            data = batch[:-1]
            with torch.no_grad():
                logits = model(*data)
            all_test_logits.append(logits)
            all_test_labels.append(labels)

        all_test_logits = torch.cat(all_test_logits).cpu()
        all_test_preds = torch.argmax(all_test_logits, dim=1)
        all_test_labels = torch.cat(all_test_labels).cpu()

        
        with torch.no_grad():
            loss = loss_fn(all_test_logits, all_test_labels).item()
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_loss_' + metric, loss)

        
        all_test_preds = all_test_preds.detach().numpy()
        all_test_labels = all_test_labels.detach().numpy()
        accuracy = accuracy_score(all_test_labels, all_test_preds)
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_accuracy_' + metric, accuracy, step=step)

        
        weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_weighted_f1_score_' + metric, weighted_f1_score, step=step)

        '''
            compute and store the best classification report
        '''
        if weighted_f1_score > best_F1:
            best_F1 = weighted_f1_score
            report = classification_report(all_test_labels, all_test_preds, digits=2)
            out_path = os.path.join(model_path, 'classification_report.txt')
            with open(out_path, "w") as f:
                f.write(report)

        if args.logger == 'mlflow':
            artifact_path = 'evaluators/class_report_{}'.format(metric)
            mlflow.log_artifact(out_path, artifact_path=artifact_path)
            # [rerun-fix] was called unconditionally, even with --logger none.
            mlflow.pytorch.log_model(model, 'model_' + metric)

        print('Test loss: {} | Test weighted F1: {} | Test accuracy: {}'.format(loss, weighted_f1_score, accuracy))
        with open(os.path.join(model_path, 'test_results.txt'), 'a') as f:
            f.write('{}\tloss={:.4f}\tweighted_f1={:.4f}\tmacro_f1={:.4f}\taccuracy={:.4f}\tseed={}\t'
                    'val_split={}\tbest_val_wf1={:.4f}\tbest_val_wf1_epoch={}\tmorph={}\tmodel={}\tffn_dim={}\t'
                    'best_test_wf1_epoch={}\tinit={}\n'.format(
                metric, loss, weighted_f1_score, f1_score(all_test_labels, all_test_preds, average='macro'),
                accuracy, args.seed, args.val_split, best_val_weighted_f1_score, best_val_wf1_epoch, args.morph,
                model_label(args), args.ffn_dim, best_test_wf1_epoch,
                os.path.basename(args.init_from) if args.init_from else 'random'))
        with open(os.path.join(model_path, 'classification_report_{}.txt'.format(metric)), 'w') as f:
            f.write(classification_report(all_test_labels, all_test_preds, digits=4))

    print('\nAll outputs saved in: {}'.format(model_path))

    if args.logger == 'mlflow':
        shutil.rmtree(model_path)


if __name__ == "__main__":
    main(args=parse_arguments())
