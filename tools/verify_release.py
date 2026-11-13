#!/usr/bin/env python3
"""Reproducibility metadata/static preflight (stdlib; works with Python >=3.7).

This verifies released manifest *integrity* and source completeness, NOT numerical training reproducibility.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'BRACS': ROOT / 'splits/BRACS/bracs_split.csv',
    'BACH': ROOT / 'splits/BACH/folds.json',
    'ISIC2018': ROOT / 'splits/ISIC2018/isic2018_lesion_split.csv',
    'CAMELYON16': ROOT / 'splits/CAMELYON16/split.csv',
}
EXPECTED = {
    'BRACS': {'train': 3155, 'val': 602, 'test': 626},
    'ISIC2018': {'train': 7047, 'val': 1468, 'test': 1500},
    'CAMELYON16': {'train': 7956, 'val': 1821, 'test': 2247},
}


def check_partition(dataset, identity, group):
    f = FILES[dataset]
    with f.open(newline='', encoding='utf-8-sig') as h:
        rows = list(csv.DictReader(h))
    assert rows, '{}: no rows'.format(dataset)
    assert all(identity in r and group in r and 'split' in r for r in rows), '{}: missing columns'.format(dataset)
    counts = collections.Counter(r['split'] for r in rows)
    assert dict(counts) == EXPECTED[dataset], '{} counts {} != {}'.format(dataset, dict(counts), EXPECTED[dataset])
    ids = [r[identity].strip() for r in rows]
    assert len(ids) == len(set(ids)), '{} duplicate IDs'.format(dataset)
    g2s = collections.defaultdict(set)
    for r in rows:
        assert r[group].strip(), '{}: missing group'.format(dataset)
        g2s[r[group].strip()].add(r['split'])
    shared = {g: s for g, s in g2s.items() if len(s) > 1}
    assert not shared, '{} group leakage {}'.format(dataset, list(shared.items())[:5])
    if dataset == 'CAMELYON16':
        assert all(str(r['level']).strip() == '2' for r in rows), 'CAMELYON16 mixed levels'
        assert {k: len({r['group'] for r in rows if r['split'] == k}) for k in EXPECTED[dataset]} == {'train': 9, 'val': 2, 'test': 3}, 'CAMELYON16 group counts'
    print('[PASS] {:11s} samples={} split={} group_count={} group_overlap=0'.format(dataset, len(rows), dict(counts), len(g2s)))


def check_bach():
    f = FILES['BACH']
    assignment = json.loads(f.read_text(encoding='utf-8'))
    assert len(assignment) == 400, 'BACH expected 400 images'
    assert len(set(assignment)) == 400, 'BACH duplicate images'
    counts = collections.Counter(int(v) for v in assignment.values())
    assert dict(counts) == {0: 83, 1: 65, 2: 73, 3: 94, 4: 85}, 'BACH fold sizes {}'.format(dict(counts))
    info = json.loads((f.parent / 'folds_info.json').read_text(encoding='utf-8'))
    assert int(info['images_identified']) == 145 and int(info['patients_identified']) == 29, 'BACH metadata changed'
    assert int(info['patients_split_across_folds']) == 0, 'BACH identified-patient overlap'
    print('[PASS] BACH        400 images, test fold sizes {}, 29 identified patients across 145 images'.format(dict(counts)))
    print('[NOTE] Patient independence is unverified for remaining 255 BACH images')


def check_code():
    py_files = list((ROOT / 'SGT/core').rglob('*.py')) + list((ROOT / 'tools').rglob('*.py'))
    for f in py_files:
        ast.parse(f.read_text(encoding='utf-8'), filename=str(f))
    for f in [ROOT/'SGT/core/config/bracs_grapher_7_classes_pna.yml', ROOT/'SGT/core/build_tg_camelyon.py', ROOT/'SGT/core/make_camelyon_split.py', ROOT/'SGT/core/train.py', ROOT/'SGT/run_cgt_bracs.sh']:
        assert f.is_file(), 'missing source file {}'.format(f)
    train=(ROOT/'SGT/core/train.py').read_text(encoding='utf-8')
    launcher=(ROOT/'SGT/run_cgt_bracs.sh').read_text(encoding='utf-8')
    assert "'cam2': 2" in train and 'cam_graphs)' in launcher and 'cam_split)' in launcher, 'CAMELYON16 task not integrated'
    assert "default='val'" in train and 'ALLOW_TEST_SELECTION' in train, 'unsafe default checkpoint selection'
    print('[PASS] Source code syntax: {} Python modules; CAMELYON16 task and validation default present'.format(len(py_files)))


def verify_predictions(run_root):
    """Optional ID-only crosscheck on actual training runs; requires local user-provided output."""
    from fnmatch import fnmatch
    expected = {}
    for dataset, field in [('BRACS','roi'),('ISIC2018','image'),('CAMELYON16','sample_id')]:
        with FILES[dataset].open(newline='', encoding='utf-8-sig') as h:
            expected[dataset] = {r[field] for r in csv.DictReader(h) if r['split'] == 'test'}
    assignment=json.loads(FILES['BACH'].read_text(encoding='utf-8'))
    patterns = {
        'BRACS': ('runs_*c3_e100_val_seed*','runs_c3_e100_val_seed*'),
        'BACH': ('runs_*bach_f*_e100_val_seed*',),
        'ISIC2018': ('runs_*isic_e100_val_seed*',),
        'CAMELYON16': ('runs_*cam_e100_val_seed*',),
    }
    passes=collections.Counter()
    for folder in sorted(run_root.iterdir()):
        if not folder.is_dir():
            continue
        ds = next((d for d, ps in patterns.items() if any(fnmatch(folder.name, q) for q in ps)), None)
        if ds is None: continue
        fs=list(folder.glob('logs/**/test_predictions.csv'))
        assert len(fs)==1,'{} prediction files {}'.format(folder.name,len(fs))
        with fs[0].open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
        ids=[r['graph'] for r in rows]
        if ds=='BACH':
            import re
            m=re.search(r'bach_f([0-4])_',folder.name)
            assert m, 'No BACH fold in {}'.format(folder.name)
            fold=int(m.group(1))
            target={x for x,v in assignment.items() if int(v)==fold}
        else:target=expected[ds]
        assert len(ids)==len(target) and len(set(ids))==len(target) and set(ids)==target,'{} prediction ID mismatch'.format(folder.name)
        passes[ds]+=1
    print('[PASS] prediction identity checks:',dict(passes))
    return passes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, help='optional root with complete original runs*/logs/**/test_predictions.csv')
    a=ap.parse_args()
    for ds, identity, group in [('BRACS','roi','slide'),('ISIC2018','image','group'),('CAMELYON16','sample_id','group')]:
        check_partition(ds,identity,group)
    check_bach();check_code()
    if a.runs:verify_predictions(a.runs)
    else: print('[NOTE] No saved predictions supplied; IDs matched to runs are not re-checked by this audit')
    print('[NOTE] Does not test CUDA/DGL execution or verify numerical reproducibility')


if __name__=='__main__':
    try:main()
    except (AssertionError, FileNotFoundError, ValueError, KeyError) as e:
        print('[FAIL]',str(e),file=sys.stderr);sys.exit(1)
