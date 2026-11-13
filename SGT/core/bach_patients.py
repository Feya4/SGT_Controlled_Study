#!/usr/bin/env python3
"""[sgt] Read the BACH patient file (image -> patient), whatever its layout, with the standard library only.

The organisers provide the patient of only part of the microscopy images ("ICIAR2018_BACH_dataset_micro_patient",
.csv or .xlsx) and state that every unidentified image comes from a patient other than the identified ones; such
images are kept as their own group. Image names are recognised as n/b/is/iv + number, with or without a folder,
extension or zero padding (Benign/b001.tif, b001, B1 -> b001); ranges such as n001-n010 are expanded.

Layouts handled:
  * one row per image:     image, patient [, label ...]    (any column order; a 'patient' header is used if present)
  * one row per patient:   patient, image, image, ...      (or the images in one cell, separated by spaces/commas)
  * one column per patient: header = patient, images below

    python bach_patients.py --patients ICIAR2018_BACH_dataset_micro_patient.xlsx --images .../BACH_data/images
prints what was read (rows, layout, images matched, patients) without changing anything.
"""
import argparse
import csv
import io
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from glob import glob

PREFIX = {'n': 'n', 'b': 'b', 'is': 'is', 'iv': 'iv'}
STEM_RE = re.compile(r'^(n|b|is|iv)0*(\d{1,4})$', re.I)
RANGE_RE = re.compile(r'^(n|b|is|iv)0*(\d{1,4})-(?:(n|b|is|iv)0*)?(\d{1,4})$', re.I)
CLASS_WORDS = {'normal', 'benign', 'insitu', 'in situ', 'in-situ', 'invasive', 'in situ carcinoma',
               'invasive carcinoma', 'label', 'class'}
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def canon(token):
    """'Benign/b001.tif' -> 'b001'; None if the token is not an image name"""
    t = token.strip().replace('\\', '/').rsplit('/', 1)[-1]
    t = re.sub(r'\.(tiff?|png|jpe?g)$', '', t, flags=re.I)
    m = STEM_RE.match(t)
    return '{}{:03d}'.format(m.group(1).lower(), int(m.group(2))) if m else None


def images_in(cell):
    """all image names in one cell (several names or a range allowed)"""
    text = re.sub(r'(?<=\d)\s*(?:-|–|~|\bto\b)\s*(?=(?:n|b|is|iv)?\d)', '-', cell.strip(), flags=re.I)   # "n001 - n010"
    out = []
    for tok in re.split(r'[\s,;]+', text):
        if not tok:
            continue
        r = RANGE_RE.match(re.sub(r'\.(tiff?|png|jpe?g)', '', tok.replace('\\', '/').rsplit('/', 1)[-1], flags=re.I))
        if r and (r.group(3) is None or r.group(3).lower() == r.group(1).lower()):
            p, a, b = r.group(1).lower(), int(r.group(2)), int(r.group(4))
            if a <= b and b - a < 200:
                out += ['{}{:03d}'.format(p, i) for i in range(a, b + 1)]
                continue
        c = canon(tok)
        if c:
            out.append(c)
    return out


def _xlsx_rows(path):
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS):
            shared.append(''.join(t.text or '' for t in si.iter('{%s}t' % NS['m'])))
    sheets = sorted(n for n in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml$', n))
    rows = []
    for sh in sheets:
        for r in ET.fromstring(z.read(sh)).iter('{%s}row' % NS['m']):
            cells = {}
            for c in r.findall('m:c', NS):
                ref = re.match(r'([A-Z]+)', c.get('r', 'A'))
                col = 0
                for ch in ref.group(1):
                    col = col * 26 + ord(ch) - 64
                v = c.find('m:v', NS)
                if c.get('t') == 's' and v is not None:
                    text = shared[int(v.text)]
                elif c.get('t') == 'inlineStr':
                    text = ''.join(t.text or '' for t in c.iter('{%s}t' % NS['m']))
                else:
                    text = v.text if v is not None and v.text else ''
                    if re.match(r'^\d+\.0$', text):
                        text = text[:-2]
                cells[col - 1] = text
            if cells:
                rows.append([cells.get(i, '') for i in range(max(cells) + 1)])
    return rows


def read_table(path):
    with open(path, 'rb') as f:
        head = f.read(8)
    if head[:2] == b'PK':
        return _xlsx_rows(path), 'xlsx'
    if head == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        try:
            import xlrd
            book = xlrd.open_workbook(path)
            return [[str(c.value).rstrip('0').rstrip('.') if isinstance(c.value, float) else str(c.value)
                     for c in sh.row(i)] for sh in book.sheets() for i in range(sh.nrows)], 'xls'
        except ImportError:
            raise SystemExit('{} is an old Excel .xls file and xlrd is not installed: open it in WPS/Excel and '
                             'save it as CSV (or .xlsx), then use that file'.format(path))
    raw = open(path, 'rb').read()
    for enc in ('utf-8-sig', 'gbk', 'latin-1'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=',;\t')
    except csv.Error:
        dialect = csv.excel
    return [row for row in csv.reader(io.StringIO(text), dialect)], 'text'


def read_patients(path, stems=None, verbose=True):
    """{stem: patient id} for the images the file identifies (restricted to stems if given)"""
    rows, kind = read_table(path)
    rows = [[c.strip() for c in r] for r in rows if any(c.strip() for c in r)]
    first = [images_in(c) for c in rows[0]] if rows else []
    header = rows[0] if rows and not any(first) else None
    pcol = None
    if header:
        for j, h in enumerate(header):
            if re.search(r'patient|case|subject|slide', h, re.I) and not re.search(r'image|file|name', h, re.I):
                pcol = j
                break
    # the patient column never counts as images (a patient code such as "B12" would look like image b012)
    imgs = [[[] if j == pcol else images_in(c) for j, c in enumerate(r)] for r in rows]
    assign, conflicts, layout = {}, [], None

    def put(stem, pid, where):
        if stems is not None and stem not in stems:
            return
        if stem in assign and assign[stem] != pid:
            conflicts.append((stem, assign[stem], pid, where))
            return
        assign[stem] = pid

    start = 1 if header else 0
    labelled = any(c and not im for i in range(start, len(rows)) if any(imgs[i])
                   for c, im in zip(rows[i], imgs[i]))
    for i in (range(start, len(rows)) if labelled or not header else []):   # row-wise layouts
        found = [s for cell in imgs[i] for s in cell]
        if not found:
            continue
        if pcol is not None and pcol < len(rows[i]) and rows[i][pcol] and not imgs[i][pcol]:
            pid = rows[i][pcol]
        else:
            others = [c for c, im in zip(rows[i], imgs[i]) if c and not im and c.lower() not in CLASS_WORDS]
            pid = others[0] if others else (('row{}'.format(i)) if len(found) > 1 else None)
        if pid is None:
            continue
        layout = layout or ('one row per image' if len(found) == 1 else 'one row per patient')
        for s in found:
            put(s, pid, 'row {}'.format(i + 1))
    if not assign and header:                              # header of patients, images below: one column per patient
        layout = 'one column per patient'
        for j, h in enumerate(header):
            for i in range(1, len(rows)):
                if j < len(imgs[i]):
                    for s in imgs[i][j]:
                        put(s, h or 'col{}'.format(j), 'column {}'.format(j + 1))
    if verbose:
        sizes = Counter(assign.values())
        print('patient file {} ({}): {} rows{}, layout: {}'.format(
            os.path.basename(path), kind, len(rows), ', header ' + str(header) if header else '', layout or 'not recognised'))
        print('  images identified: {}{} | patients: {} | images per patient: min {} median {} max {}'.format(
            len(assign), ' of {}'.format(len(stems)) if stems is not None else '', len(sizes),
            min(sizes.values()) if sizes else 0, sorted(sizes.values())[len(sizes) // 2] if sizes else 0,
            max(sizes.values()) if sizes else 0))
        by_class = defaultdict(set)
        for s, p in assign.items():
            by_class[re.match(r'[a-z]+', s).group(0)].add(p)
        print('  patients per class: ' + ' | '.join('{} {}'.format(n, len(by_class[k])) for k, n in
                                                     (('n', 'normal'), ('b', 'benign'), ('is', 'in situ'), ('iv', 'invasive'))))
        if conflicts:
            print('  WARNING: {} images listed under two patients (first kept), e.g. {}'.format(len(conflicts), conflicts[:3]))
        print('  first rows as read: {}'.format(rows[:4]))
    if not assign:
        raise SystemExit('no image -> patient pairs recognised in {}; send the first rows above'.format(path))
    return assign


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', required=True)
    ap.add_argument('--images', default=None, help='BACH_data/images: restrict to the images that exist')
    a = ap.parse_args()
    stems = None
    if a.images:
        stems = {os.path.splitext(os.path.basename(f))[0] for f in glob(os.path.join(a.images, '*', '*.tif*'))}
        print('images found: {}'.format(len(stems)))
    pat = read_patients(a.patients, stems)
    groups = Counter(pat.values())
    print('largest patients:', ', '.join('{} ({} images)'.format(p, n) for p, n in groups.most_common(5)))
    if stems is not None:
        print('unidentified images (each kept as its own group, as the organisers advise): {}'.format(len(stems - set(pat))))
