"""
Manual annotation of garment type for the catalog subset.

The catalog articleType field follows the product name and is noisy: hoodies,
fleeces and heavy long-sleeved items are distributed inconsistently between
Tshirts and Sweatshirts. This tool produces clean labels and measures the
disagreement with the catalog, which is reported in the paper.

Only Topwear items of the listed articleTypes are shown.

Labelling rule, by construction rather than sleeve length:
    sweatshirt  heavy brushed knit, including hoodies and fleece
    tshirt      light jersey, including long-sleeved
    skip        not decidable from the photograph

Keys:
    1 / T  tshirt          2 / S  sweatshirt      0 / X  skip
    Backspace  back        U  next unlabelled     D  toggle catalog label
    + / -  zoom            Esc / Q  quit

Progress is written after every label, so the tool can be closed at any point.

    python -m src.label_catalog --src ./fashion-dataset
    python -m src.label_catalog --src ./fashion-dataset --blind
    python -m src.label_catalog --src ./fashion-dataset --report
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Only these articleTypes are shown; everything else is filtered out.
RELEVANT_TYPES = ['Tshirts', 'Sweatshirts']

LABELS = {'tshirt': (80, 200, 255), 'sweatshirt': (255, 180, 60), 'skip': (150, 150, 150)}


def load_pool(src: Path, types: list, max_per_type: int | None) -> pd.DataFrame:
    path = src / 'styles.csv'
    if not path.exists():
        sys.exit(f'not found: {path}')
    try:
        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
    except TypeError:
        df = pd.read_csv(path, error_bad_lines=False, engine='python')

    df = df.dropna(subset=['id', 'articleType', 'baseColour'])
    df['id'] = df['id'].astype(int)
    df = df[df['articleType'].isin(types)]

    img_dir = src / 'images'
    have = {int(p.stem) for p in img_dir.glob('*.jpg') if p.stem.isdigit()}
    df = df[df['id'].isin(have)].copy()

    if max_per_type:
        parts = []
        rng = np.random.default_rng(42)
        for t, g in df.groupby('articleType'):
            take = min(max_per_type, len(g))
            parts.append(g.iloc[rng.choice(len(g), take, replace=False)])
        df = pd.concat(parts)

    return df.sort_values('id').reset_index(drop=True)


def draw(img, info, label, catalog, show_catalog, view_h=900):
    h, w = img.shape[:2]
    scale = view_h / h
    vis = cv2.resize(img, (max(1, int(w * scale)), int(h * scale)),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    pad = np.zeros((vis.shape[0], 340, 3), np.uint8)
    vis = np.hstack([vis, pad])
    x0 = vis.shape[1] - 330

    for i, line in enumerate(info):
        cv2.putText(vis, line, (x0, 30 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    y = 30 + len(info) * 24 + 20
    if show_catalog:
        cv2.putText(vis, f'catalog: {catalog}', (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
    y += 40

    if label:
        cv2.putText(vis, label.upper(), (x0, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, LABELS[label], 2)

    y += 60
    for txt in ['1 / T  tshirt', '2 / S  sweatshirt', '0 / X  skip',
                '', 'Backspace  back', 'U  next unlabelled',
                'D  catalog label', '+ / -  zoom', 'Esc  quit']:
        if txt:
            cv2.putText(vis, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (180, 180, 180), 1)
        y += 24
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='./fashion-dataset')
    ap.add_argument('--out', default='./manual_labels.json')
    ap.add_argument('--types', nargs='+', default=RELEVANT_TYPES)
    ap.add_argument('--max_per_type', type=int, default=None,
                    help='cap the number of items per articleType')
    ap.add_argument('--blind', action='store_true',
                    help='hide the catalog label so it does not anchor the annotation')
    ap.add_argument('--report', action='store_true', help='print the report only')
    ap.add_argument('--view_h', type=int, default=900,
                    help='display height in pixels; adjust with + and - in the window')
    args = ap.parse_args()

    src = Path(args.src)
    df = load_pool(src, args.types, args.max_per_type)
    print(f'  to label: {len(df)}')
    print(f'  {Counter(df["articleType"])}\n')

    out_path = Path(args.out)
    labels = json.loads(out_path.read_text(encoding='utf-8')) \
        if out_path.exists() else {}
    print(f'  already labelled: {len(labels)}\n')

    if args.report:
        report(df, labels)
        return

    def save():
        out_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2),
                            encoding='utf-8')

    img_dir = src / 'images'
    idx = 0
    # start at the first unlabelled item
    while idx < len(df) and str(int(df.iloc[idx]['id'])) in labels:
        idx += 1
    show_catalog = not args.blind

    win = 'label'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1100, args.view_h + 40)

    while True:
        idx = max(0, min(idx, len(df) - 1))
        row = df.iloc[idx]
        pid = str(int(row['id']))
        img = cv2.imread(str(img_dir / f'{pid}.jpg'))
        if img is None:
            idx += 1
            continue

        done = len(labels)
        agree = sum(1 for k, v in labels.items()
                    if v != 'skip' and _cat(df, k) == v)
        checked = sum(1 for v in labels.values() if v != 'skip')
        info = [
            f'[{idx + 1}/{len(df)}]  id={pid}',
            f'labelled: {done}',
            f'agreement with catalog: '
            f'{(agree / checked if checked else 0):.0%} ({agree}/{checked})',
            '',
            str(row['productDisplayName'])[:38],
            str(row['productDisplayName'])[38:76],
        ]

        cv2.imshow(win, draw(img, info, labels.get(pid),
                             row['articleType'], show_catalog, args.view_h))
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('+'), ord('=')):
            args.view_h = min(args.view_h + 100, 1600)
        elif key in (ord('-'), ord('_')):
            args.view_h = max(args.view_h - 100, 300)
        elif key in (ord('1'), ord('t'), ord('T')):
            labels[pid] = 'tshirt'; save(); idx += 1
        elif key in (ord('2'), ord('s'), ord('S')):
            labels[pid] = 'sweatshirt'; save(); idx += 1
        elif key in (ord('0'), ord('x'), ord('X')):
            labels[pid] = 'skip'; save(); idx += 1
        elif key == 8:
            idx -= 1
        elif key in (ord('u'), ord('U')):
            nxt = [i for i in range(idx + 1, len(df))
                   if str(int(df.iloc[i]['id'])) not in labels]
            idx = nxt[0] if nxt else idx
        elif key in (ord('d'), ord('D')):
            show_catalog = not show_catalog

    cv2.destroyAllWindows()
    save()
    report(df, labels)


def _cat(df, pid):
    """Catalog label mapped onto the annotation vocabulary."""
    m = {'Tshirts': 'tshirt', 'Sweatshirts': 'sweatshirt'}
    r = df[df['id'] == int(pid)]
    return m.get(r.iloc[0]['articleType']) if len(r) else None


def report(df, labels):
    print('\n' + '=' * 66)
    print('  ANNOTATION SUMMARY')
    print('=' * 66)
    if not labels:
        print('  empty\n')
        return

    c = Counter(labels.values())
    print(f'  labelled: {len(labels)}')
    for k in ['tshirt', 'sweatshirt', 'skip']:
        print(f'    {k:<12}{c.get(k, 0):>6}')

    usable = {k: v for k, v in labels.items() if v != 'skip'}
    if not usable:
        print()
        return

    agree = sum(1 for k, v in usable.items() if _cat(df, k) == v)
    print(f'\n  agreement with catalog: {agree}/{len(usable)} '
          f'({agree / len(usable):.1%})')
    print(f'  disagreements: {len(usable) - agree} ({1 - agree / len(usable):.1%})')

    print(f'\n  matrix (rows: catalog, columns: manual):')
    print(f'  {"":<14}{"tshirt":>12}{"sweatshirt":>12}')
    for cat in ['tshirt', 'sweatshirt']:
        row = [sum(1 for k, v in usable.items()
                   if _cat(df, k) == cat and v == man)
               for man in ['tshirt', 'sweatshirt']]
        print(f'  {cat:<14}{row[0]:>12}{row[1]:>12}')

    print(f'\n  distribution of manual labels:')
    print(f'    tshirt     {c.get("tshirt", 0)}')
    print(f'    sweatshirt {c.get("sweatshirt", 0)}')
    n = min(c.get('tshirt', 0), c.get('sweatshirt', 0))
    print(f'\n  balanced subset: {n} x 2 = {2 * n}, majority 0.500')
    print('=' * 66 + '\n')


if __name__ == '__main__':
    main()
