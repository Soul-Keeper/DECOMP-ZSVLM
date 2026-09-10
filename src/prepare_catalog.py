"""
Builds the external catalog subsets from the Fashion Product Images Dataset.

The catalog is heavily skewed: 7066 t-shirts against 285 sweatshirts, and the
warm neutrals are rare. A single proportional sample would reproduce that skew
and leave the tables uninterpretable, so two subsets are drawn, each balanced
for its own task:

    type/    568 images, 284 per class, labels from manual annotation
    color/   1000 images, 200 per class, drawn from all upper-body categories

Both are written as COCO directories that src/dataset.py reads unchanged. The
catalog has no boxes, so a full-frame placeholder is used and detection is not
evaluated on these subsets.

By default the exact subsets of the paper are rebuilt from
metadata/catalog/*.csv. Pass --resample to draw new ones.

    python -m src.prepare_catalog --src ./fashion-dataset --out ./data_ext
    python -m src.prepare_catalog --src ./fashion-dataset --dry-run
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Catalog colors mapped onto the deployment label set. This is a dataset-level
# mapping and is distinct from the parser rules of src/parsing.py, which govern
# how model responses are read.
COLOR_MAP = {
    "Black": "black",
    "White": "white", "Off White": "white", "Cream": "white",
    "Brown": "brown", "Coffee Brown": "brown",
    "Beige": "nude", "Tan": "nude", "Nude": "nude",
    "Skin": "nude", "Taupe": "nude", "Khaki": "nude",
    "Pink": "pink", "Rose": "pink",
}

COLOR_SUBCATEGORY = "Topwear"
MIN_SIDE = 200


def read_styles(src):
    path = Path(src) / "styles.csv"
    if not path.exists():
        sys.exit(f"not found: {path}")
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df = df.dropna(subset=["id", "articleType", "baseColour"])
    df["id"] = df["id"].astype(int)
    return df


def filter_existing(src, df):
    image_dir = Path(src) / "images"
    if not image_dir.is_dir():
        sys.exit(f"not found: {image_dir}")
    available = {int(p.stem) for p in image_dir.glob("*.jpg") if p.stem.isdigit()}
    df = df[df["id"].isin(available)].copy()

    from PIL import Image
    sides = []
    for i in df["id"].head(20):
        try:
            with Image.open(image_dir / f"{i}.jpg") as im:
                sides.append(min(im.size))
        except Exception:
            pass
    if sides and int(np.median(sides)) < MIN_SIDE:
        print(f"[!] median image side {int(np.median(sides))} px, this looks like "
              f"the thumbnail release; the full version is needed")
    return image_dir, df


def cap_per_class(df, column, cap, seed):
    rng = np.random.default_rng(seed)
    parts = [g.iloc[rng.choice(len(g), min(cap, len(g)), replace=False)]
             for _, g in df.groupby(column)]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def balance_per_class(df, column, seed, cap=None):
    rng = np.random.default_rng(seed)
    n = int(df[column].value_counts().min())
    if cap:
        n = min(n, cap)
    parts = [g.iloc[rng.choice(len(g), n, replace=False)]
             for _, g in df.groupby(column)]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def build_coco(df, image_dir, out_dir, type_column, color_column, copy_images):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    for new_id, (_, row) in enumerate(df.iterrows()):
        source = image_dir / f'{int(row["id"])}.jpg'
        try:
            with Image.open(source) as im:
                width, height = im.size
        except Exception:
            continue

        name = f'{int(row["id"])}.jpg'
        if copy_images:
            shutil.copy2(source, out_dir / name)

        images.append({
            "id": new_id, "file_name": name, "width": width, "height": height,
            "extra": {
                "user_tags": [f"type-{row[type_column]}", f"color-{row[color_column]}"],
                "source_id": int(row["id"]),
                "source_articleType": row["articleType"],
                "source_baseColour": row["baseColour"],
            },
        })
        annotations.append({
            "id": new_id, "image_id": new_id, "category_id": 0,
            "bbox": [0, 0, width, height], "area": width * height, "iscrowd": 0,
        })

    return {
        "info": {"description": "External catalog subset",
                 "note": "bbox is a full-frame placeholder; detection is not evaluated"},
        "licenses": [],
        "categories": [{"id": 0, "name": "garment", "supercategory": "garment"}],
        "images": images, "annotations": annotations,
    }


def report(df, column, title):
    counts = Counter(df[column])
    top, n = counts.most_common(1)[0]
    print(f"  {title}: n={len(df)}, majority {top} {n / len(df):.3f}")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<14}{v:>6}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="./fashion-dataset")
    ap.add_argument("--out", default="./data_ext")
    ap.add_argument("--metadata", default="metadata")
    ap.add_argument("--manual", default="./manual_labels.json")
    ap.add_argument("--resample", action="store_true",
                    help="draw new subsets instead of rebuilding the released ones")
    ap.add_argument("--type-cap", type=int, default=None)
    ap.add_argument("--color-cap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-copy", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    catalog = Path(args.metadata) / "catalog"
    df = read_styles(src)
    image_dir, df = filter_existing(src, df)
    print(f"rows with an existing image: {len(df)}")
    df["color_lbl"] = df["baseColour"].map(COLOR_MAP)

    if not args.resample:
        subsets = {}
        for name, column in [("type", "type_lbl"), ("color", "color_lbl")]:
            path = catalog / f"{name}_subset.csv"
            if not path.exists():
                sys.exit(f"not found: {path} (use --resample to draw new subsets)")
            frozen = pd.read_csv(path)
            merged = df[df["id"].isin(frozen["id"])].merge(
                frozen[["id", "type_lbl", "color_lbl"]], on="id", suffixes=("_src", ""))
            missing = len(frozen) - len(merged)
            print(f"\n{name} subset: {len(merged)} of {len(frozen)} images available"
                  + (f", {missing} missing from this copy of the catalog" if missing else ""))
            report(merged, column, name)
            subsets[name] = merged
    else:
        print("\ntype subset")
        manual_path = Path(args.manual)
        if not manual_path.exists():
            sys.exit(f"not found: {manual_path}; run src/label_catalog.py first")
        manual = json.loads(manual_path.read_text(encoding="utf-8"))
        usable = {int(k): v for k, v in manual.items() if v in ("tshirt", "sweatshirt")}
        print(f"  manual labels: {len(manual)}, usable: {len(usable)}")

        t = df[df["id"].isin(usable)].copy()
        t["type_lbl"] = t["id"].map(usable)
        t["color_lbl"] = t["color_lbl"].fillna("black")
        agreement = (t["articleType"].map({"Tshirts": "tshirt", "Sweatshirts": "sweatshirt"})
                     == t["type_lbl"]).mean()
        print(f"  agreement with catalog metadata: {agreement:.1%}")
        subsets = {"type": balance_per_class(t, "type_lbl", args.seed, cap=args.type_cap)}
        report(subsets["type"], "type_lbl", "type")

        print("\ncolor subset")
        c = df[(df["subCategory"] == COLOR_SUBCATEGORY) & df["color_lbl"].notna()].copy()
        print(f"  {COLOR_SUBCATEGORY} with a recognised color: {len(c)}")
        c["type_lbl"] = c["articleType"].map(
            {"Tshirts": "tshirt", "Sweatshirts": "sweatshirt"}).fillna("tshirt")
        subsets["color"] = cap_per_class(c, "color_lbl", args.color_cap, args.seed)
        report(subsets["color"], "color_lbl", "color")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    for name, subset in subsets.items():
        target = out / name
        coco = build_coco(subset, image_dir, target, "type_lbl", "color_lbl",
                          copy_images=not args.no_copy)
        (target / "_annotations.coco.json").write_text(
            json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        subset[["id", "articleType", "baseColour", "type_lbl", "color_lbl"]].to_csv(
            target / "manifest.csv", index=False)
        print(f"\n{name}: {len(coco['images'])} images -> {target}")

    print("\nRun the four strategies on each subset, then read type metrics from")
    print("type/ and color metrics from color/. The secondary label in each")
    print("subset is filled in formally and carries no meaning.\n")
    print("  python -m src.run_benchmark --models QWEN2.5-VL-7B --strategy descriptive \\")
    print(f"    --data-dir {out}/type --output-dir ./outputs_ext/type_descriptive --no-detection")


if __name__ == "__main__":
    main()
