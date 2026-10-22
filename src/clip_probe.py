"""
Linear probe on frozen CLIP ViT-L/14 image features (Table 10): how many
labelled frames a supervised head needs to reach zero-shot accuracy.

    python -m src.clip_probe extract --data-dir ./data   # GPU, writes outputs_probe/clip_features.npz
    python -m src.clip_probe probe                       # CPU, prints Table 10, writes outputs_probe/probe.csv

The features need the production images, so Table 10 cannot be recomputed from
this release alone. The checkpoint and preprocessing are those of the zero-shot
runs (openai/clip-vit-large-patch14, CLIPProcessor on the full frame), and the
embeddings are normalized to unit length.

For each N, 50 draws of N frames are taken from the training fold, one frame per
acquisition segment until the segments run out and every class present. The
regularization strength is chosen within the N frames by five-fold
cross-validation on log-loss, folds grouped by segment and stratified by class
where the counts allow, so that no validation or test frame informs it. The row
for the full training fold is a single fit. Accuracy is measured on the held-out
fold, with the half-width of the 95% cluster-bootstrap interval over its
segments. probe needs scikit-learn.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

TYPES = ["tshirt", "sweatshirt"]
COLORS = ["black", "brown", "nude", "white", "pink"]
SIZES = [10, 25, 50, 100, 200]
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ZERO_SHOT = {"type": "CLIP-VIT-L14_descriptive", "color": "QWEN3-VL-8B_descriptive"}
FINE_TUNED = "YOLO11n-cls"


def read_frames(metadata):
    frames = list(csv.DictReader(open(Path(metadata) / "frames.csv", encoding="utf-8")))
    split = {r["segment_id"]: r["fold"] for r in
             csv.DictReader(open(Path(metadata) / "split.csv", encoding="utf-8"))}
    for f in frames:
        f["fold"] = split[f["segment_id"]]
    return frames


# ── extraction ─────────────────────────────────────────────────────────────

def extract(args):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

    frames = read_frames(args.metadata)
    missing = [f["file_name"] for f in frames if not (Path(args.data_dir) / f["file_name"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images not found in {args.data_dir}, e.g. {missing[0]}")

    feats = []
    for i in range(0, len(frames), args.batch_size):
        batch = frames[i:i + args.batch_size]
        images = [Image.open(Path(args.data_dir) / f["file_name"]).convert("RGB") for f in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(pixel_values=inputs["pixel_values"].to(model.dtype))
        feats.append(emb.float().cpu().numpy())
        print(f"\r{i + len(batch)}/{len(frames)}", end="", flush=True)
    print()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "clip_features.npz",
                        features=np.concatenate(feats).astype(np.float32),
                        file_name=np.array([f["file_name"] for f in frames]))
    print(f"saved {out / 'clip_features.npz'}")


# ── probe ──────────────────────────────────────────────────────────────────

def draw(rng, idx, labels, segments, n, n_classes, tries=1000):
    """n frames from idx, one per segment until the segments run out, every class present."""
    for _ in range(tries):
        order = rng.permutation(np.unique(segments[idx]))
        first = np.array([rng.choice(idx[segments[idx] == s]) for s in order])
        if n <= len(first):
            pick = first[:n]
        else:
            rest = np.setdiff1d(idx, first)
            pick = np.concatenate([first, rng.choice(rest, size=n - len(first), replace=False)])
        if len(np.unique(labels[pick])) == n_classes:
            return pick
    raise RuntimeError(f"could not draw {n} frames covering all classes")


def choose_c(X, y, groups, n_classes):
    """C with the lowest cross-validated log-loss within the drawn frames, or None."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    k = min(5, len(np.unique(groups)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # classes with fewer members than folds
        try:
            splits = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0).split(X, y, groups))
        except ValueError:  # every class has fewer members than folds
            splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    loss, used = dict.fromkeys(C_GRID, 0.0), 0
    for train, test in splits:
        if len(np.unique(y[train])) < 2:
            continue
        used += 1
        clf = LogisticRegression(max_iter=5000, warm_start=True)
        for C in C_GRID:
            clf.set_params(C=C)
            clf.fit(X[train], y[train])
            # a class absent from the training part gets the floor probability for every C
            prob = np.full((len(test), n_classes), 1e-6)
            prob[:, clf.classes_] = np.clip(clf.predict_proba(X[test]), 1e-6, 1.0)
            prob /= prob.sum(axis=1, keepdims=True)
            loss[C] -= np.log(prob[np.arange(len(test)), y[test]]).sum()
    return min(C_GRID, key=lambda c: (loss[c], c)) if used else None


def fit_predict(X, y, pick, test, C):
    """Fit along the C grid with warm start up to C, as in the reported run."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=5000, warm_start=True)
    for c in C_GRID[:C_GRID.index(C) + 1]:
        clf.set_params(C=c)
        clf.fit(X[pick], y[pick])
    return (clf.predict(X[test]) == y[test]).astype(float)


def zero_shot(args, test_names, task):
    from src.bootstrap import cluster_ci, half_width
    from src.loading import load, load_segments
    d = load(Path(args.predictions) / "main_benchmark" / f"{ZERO_SHOT[task]}_predictions.jsonl",
             segments=load_segments(Path(args.metadata) / "frames.csv"))
    mask = np.isin(d["file_name"], test_names)
    values = d[f"correct_{task}"][mask]
    return values.mean(), half_width(*cluster_ci(values, d["segment"][mask], args.n_boot))


def probe(args):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from src.bootstrap import cluster_ci, half_width
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # only at the largest C

    frames = read_frames(args.metadata)
    data = np.load(Path(args.out_dir) / "clip_features.npz")
    pos = {n: i for i, n in enumerate(data["file_name"])}
    X = data["features"][[pos[f["file_name"]] for f in frames]]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    fold = np.array([f["fold"] for f in frames])
    segments = np.array([f["segment_id"] for f in frames])
    names = np.array([f["file_name"] for f in frames])
    train, test = np.where(fold == "train")[0], np.where(fold == "test")[0]

    rows, table, without_cv = [], {}, 0
    for task, classes in [("type", TYPES), ("color", COLORS)]:
        y = np.array([classes.index(f[f"gt_{task}"]) for f in frames])
        zs, zs_hw = zero_shot(args, names[test], task)
        for n in SIZES + [len(train)]:
            rng = np.random.default_rng(0)
            n_draws = 1 if n == len(train) else args.draws
            acc, correct, chosen = [], np.zeros(len(test)), []
            for _ in range(n_draws):
                pick = train if n == len(train) else draw(rng, train, y, segments, n, len(classes))
                C = choose_c(X[pick], y[pick], segments[pick], len(classes))
                if C is None:
                    without_cv += 1
                    C = 1.0
                hit = fit_predict(X, y, pick, test, C)
                acc.append(hit.mean())
                correct += hit
                chosen.append(C)
            acc = np.array(acc)
            hw = half_width(*cluster_ci(correct / n_draws, segments[test], args.n_boot))
            table[(task, n)] = (acc.mean(), hw)
            rows.append(dict(task=task, n=n, draws=n_draws, mean=acc.mean(), half_width=hw,
                             p5=np.percentile(acc, 5), p95=np.percentile(acc, 95),
                             share_above_zero_shot=(acc > zs).mean(),
                             typical_c=max(set(chosen), key=chosen.count)))
        table[(task, "zs")] = (zs, zs_hw)

    reference = {r["display"]: r for r in
                 csv.DictReader(open(Path(args.metadata) / "reference_points.csv", encoding="utf-8"))}
    print(f"\nTABLE 10  Linear probe on frozen CLIP ViT-L/14 features "
          f"(held-out fold: {len(test)} frames, {len(np.unique(segments[test]))} segments)")
    print("-" * 64)
    print(f'{"Labeled frames":<26}{"Type_adj":>19}{"Color_adj":>19}')
    print("-" * 64)
    for n in SIZES + [len(train)]:
        label = f"{n} (all)" if n == len(train) else str(n)
        cells = "".join(f'{f"{m:.3f}±{hw:.3f}":>19}' for m, hw in (table[("type", n)], table[("color", n)]))
        print(f"{label:<26}{cells}")
    cells = "".join(f'{f"{m:.3f}±{hw:.3f}":>19}' for m, hw in (table[("type", "zs")], table[("color", "zs")]))
    print(f'{"Zero-shot, no labels":<26}{cells}')
    ft = reference[FINE_TUNED]
    print(f'{"Fine-tuned (" + FINE_TUNED + ")":<26}{ft["type_adj"]:>19}{ft["color_adj"]:>19}')
    if without_cv:
        print(f"[!] cross-validation unusable in {without_cv} draws, C = 1 used there")

    out = Path(args.out_dir) / "probe.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data-dir", default="data")
    e.add_argument("--batch-size", type=int, default=32)
    p = sub.add_parser("probe")
    p.add_argument("--draws", type=int, default=50)
    p.add_argument("--n-boot", type=int, default=10000, dest="n_boot")
    p.add_argument("--predictions", default="predictions")
    for s in (e, p):
        s.add_argument("--metadata", default="metadata")
        s.add_argument("--out-dir", default="outputs_probe", dest="out_dir")
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else probe(args)


if __name__ == "__main__":
    main()
