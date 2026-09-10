"""
Builds metadata/frames.csv and metadata/split.csv from the released predictions.

The acquisition segment is the capture timestamp embedded in the frame name:
    07M40S_1678082860_42_jpg.rf.<hash>.jpg
             ^^^^^^^^^^
Three frames follow a different naming convention and carry no timestamp; they
are grouped into a single segment, "unknown", matching src/dataset.py and
bringing the total to 201.

The split assigns whole segments to folds, stratified by (type, color) of the
first frame of each segment. It depends on the order in which segments first
appear in the prediction records, so it is frozen here rather than recomputed.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def segment_of(file_name):
    parts = file_name.split("_")
    if len(parts) > 2 and parts[1].isdigit():
        return parts[1]
    return "unknown"


def read_records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def ground_truth(records):
    return {
        r["file_name"]: (
            r["gt_type"],
            r["gt_color"],
            tuple(r["gt_bbox_xyxy"]) if r.get("gt_bbox_xyxy") else None,
        )
        for r in records
    }


def assign_folds(records):
    rng = np.random.default_rng(SEED)

    stratum_of = {}
    for r in records:
        seg = segment_of(r["file_name"])
        if seg not in stratum_of:
            stratum_of[seg] = (r["gt_type"], r["gt_color"])

    by_stratum = defaultdict(list)
    for seg, stratum in stratum_of.items():
        by_stratum[stratum].append(seg)

    fold = {}
    for stratum in sorted(by_stratum):
        shuffled = rng.permutation(np.array(by_stratum[stratum])).tolist()
        n = len(shuffled)
        n_test = max(1, round(n * TEST_RATIO))
        n_val = max(1, round(n * VAL_RATIO))
        if n - n_test - n_val < 1:
            n_test = n_val = max(1, n // 3)
        for seg in shuffled[: n - n_test - n_val]:
            fold[seg] = "train"
        for seg in shuffled[n - n_test - n_val : n - n_test]:
            fold[seg] = "val"
        for seg in shuffled[n - n_test :]:
            fold[seg] = "test"
    return fold


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=Path("predictions/main_benchmark"))
    ap.add_argument("--out", type=Path, default=Path("metadata"))
    args = ap.parse_args()

    files = sorted(args.predictions.glob("*_predictions.jsonl"))
    if not files:
        raise SystemExit(f"no prediction files under {args.predictions}")

    records = read_records(files[0])
    reference = ground_truth(records)
    for path in files[1:]:
        if ground_truth(read_records(path)) != reference:
            raise SystemExit(f"ground truth in {path.name} differs from {files[0].name}")

    fold = assign_folds(records)

    frames = []
    for name in sorted(reference):
        gt_type, gt_color, bbox = reference[name]
        frames.append({
            "file_name": name,
            "segment_id": segment_of(name),
            "gt_type": gt_type,
            "gt_color": gt_color,
            "bbox_x1": bbox[0], "bbox_y1": bbox[1],
            "bbox_x2": bbox[2], "bbox_y2": bbox[3],
        })
    write_csv(args.out / "frames.csv", frames)

    split = [{"segment_id": s, "fold": f} for s, f in sorted(fold.items())]
    write_csv(args.out / "split.csv", split)

    sizes = Counter(f["segment_id"] for f in frames)
    median = sorted(sizes.values())[len(sizes) // 2]
    seg_counts = Counter(fold.values())
    frame_counts = Counter(fold[f["segment_id"]] for f in frames)

    print(f"{len(files)} prediction files agree on the ground truth")
    print(f"{len(frames)} frames, {len(sizes)} segments, median {median} frames per segment")
    print(f"types  {dict(Counter(f['gt_type'] for f in frames))}")
    print(f"colors {dict(Counter(f['gt_color'] for f in frames))}")
    print("split  segments {train}/{val}/{test}".format(**seg_counts))
    print("       frames   {train}/{val}/{test}".format(**frame_counts))
    print(f"-> {args.out / 'frames.csv'}")
    print(f"-> {args.out / 'split.csv'}")


if __name__ == "__main__":
    main()
