import csv
import json
import re
from pathlib import Path

import numpy as np

from src.parsing import compute_iou, is_contrastive, parse_color, parse_type

FILE_PATTERN = re.compile(
    r"^(?P<model>.+?)_"
    r"(?P<strategy>label_only|question|constrained|contextual|"
    r"descriptive_cat|descriptive_multitask|descriptive)"
    r"_predictions\.jsonl$"
)


def load_segments(path=Path("metadata/frames.csv")):
    with open(path, encoding="utf-8") as fh:
        return {r["file_name"]: r["segment_id"] for r in csv.DictReader(fh)}


def load_split(path=Path("metadata/split.csv")):
    with open(path, encoding="utf-8") as fh:
        return {r["segment_id"]: r["fold"] for r in csv.DictReader(fh)}


def fold_mask(data, split, fold="test"):
    if data["segment"] is None:
        raise ValueError("predictions carry no segments")
    return np.array([split[s] == fold for s in data["segment"]])


def subset(data, mask):
    out = dict(data)
    out["n"] = int(mask.sum())
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            out[key] = value[mask]
    return out


def load(path, normalize=True, segments=None):
    path = Path(path)
    match = FILE_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"unexpected file name: {path.name}")

    with open(path, encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh if l.strip()]

    contrastive = is_contrastive(records[0].get("model_name"))
    n = len(records)

    pred_type, pred_color, iou = [], [], np.zeros(n)
    for i, r in enumerate(records):
        if contrastive:
            pred_type.append(r.get("pred_type"))
            pred_color.append(r.get("pred_color"))
        else:
            pred_type.append(parse_type(r.get("raw_type") or r.get("raw_multitask"), normalize))
            pred_color.append(parse_color(r.get("raw_color") or r.get("raw_multitask"), normalize))
        if r.get("detection_valid"):
            iou[i] = compute_iou(r.get("pred_bbox_xyxy"), r.get("gt_bbox_xyxy"))

    file_name = np.array([r["file_name"] for r in records])
    segment = None
    if segments:
        mapped = [segments.get(f) for f in file_name]
        if all(m is not None for m in mapped):
            segment = np.array(mapped)
    gt_type = np.array([r["gt_type"] for r in records])
    gt_color = np.array([r["gt_color"] for r in records])
    pred_type = np.array([p or "" for p in pred_type])
    pred_color = np.array([p or "" for p in pred_color])

    return {
        "model": match.group("model"),
        "strategy": match.group("strategy"),
        "n": n,
        "file_name": file_name,
        "segment": segment,
        "gt_type": gt_type,
        "gt_color": gt_color,
        "pred_type": pred_type,
        "pred_color": pred_color,
        "correct_type": ((pred_type != "") & (pred_type == gt_type)).astype(float),
        "correct_color": ((pred_color != "") & (pred_color == gt_color)).astype(float),
        "iou": iou,
        "valid_type": (pred_type != "").astype(float),
        "valid_color": (pred_color != "").astype(float),
        "valid_det": np.array([bool(r.get("detection_valid")) for r in records], dtype=float),
    }


def discover(root):
    found = {}
    for path in sorted(Path(root).rglob("*_predictions.jsonl")):
        match = FILE_PATTERN.match(path.name)
        if match:
            found[(match.group("model"), match.group("strategy"))] = path
    return found


def transfer_share(data, src, dst):
    mask = data["gt_color"] == src
    if not mask.any():
        return float("nan")
    return float((data["pred_color"][mask] == dst).mean())


def to_pink(data):
    mask = data["gt_color"] != "pink"
    return float((data["pred_color"][mask] == "pink").mean())
