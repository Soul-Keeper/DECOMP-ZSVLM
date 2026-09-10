"""
Regenerates the tables of the paper from the released predictions.

    python -m src.compute_tables                # all tables
    python -m src.compute_tables --table 6      # one table
    python -m src.compute_tables --n-boot 1000  # faster, wider intervals

Table 7 needs opencv (pip install opencv-python-headless).
"""

import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

from src.bootstrap import cluster_ci, half_width, paired_cluster_ci
from src.loading import (fold_mask, load, load_segments, load_split, subset,
                     to_pink, transfer_share)

MODELS = [
    ("VILT-B32-FINETUNED-VQA", "ViLT-B/32", "Early VQA models"),
    ("BLIP-VQA-CAPFILT-LARGE", "BLIP-VQA", "Early VQA models"),
    ("CLIP-VIT-L14", "CLIP ViT-L/14", "Specialist models"),
    ("FLORENCE-2-LARGE", "Florence-2-L", "Specialist models"),
    ("PALIGEMMA", "PaliGemma-3B", "Instruction-tuned MLLMs"),
    ("LLAVA-1.6-MISTRAL-7B", "LLaVA-1.6-7B", "Instruction-tuned MLLMs"),
    ("INTERNVL2-8B", "InternVL2-8B", "Instruction-tuned MLLMs"),
    ("MINICPM-LLAMA3-V-2.5", "MiniCPM-V-2.5", "Instruction-tuned MLLMs"),
    ("QWEN2.5-VL-7B", "Qwen2.5-VL-7B", "Instruction-tuned MLLMs"),
    ("QWEN3-VL-8B", "Qwen3-VL-8B", "Instruction-tuned MLLMs"),
]
DISPLAY = {m: d for m, d, _ in MODELS}

ABLATION_MODELS = ["BLIP-VQA-CAPFILT-LARGE", "CLIP-VIT-L14", "LLAVA-1.6-MISTRAL-7B",
                   "PALIGEMMA", "QWEN2.5-VL-7B"]
ABLATION_STRATEGY = {"BLIP-VQA-CAPFILT-LARGE": "label_only"}
LEVELS = {"L1": "L1 full frame", "L2": "L2 polygon",
          "L3": "L3 predicted box", "L4": "L4 GT box"}

PROMPT_SHORT = {"constrained": "constr.", "label_only": "label",
                "descriptive": "desc.", "question": "question",
                "contextual": "context."}

CATALOG_MODELS = ["CLIP-VIT-L14", "LLAVA-1.6-MISTRAL-7B", "PALIGEMMA", "QWEN2.5-VL-7B"]
CATALOG_EXTRA = {"CLIP-VIT-L14": "label_only"}


def r3(x):
    return Decimal(float(x)).quantize(Decimal("0.001"), ROUND_HALF_UP)


def r0(x):
    return int(Decimal(float(x)).quantize(Decimal("1"), ROUND_HALF_UP))


def pct(x):
    """Percentage with half-up rounding; f-strings round halves to even."""
    return f'{Decimal(float(x) * 100).quantize(Decimal("0.1"), ROUND_HALF_UP)}%'


def rule(title, width=110):
    print()
    print(title)
    print("-" * width)


def ablation_strategy(model):
    return ABLATION_STRATEGY.get(model, "descriptive")


def pred_path(root, *parts):
    return Path(root).joinpath(*parts)


def ci(data, key, n_boot):
    lo, hi = cluster_ci(data[key], data["segment"], n_boot)
    return half_width(lo, hi)


# ── Table 4 ────────────────────────────────────────────────────────────────

def table04(args, segments):
    root = pred_path(args.predictions, "main_benchmark")
    files = {}
    for path in sorted(root.glob("*_predictions.jsonl")):
        d = load(path, segments=segments)
        if d["strategy"] == "descriptive_multitask":
            continue
        files.setdefault(d["model"], {})[d["strategy"]] = d

    rule("TABLE 4  Benchmark results and full prompt-strategy ablation")
    print(f'{"Model":<22}{"Strategy":<14}{"Type_adj":>16}{"r_T":>7}'
          f'{"Color_adj":>16}{"r_C":>7}{"IoU_adj":>16}{"r_D":>7}')
    print("-" * 110)

    frames = list(csv.DictReader(open(Path(args.metadata) / "frames.csv", encoding="utf-8")))
    maj_type = max(np.bincount(np.unique([f["gt_type"] for f in frames], return_inverse=True)[1]))
    maj_color = max(np.bincount(np.unique([f["gt_color"] for f in frames], return_inverse=True)[1]))
    print(f'{"Majority class":<22}{"—":<14}{r3(maj_type / len(frames)):>16}{1.000:>7.3f}'
          f'{r3(maj_color / len(frames)):>16}{1.000:>7.3f}{"—":>16}{"—":>7}')

    for row in csv.DictReader(open(Path(args.metadata) / "reference_points.csv", encoding="utf-8")):
        cells = []
        for value, rate in [("type_adj", "r_type"), ("color_adj", "r_color"), ("iou_adj", "r_det")]:
            cells.append(f'{row[value] or "—":>16}{row[rate] or "—":>7}')
        print(f'{row["display"]:<22}{row["strategy"]:<14}' + "".join(cells))

    for path in sorted(pred_path(args.predictions, "detectors").glob("*_predictions.jsonl")):
        d = detector_metrics(path)
        print(f'{d["display"]:<22}{"class names":<14}{"—":>16}{"—":>7}{"—":>16}{"—":>7}'
              f'{r3(d["iou_adj"]):>16}{d["r_det"]:>7.3f}')

    group = None
    for model, display, section in MODELS:
        if model not in files:
            continue
        if section != group:
            group = section
            print(f"\n{group}")
        has_det = any(d["valid_det"].sum() for d in files[model].values())
        for strategy in sorted(files[model]):
            d = files[model][strategy]
            t = f'{r3(d["correct_type"].mean())}±{r3(ci(d, "correct_type", args.n_boot))}'
            c = f'{r3(d["correct_color"].mean())}±{r3(ci(d, "correct_color", args.n_boot))}'
            if has_det:
                i = f'{r3(d["iou"].mean())}±{r3(ci(d, "iou", args.n_boot))}'
                rd = f'{d["valid_det"].mean():.3f}'
            else:
                i, rd = "—", "—"
            print(f'{display:<22}{strategy:<14}{t:>16}{d["valid_type"].mean():>7.3f}'
                  f'{c:>16}{d["valid_color"].mean():>7.3f}{i:>16}{rd:>7}')


def detector_metrics(path):
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ious, valid = [], 0
    for r in records:
        box = r.get("pred_polygon")
        if box:
            valid += 1
            gt = r["gt_bbox_xyxy"]
            xa, ya = max(gt[0], box[0]), max(gt[1], box[1])
            xb, yb = min(gt[2], box[2]), min(gt[3], box[3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            union = ((gt[2] - gt[0]) * (gt[3] - gt[1])
                     + (box[2] - box[0]) * (box[3] - box[1]) - inter)
            ious.append(inter / union if union else 0.0)
        else:
            ious.append(0.0)
    name = path.stem.replace("_predictions", "")
    return {"display": name, "iou_adj": float(np.mean(ious)), "r_det": valid / len(records)}


# ── Table 5 ────────────────────────────────────────────────────────────────

def table05(args, segments):
    root = pred_path(args.predictions, "main_benchmark")
    profile = json.load(open(Path(args.metadata) / "profiling.json", encoding="utf-8"))
    latency = {m["model"]: m for m in profile["multi_task"]}

    rule("TABLE 5  Multi-task vs single-task evaluation", 78)
    print(f'{"Model":<18}{"Mode":<9}{"Type":>8}{"Color":>8}{"IoU":>8}{"Lat. (ms)":>11}{"Speedup":>10}')
    print("-" * 78)

    for model in latency:
        rows = []
        for mode, strategy in [("Single", "descriptive"), ("Multi", "descriptive_multitask")]:
            path = root / f"{model}_{strategy}_predictions.jsonl"
            if not path.exists():
                continue
            d = load(path, segments=segments)
            ms = latency[model]["single_ms" if mode == "Single" else "multi_ms"]
            rows.append((mode, d, ms))
        if not rows:
            continue
        speedup = latency[model]["single_ms"] / latency[model]["multi_ms"]
        for i, (mode, d, ms) in enumerate(rows):
            tail = f'{speedup:>9.1f}×' if i == 0 else ""
            print(f'{DISPLAY[model] if i == 0 else "":<18}{mode:<9}'
                  f'{r3(d["correct_type"].mean()):>8}{r3(d["correct_color"].mean()):>8}'
                  f'{r3(d["iou"].mean()):>8}{r0(ms):>11}{tail:>10}')


# ── Table 6 ────────────────────────────────────────────────────────────────

def load_ablation(args, segments):
    out = {}
    for model in ABLATION_MODELS:
        strategy = ablation_strategy(model)
        for level in LEVELS:
            path = pred_path(args.predictions, "spatial_ablation", level,
                             f"{model}_{strategy}_predictions.jsonl")
            if path.exists():
                out[(model, level)] = load(path, segments=segments)
    return out


def table06(args, segments):
    data = load_ablation(args, segments)

    rule("TABLE 6  Spatial ablation")
    print(f'{"Model":<16}{"Level":<20}{"Color_adj":>10}{"Δ":>18}'
          f'{"Type_adj":>10}{"Δ":>18}{"→pink":>8}{"nude→br":>9}{"br→nude":>9}')
    print("-" * 118)

    for model in ABLATION_MODELS:
        base = data.get((model, "L1"))
        if base is None:
            continue
        for level, label in LEVELS.items():
            d = data.get((model, level))
            if d is None:
                continue
            cells = []
            for key in ["correct_color", "correct_type"]:
                if level == "L1":
                    cells.append(f'{r3(d[key].mean()):>10}{"—":>18}')
                else:
                    delta = d[key].mean() - base[key].mean()
                    lo, hi = paired_cluster_ci(d[key], base[key], d["segment"], args.n_boot)
                    text = f'{delta:+.3f}±{half_width(lo, hi):.3f}'
                    cells.append(f'{r3(d[key].mean()):>10}{text:>18}')
            print(f'{DISPLAY[model]:<16}{label:<20}' + "".join(cells)
                  + f'{pct(to_pink(d)):>8}{pct(transfer_share(d, "nude", "brown")):>9}'
                    f'{pct(transfer_share(d, "brown", "nude")):>9}')
        print()


# ── Table 7 ────────────────────────────────────────────────────────────────

def gt_coverage(args, frames):
    """Fraction of the annotated box that survives each mask."""
    import cv2

    area = json.load(open(Path(args.metadata) / "working_area.json", encoding="utf-8"))
    width, height = area["frame_size"]
    masks = {k: cv2.fillPoly(np.zeros((height, width), np.uint8),
                             [np.array(v, np.int32)], 1)
             for k, v in area["polygons"].items()}

    detector = {}
    path = pred_path(args.predictions, "detectors", "yolo-world-S_predictions.jsonl")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        detector[r["file_name"]] = (r.get("polygon", "A"), r.get("pred_polygon"))

    l2, l3 = {}, {}
    for f in frames:
        name = f["file_name"]
        box = [int(f["bbox_x1"]), int(f["bbox_y1"]), int(f["bbox_x2"]), int(f["bbox_y2"])]
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(width, box[2]), min(height, box[3])
        size = (x2 - x1) * (y2 - y1)
        polygon, predicted = detector.get(name, ("A", None))
        l2[name] = masks[polygon][y1:y2, x1:x2].sum() / size if size > 0 else 0.0
        if predicted and size > 0:
            xa, ya = max(box[0], predicted[0]), max(box[1], predicted[1])
            xb, yb = min(box[2], predicted[2]), min(box[3], predicted[3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            l3[name] = inter / ((box[2] - box[0]) * (box[3] - box[1]))
        else:
            l3[name] = 0.0
    return {"L2": l2, "L3": l3}


def table07(args, segments):
    frames = list(csv.DictReader(open(Path(args.metadata) / "frames.csv", encoding="utf-8")))
    coverage = gt_coverage(args, frames)
    data = load_ablation(args, segments)

    rule("TABLE 7  Color accuracy change under masking, by target coverage", 66)
    print(f'{"Model":<16}{"Level":<7}{"n":>7}{"Δ intact":>11}{"n":>9}{"Δ partial":>12}')
    print("-" * 66)

    for model in ABLATION_MODELS:
        base = data.get((model, "L1"))
        if base is None:
            continue
        for level in ["L2", "L3"]:
            d = data.get((model, level))
            if d is None:
                continue
            intact = np.array([coverage[level][f] >= 0.9 for f in d["file_name"]])
            cells = []
            for mask in [intact, ~intact]:
                if mask.sum() == 0:
                    cells.append(f'{0:>7}{"—":>11}')
                    continue
                delta = d["correct_color"][mask].mean() - base["correct_color"][mask].mean()
                cells.append(f'{int(mask.sum()):>7}{delta:>+11.3f}')
            print(f'{DISPLAY[model]:<16}{level:<7}' + cells[0] + f"{cells[1]:>21}")


# ── Table 8 ────────────────────────────────────────────────────────────────

def table08(args, segments):
    rule("TABLE 8  Production line against catalog photographs")
    print(f'{"Model":<16}{"Source / strategy":<28}{"→pink":>8}{"br→pink":>9}'
          f'{"nude→br":>9}{"br→nude":>9}{"Color_adj":>11}{"Type_adj":>10}')
    print("-" * 110)

    for model in CATALOG_MODELS:
        rows = [("production / descriptive", "main_benchmark", "descriptive", "descriptive")]
        for strategy in ["descriptive", "descriptive_cat", CATALOG_EXTRA.get(model, "constrained")]:
            label = strategy.replace("descriptive_cat", "descriptive-cat")
            rows.append((f"catalog / {label}", "catalog", strategy, strategy))

        for label, source, color_strategy, type_strategy in rows:
            if source == "main_benchmark":
                color = load(pred_path(args.predictions, source,
                                       f"{model}_{color_strategy}_predictions.jsonl"),
                             segments=segments)
                type_data = color
            else:
                color = load(pred_path(args.predictions, "catalog", "color",
                                       f"{model}_{color_strategy}_predictions.jsonl"))
                type_data = load(pred_path(args.predictions, "catalog", "type",
                                           f"{model}_{type_strategy}_predictions.jsonl"))
            print(f'{DISPLAY[model]:<16}{label:<28}'
                  f'{pct(to_pink(color)):>8}'
                  f'{pct(transfer_share(color, "brown", "pink")):>9}'
                  f'{pct(transfer_share(color, "nude", "brown")):>9}'
                  f'{pct(transfer_share(color, "brown", "nude")):>9}'
                  f'{r3(color["correct_color"].mean()):>11}'
                  f'{r3(type_data["correct_type"].mean()):>10}')
        print()


# ── Table 9 ────────────────────────────────────────────────────────────────

def table09(args, segments=None):
    profile = json.load(open(Path(args.metadata) / "profiling.json", encoding="utf-8"))

    rule("TABLE 9  Computational profiling", 74)
    print(f'{"Model":<18}{"Prompt":<10}{"Params":>9}{"VRAM (GB)":>12}{"A100":>11}{"4070 S":>11}')
    print("-" * 74)
    for m in profile["single_task"]:
        a100 = m["latency_ms"]["A100"]
        consumer = m["latency_ms"]["4070S"]
        mark = "" if m["bbox"] else "†"
        print(f'{m["display"] + mark:<18}{PROMPT_SHORT.get(m["strategy"], m["strategy"]):<10}{m["params"]:>9}'
              f'{m["vram_gb"]:>12.2f}{r0(a100):>11}'
              f'{(r0(consumer) if consumer else "—"):>11}')
    print("† no bounding-box prediction (latency covers two tasks)")


# ── Table 11 ───────────────────────────────────────────────────────────────

def table11(args, segments):
    root = pred_path(args.predictions, "main_benchmark")

    rule("TABLE 11  Effect of surface-form normalization (changes above 0.05)", 60)
    print(f'{"Model":<18}{"Strategy":<22}{"Type":>10}{"Color":>10}')
    print("-" * 60)

    for model, display, _ in MODELS:
        printed = False
        for path in sorted(root.glob(f"{model}_*_predictions.jsonl")):
            on = load(path, normalize=True, segments=segments)
            off = load(path, normalize=False, segments=segments)
            deltas = {}
            for key, name in [("correct_type", "Type"), ("correct_color", "Color")]:
                delta = on[key].mean() - off[key].mean()
                deltas[name] = delta if abs(delta) > 0.05 else None
            if not any(deltas.values()):
                continue
            printed = True
            cells = [f'{v:>+10.3f}' if v else f'{"—":>10}' for v in deltas.values()]
            print(f'{display:<18}{on["strategy"]:<22}' + "".join(cells))
        if printed:
            print()


TABLES = {4: table04, 5: table05, 6: table06, 7: table07,
          8: table08, 9: table09, 11: table11}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, choices=sorted(TABLES), default=None)
    ap.add_argument("--predictions", default="predictions")
    ap.add_argument("--metadata", default="metadata")
    ap.add_argument("--n-boot", type=int, default=10000, dest="n_boot")
    args = ap.parse_args()

    segments = load_segments(Path(args.metadata) / "frames.csv")
    wanted = [args.table] if args.table else sorted(TABLES)
    for number in wanted:
        TABLES[number](args, segments)
    print()


if __name__ == "__main__":
    main()
