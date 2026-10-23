"""
Recomputes the figures quoted in the text of the paper from the released
predictions, section by section. The tables are produced by src/compute_tables.py.

    python -m src.compute_text                  # all sections
    python -m src.compute_text --section 7a     # one section (VII-A)

Sections: 3, 5b, 5d, 5e, 6, 7a, 7b, 7c, 8, a (Section V-B is 5b, and so on).
The check of the parser against a human reading (Section V-D, Appendix A) is
recomputed by src/parser_check.py. The class colour statistics of Section III
and the linear probe of Table 10 need the images: src/class_hue.py and
src/clip_probe.py.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from src.bootstrap import cluster_ci, half_width, image_ci
from src.compute_tables import (ABLATION_MODELS, DISPLAY, MODELS, align, load_ablation, pct,
                                procedure_gain, r3)
from src.loading import discover, fold_mask, load, load_segments, load_split, transfer_share
from src.parsing import (TYPE_LABELS, TYPE_RULES, bbox_text, compute_iou,
                         parse_bbox, parse_color, parse_json_bbox, parse_type)

TASKS = [("correct_type", "valid_type"), ("correct_color", "valid_color"), ("iou", "valid_det")]
INSTRUCTION_TUNED = ["PALIGEMMA", "LLAVA-1.6-MISTRAL-7B", "INTERNVL2-8B",
                     "MINICPM-LLAMA3-V-2.5", "QWEN2.5-VL-7B", "QWEN3-VL-8B"]


def head(title):
    print(f"\n{title}\n" + "-" * len(title))


def records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1]) if box else 0


def overlap(a, b):
    if a is None or b is None:
        return 0
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


class Data:
    """Released predictions, loaded once."""

    def __init__(self, args):
        self.args = args
        self.root = Path(args.predictions)
        self.segments = load_segments(Path(args.metadata) / "frames.csv")
        self.split = load_split(Path(args.metadata) / "split.csv")
        self._cache = {}

    def main(self, model, strategy, normalize=True):
        key = ("main", model, strategy, normalize)
        if key not in self._cache:
            self._cache[key] = load(self.root / "main_benchmark" / f"{model}_{strategy}_predictions.jsonl",
                                    normalize=normalize, segments=self.segments)
        return self._cache[key]

    def raw(self, model, strategy, source="main_benchmark"):
        return records(self.root / source / f"{model}_{strategy}_predictions.jsonl")

    def ablation(self, model, level):
        if "ablation" not in self._cache:
            self._cache["ablation"] = load_ablation(self.args, self.segments)
        return self._cache["ablation"][(model, level)]

    def detector(self, name):
        return records(self.root / "detectors" / f"{name}_predictions.jsonl")

    def runs(self):
        """Single-task runs of every model: {model: {strategy: data}}."""
        out = {}
        for (model, strategy) in discover(self.root / "main_benchmark"):
            if strategy != "descriptive_multitask":
                out.setdefault(model, {})[strategy] = self.main(model, strategy)
        return out


def supported_tasks(runs):
    """A task counts for a model if any of its strategies produces a valid output."""
    return [key for key, valid in TASKS if any(d[valid].sum() > 0 for d in runs.values())]


def joint_scores(runs, idx=None):
    tasks = supported_tasks(runs)
    return {s: float(np.mean([d[k].mean() if idx is None else d[k][idx].mean() for k in tasks]))
            for s, d in runs.items()}


# ── Section III ────────────────────────────────────────────────────────────

def section3(data):
    head("Section III  Frames in segments with a constant label")
    frames = list(csv.DictReader(open(Path(data.args.metadata) / "frames.csv", encoding="utf-8")))
    for key in ["gt_type", "gt_color"]:
        labels = {}
        for f in frames:
            labels.setdefault(f["segment_id"], set()).add(f[key])
        print(f"{key[3:]:<6}{pct(np.mean([len(labels[f['segment_id']]) == 1 for f in frames]))}")

    head("Section III  Operator in the frame (YOLO-World-L, person query)")
    det = data.detector("yolo-world-L")
    person = np.array([r["person_bbox_xyxy"] is not None for r in det])
    overlaps = np.array([(r["person_gt_iou"] or 0) > 0.1 for r in det])
    print(f"person detected: {pct(person.mean())} of {len(det)} frames")
    print(f"person region overlaps the annotated garment (IoU > 0.1): {pct(overlaps.mean())}")

# ── Section IV-B ───────────────────────────────────────────────────────────

def section4(data):
    head("Section IV-B  Open-vocabulary detectors: garment boxes per frame and adjusted IoU")
    for name in ["yolo-world-S", "yolo-world-M", "yolo-world-L", "yolo-world-X", "grounding-dino"]:
        det = data.detector(name)
        boxes = np.median([r["n_garment_boxes"] for r in det])
        valid = np.mean([r["pred_bbox_xyxy"] is not None for r in det])
        print(f"{name:<16}median {boxes:.0f} garment boxes   IoU_adj {np.mean([r['iou'] for r in det]):.3f}   r_D {valid:.3f}")

# ── Section V-B ────────────────────────────────────────────────────────────

def section5b(data, sizes=(10, 25, 50, 200), draws=200):
    runs = data.runs()
    head("Section V-B  Joint strategy: mean adjusted score over supported tasks")
    chosen = {}
    for model, display, _ in MODELS:
        scores = joint_scores(runs[model])
        ranked = sorted(scores, key=scores.get, reverse=True)
        chosen[model] = ranked[0]
        print(f"{display:<16}{ranked[0]:<14}lead {scores[ranked[0]] - scores[ranked[1]]:.3f}   "
              + "  ".join(f"{s} {scores[s]:.3f}" for s in sorted(scores)))
    print(f"descriptive chosen for {sum(s == 'descriptive' for s in chosen.values())} of {len(chosen)} models, "
          f"{sum(chosen[m] == 'descriptive' for m in INSTRUCTION_TUNED)} of {len(INSTRUCTION_TUNED)} "
          "instruction-tuned")

    head("Section V-B  Aggregation rule: mean, worst task, mean rank")
    same = 0
    for model, display, _ in MODELS:
        tasks = supported_tasks(runs[model])
        strategies = sorted(runs[model])
        values = {s: [runs[model][s][k].mean() for k in tasks] for s in strategies}
        by_min = max(strategies, key=lambda s: min(values[s]))
        points = dict.fromkeys(strategies, 0)
        for j in range(len(tasks)):
            for i, s in enumerate(sorted(strategies, key=lambda s: -values[s][j])):
                points[s] += len(strategies) - i
        by_rank = max(strategies, key=lambda s: points[s])
        same += chosen[model] == by_min == by_rank
        print(f"{display:<16}mean {chosen[model]:<13}worst task {by_min:<13}mean rank {by_rank}")
    print(f"same strategy under all three rules: {same} of {len(MODELS)}")

    head(f"Section V-B  Stability of the joint choice on {draws} random subsets per model and size")
    n = len(next(iter(runs[MODELS[0][0]].values()))["file_name"])
    rng = np.random.default_rng(0)
    for size in sizes:
        hits, losses, misses = [], [], {}
        for model in sorted(runs):
            display = DISPLAY[model]
            full = joint_scores(runs[model])
            tasks = supported_tasks(runs[model])
            strategies = sorted(runs[model])
            stacked = {s: np.stack([runs[model][s][k] for k in tasks]) for s in strategies}
            for _ in range(draws):
                idx = rng.choice(n, size, replace=False)
                pick = max(strategies, key=lambda s: stacked[s][:, idx].mean())
                hits.append(pick == chosen[model])
                losses.append(full[chosen[model]] - full[pick])
                if pick != chosen[model]:
                    misses[display] = misses.get(display, 0) + 1
        print(f"{size:>4} images: full-set choice recovered in {pct(np.mean(hits))} of {len(hits)} "
              f"model-draw pairs, mean loss {np.mean(losses):.4f}   misses {misses}")


# ── Section V-D ────────────────────────────────────────────────────────────

def section5d(data):
    head("Section V-D  Coordinate unit: Qwen3-VL-8B read on its grid and as pixels")
    for strategy in ["constrained", "descriptive", "label_only", "question"]:
        rows = data.raw("QWEN3-VL-8B", strategy)
        cells = []
        for grid in [True, False]:
            boxes = [parse_json_bbox(r.get("raw_detection"), grid) for r in rows]
            iou = np.mean([compute_iou(b, r["gt_bbox_xyxy"]) if b else 0.0 for b, r in zip(boxes, rows)])
            valid = np.mean([b is not None for b in boxes])
            cells.append(f'{"grid" if grid else "pixels"}: IoU_adj {iou:.3f}, valid {valid:.3f}')
        print(f"{strategy:<13}" + "   ".join(cells))

    head("Section V-D  Coordinate unit: MiniCPM-V-2.5 read on its grid and as pixels")
    for strategy in ["constrained", "descriptive", "label_only", "question", "descriptive_multitask"]:
        rows = data.raw("MINICPM-LLAMA3-V-2.5", strategy)
        cells = []
        for grid in [True, False]:
            iou = np.mean([compute_iou(b, r["gt_bbox_xyxy"]) if (b := parse_bbox(bbox_text(r), grid)) else 0.0
                           for r in rows])
            cells.append(f'{"grid" if grid else "pixels"} {iou:.3f}')
        print(f"{strategy:<23}" + "   ".join(cells))


# ── Section V-E ────────────────────────────────────────────────────────────

def section5e(data):
    head("Section V-E  Segment bootstrap against frame bootstrap (joint strategies, type and colour)")
    runs = data.runs()
    ratios = []
    for model, display, _ in MODELS:
        joint = max(joint_scores(runs[model]).items(), key=lambda kv: kv[1])[0]
        d = runs[model][joint]
        for key in ["correct_type", "correct_color"]:
            if d[key].max() > d[key].min():
                ratios.append(half_width(*cluster_ci(d[key], d["segment"], data.args.n_boot))
                              / half_width(*image_ci(d[key], data.args.n_boot)))
    print(f"median widening of the 95% interval: {np.median(ratios):.2f} over {len(ratios)} metrics")


# ── Section VI ─────────────────────────────────────────────────────────────

def section6(data):
    runs = data.runs()
    head("Section VI-B  Single-task runs at or below the majority-class reference on a task")
    frames = list(csv.DictReader(open(Path(data.args.metadata) / "frames.csv", encoding="utf-8")))
    majority = {k: max(np.unique([f[k] for f in frames], return_counts=True)[1]) / len(frames)
                for k in ["gt_type", "gt_color"]}
    below = [r3(d["correct_type"].mean()) <= r3(majority["gt_type"])
             or r3(d["correct_color"].mean()) <= r3(majority["gt_color"])
             for model in runs for d in runs[model].values()]
    print(f"{sum(below)} of {len(below)} (majority class: type {r3(majority['gt_type'])}, "
          f"colour {r3(majority['gt_color'])})")

    head("Section VI-B  Held-out fold: strongest configuration per task under the joint strategy")
    for key in ["correct_type", "correct_color", "iou"]:
        rows = []
        for model, display, _ in MODELS:
            joint = max(joint_scores(runs[model]).items(), key=lambda kv: kv[1])[0]
            d = runs[model][joint]
            mask = fold_mask(d, data.split)
            lo, hi = cluster_ci(d[key][mask], d["segment"][mask], data.args.n_boot)
            rows.append((d[key][mask].mean(), half_width(lo, hi), display, joint))
        rows.sort(reverse=True)
        print(f"{key:<14}" + "   ".join(f"{name} ({s}) {r3(v)}±{r3(hw)}" for v, hw, name, s in rows[:3]))

    head("Section VI-B  Full set against the held-out fold, single-task runs")
    changes, chosen_full, chosen_fold = [], {}, {}
    for model, display, _ in MODELS:
        for strategy, d in runs[model].items():
            mask = fold_mask(d, data.split)
            for key, _ in TASKS:
                full, fold = d[key].mean(), d[key][mask].mean()
                if full or fold:
                    changes.append((abs(full - fold), display, strategy, key))
        idx = fold_mask(next(iter(runs[model].values())), data.split)
        chosen_full[model] = max(joint_scores(runs[model]).items(), key=lambda kv: kv[1])[0]
        chosen_fold[model] = max(joint_scores(runs[model], idx).items(), key=lambda kv: kv[1])[0]
    changes.sort(reverse=True)
    big = [c for c in changes if c[0] > 0.05]
    print(f"median change {np.median([c[0] for c in changes]):.3f} over {len(changes)} metrics; "
          f"{len(big)} above 0.05, at most {changes[0][0]:.3f}")
    for c in big:
        print(f"   {c[1]:<16}{c[2]:<14}{c[3]:<14}{c[0]:.3f}")
    print(f"joint strategy unchanged on the fold for {sum(chosen_full[m] == chosen_fold[m] for m in chosen_full)} "
          f"of {len(chosen_full)} models")

    head("Section VI-C  Prompt sensitivity within a model and task")
    spread, structured = [], []
    for model, display, _ in MODELS:
        for key in ["correct_type", "correct_color"]:
            values = {s: r3(d[key].mean()) for s, d in runs[model].items()}
            best, worst = max(values, key=values.get), min(values, key=values.get)
            spread.append((values[best] - values[worst], display, key, best, worst))
            if {"constrained", "descriptive"} <= set(values):
                structured.append((abs(values["constrained"] - values["descriptive"]), display, key,
                                   "constrained", "descriptive"))
    for name, rows in [("best and worst strategy", spread), ("constrained and descriptive", structured)]:
        diff, display, key, a, b = max(rows)
        print(f"largest difference in accuracy between {name}: {diff} ({display}, {key}, {a} - {b})")

    head("Section VI-D  Multi-task output format (Type:, Color:, Box: all present)")
    for model in ["INTERNVL2-8B", "MINICPM-LLAMA3-V-2.5", "QWEN2.5-VL-7B"]:
        rows = data.raw(model, "descriptive_multitask")
        kept = [all(re.search(p, r.get("raw_multitask") or "", re.I)
                    for p in [r"type\s*:", r"colou?r\s*:", r"(box|bbox|bounding box)\s*:"]) for r in rows]
        print(f"{DISPLAY[model]:<16}{pct(np.mean(kept))}")


# ── Section VII-A ──────────────────────────────────────────────────────────

WEARS_PINK = re.compile(r"\bwear\w*\s+(?:an?\s+)?(?:\w+\s+)?pink", re.I)


def section7a(data):
    head("Section VII-A  Label-only colour responses saying that someone wears pink")
    for model in ["LLAVA-1.6-MISTRAL-7B", "QWEN2.5-VL-7B", "QWEN3-VL-8B"]:
        rows = data.raw(model, "label_only")
        share = np.mean([bool(WEARS_PINK.search((r.get("raw_color") or "").replace("*", ""))) for r in rows])
        print(f"{DISPLAY[model]:<16}{pct(share)}")

    head("Section VII-A  LLaVA-1.6-7B: share of each class assigned to pink")
    for level in ["L1", "L2", "L4"]:
        d = data.ablation("LLAVA-1.6-MISTRAL-7B", level)
        cells = []
        for cls in ["black", "brown", "nude", "white"]:
            mask = d["gt_color"] == cls
            to_pink = d["pred_color"][mask] == "pink"
            cells.append(f"{cls} {pct(to_pink.mean())} ({int(to_pink.sum())})")
        print(f"{level}  " + "   ".join(cells))

    head("Section VII-A  CLIP ViT-L/14: brown garments with pink removed from the candidate labels")
    moved, other = [], []
    for r in data.raw("CLIP-VIT-L14", "descriptive"):
        if r["gt_color"] != "brown":
            continue
        scores = {k.split()[-2]: v for k, v in json.loads(r["raw_color"]).items()}
        top = max(scores, key=scores.get)
        if top == "pink":
            del scores["pink"]
            moved.append(max(scores, key=scores.get) == "nude")
        else:
            other.append(top == "nude")
    print(f"assigned to pink, nude once pink is removed: {pct(np.mean(moved))} of {len(moved)}")
    print(f"other brown garments assigned to nude: {pct(np.mean(other))} of {len(other)}")

    head("Section VII-A  Brown garments: full frame (L1) to polygon (L2)")
    pooled = [0, 0]
    for model in ["QWEN2.5-VL-7B", "QWEN3-VL-8B", "LLAVA-1.6-MISTRAL-7B"]:
        l1, l2 = data.ablation(model, "L1"), data.ablation(model, "L2")
        brown = l1["gt_color"] == "brown"
        before, after = l1["pred_color"][brown], l2["pred_color"][align(l1, l2)][brown]
        cells = []
        for name in ["brown", "pink"]:
            mask = before == name
            if mask.any():
                cells.append(f"named {name} on L1: {pct((after[mask] == 'nude').mean())} to nude ({int(mask.sum())})")
        if model.startswith("QWEN"):
            mask = before == "brown"
            pooled[0] += int((after[mask] == "nude").sum())
            pooled[1] += int(mask.sum())
        print(f"{DISPLAY[model]:<16}" + "   ".join(cells))
    print(f"two Qwen models pooled: {pct(pooled[0] / pooled[1])}")

    head("Section VII-A  Masked share of the frame")
    import cv2
    area_json = json.load(open(Path(data.args.metadata) / "working_area.json", encoding="utf-8"))
    width, height = area_json["frame_size"]
    polygon_share = {k: cv2.fillPoly(np.zeros((height, width), np.uint8), [np.array(v, np.int32)], 1).mean()
                     for k, v in area_json["polygons"].items()}
    det = data.detector("yolo-world-S")
    kept = [area(r["pred_polygon"]) / (width * height) if r["pred_polygon"] else 1.0 for r in det]
    print(f"polygon (L2): {pct(1 - np.mean([polygon_share[r['polygon']] for r in det]))}")
    print(f"predicted box (L3): {pct(1 - np.mean(kept))}")

    head("Section VII-A  Operator inside the region (share of the detected person box)")
    det_s = data.detector("yolo-world-S")
    in_gt = [overlap(r["gt_bbox_xyxy"], r["person_bbox_xyxy"]) / (area(r["person_bbox_xyxy"]) + 1e-9)
             if r["person_bbox_xyxy"] else 0.0 for r in det_s]
    print(f"ground-truth box, person from YOLO-World-S: median {pct(np.median(in_gt))}")
    for name in ["grounding-dino", "yolo-world-L", "yolo-world-S"]:
        det = data.detector(name)
        share = [overlap(r["pred_polygon"], r["person_bbox_xyxy"]) / (area(r["person_bbox_xyxy"]) + 1e-9)
                 if r["pred_polygon"] and r["person_bbox_xyxy"] else 0.0 for r in det]
        ratio = [area(r["pred_polygon"]) / area(r["gt_bbox_xyxy"]) for r in det if r["pred_polygon"]]
        print(f"{name:<16}predicted box: median person share {pct(np.median(share))}, "
              f"median box/annotation area {np.median(ratio):.2f}x")

    head("Section VII-A  Grounding DINO boxes: clean isolation")
    modes = {"clean": 0, "operator": 0, "enlarged": 0, "garment missed": 0, "no box": 0}
    det = data.detector("grounding-dino")
    for r in det:
        box, gt, person = r["pred_polygon"], r["gt_bbox_xyxy"], r["person_bbox_xyxy"]
        if box is None:
            modes["no box"] += 1
        elif overlap(gt, box) / (area(gt) + 1e-9) < 0.3:
            modes["garment missed"] += 1
        elif person and overlap(box, person) / (area(person) + 1e-9) >= 0.3:
            modes["operator"] += 1
        elif area(box) / (area(gt) + 1e-9) >= 2.5:
            modes["enlarged"] += 1
        else:
            modes["clean"] += 1
    print("   ".join(f"{k} {pct(v / len(det))}" for k, v in modes.items()))
    rest = len(det) - modes["clean"] - modes["operator"]
    print(f"remaining {pct(rest / len(det))}, of which the garment is missed in {modes['garment missed']} of {rest}")
    print("clean: box covers at least 30% of the garment and holds less than 30% of the person box")


# ── Section VII-B ──────────────────────────────────────────────────────────

PURPLE = re.compile(r"\b(purple|mauve|lavender)\b", re.I)
TAUPE = re.compile(r"\btaupe\b", re.I)
POLO = re.compile(r"\b(polo|tank\s*-?\s*top)\b", re.I)


def section7b(data):
    head("Section VII-B  Nude-brown transfers in the main benchmark (nude->br, br->nude)")
    for model, strategies in [("QWEN2.5-VL-7B", ["descriptive", "constrained"]),
                              ("INTERNVL2-8B", ["descriptive", "constrained"]),
                              ("PALIGEMMA", ["descriptive"])]:
        for s in strategies:
            d = data.main(model, s)
            print(f"{DISPLAY[model]:<16}{s:<13}{pct(transfer_share(d, 'nude', 'brown')):>7}"
                  f"{pct(transfer_share(d, 'brown', 'nude')):>7}   colour {r3(d['correct_color'].mean())}")

    head("Section VII-B  Question strategy: brown garments named purple, mauve or lavender (taupe)")
    other_max = 0.0
    for model in ["LLAVA-1.6-MISTRAL-7B", "MINICPM-LLAMA3-V-2.5", "QWEN2.5-VL-7B", "QWEN3-VL-8B"]:
        rows = data.raw(model, "question")
        for cls in ["black", "brown", "nude", "white", "pink"]:
            text = [r.get("raw_color") or "" for r in rows if r["gt_color"] == cls]
            purple = np.mean([bool(PURPLE.search(t)) for t in text])
            taupe = np.mean([bool(TAUPE.search(t)) for t in text])
            if cls == "brown":
                print(f"{DISPLAY[model]:<16}{pct(purple)} ({pct(taupe)} taupe)")
            else:
                other_max = max(other_max, np.mean([bool(PURPLE.search(t) or TAUPE.search(t)) for t in text]))
    print(f"any of these terms for another class: at most {pct(other_max)}")

    head("Section VII-B  Qwen3-VL-8B: brown garments moved to nude at the polygon level")
    question = {r["file_name"]: r.get("raw_color") or "" for r in data.raw("QWEN3-VL-8B", "question")}
    l2 = data.ablation("QWEN3-VL-8B", "L2")
    groups = {"mauve": lambda t: bool(re.search(r"\bmauve\b", t, re.I)),
              "brown": lambda t: parse_color(t) == "brown"}
    for name, test in groups.items():
        mask = np.array([l2["gt_color"][i] == "brown" and test(question[f]) for i, f in enumerate(l2["file_name"])])
        print(f"called {name} under the question strategy: {pct((l2['pred_color'][mask] == 'nude').mean())} "
              f"of {int(mask.sum())}")

    head("Section VII-B  Recall of the nude class")
    for model, s in [("QWEN3-VL-8B", "constrained"), ("QWEN3-VL-8B", "descriptive"), ("QWEN2.5-VL-7B", "descriptive")]:
        d = data.main(model, s)
        print(f"{DISPLAY[model]:<16}{s:<13}{r3((d['pred_color'][d['gt_color'] == 'nude'] == 'nude').mean())}")
    d = load(data.root / "catalog" / "color" / "QWEN3-VL-8B_descriptive_predictions.jsonl")
    print(f"{'Qwen3-VL-8B':<16}{'catalog':<13}{r3((d['pred_color'][d['gt_color'] == 'nude'] == 'nude').mean())}")

    head("Section VII-B  Catalog type subset, descriptive-cat: polo and tank top")
    for model in ["QWEN2.5-VL-7B", "QWEN3-VL-8B"]:
        rows = data.raw(model, "descriptive_cat", "catalog/type")
        polo = [r for r in rows if re.search(r"\bpolo\b", r["raw_type"] or "", re.I)]
        tank = [r for r in rows if re.search(r"\btank\s*-?\s*top\b", r["raw_type"] or "", re.I)]
        errors = [r for r in rows if parse_type(r["raw_type"]) != r["gt_type"]]
        mapped = np.mean([parse_type(POLO.sub("tshirt", r["raw_type"] or "")) == r["gt_type"] for r in rows])
        print(f"{DISPLAY[model]:<16}polo {len(polo)} of {len(rows)} ({sum(r['gt_type'] == 'tshirt' for r in polo)} "
              f"annotated t-shirt), tank top {len(tank)}; errors {len(errors)}, "
              f"of which polo {sum(bool(re.search(r'polo', r['raw_type'] or '', re.I)) for r in errors)}; "
              f"type {r3(np.mean([parse_type(r['raw_type']) == r['gt_type'] for r in rows]))} -> {r3(mapped)} "
              "with polo and tank top mapped to tshirt")


# ── Section VII-C ──────────────────────────────────────────────────────────

def type_matches(raw):
    text = raw or ""
    for pattern, replacement in TYPE_RULES:
        text = pattern.sub(replacement, text)
    return [label for label in TYPE_LABELS if re.search(r"\b" + label + r"\b", text.lower())]


CONTRAST = re.compile(r"\b(?:rather than|instead of|not)\s+(?:an?\s+)?(?:t[\s\-_]?shirts?|sweat[\s\-_]?shirts?)\b", re.I)


def section7c(data):
    head("Section VII-C  MiniCPM-V-2.5, label-only type responses")
    rows = data.raw("MINICPM-LLAMA3-V-2.5", "label_only")
    matches = [len(type_matches(r["raw_type"])) for r in rows]
    contrast = [m == 2 and parse_type(CONTRAST.sub(" ", r["raw_type"])) is not None for r, m in zip(rows, matches)]
    print(f"both type labels {pct(np.mean(np.array(matches) == 2))}, neither {pct(np.mean(np.array(matches) == 0))}, "
          f"unparseable {pct(1 - data.main('MINICPM-LLAMA3-V-2.5', 'label_only')['valid_type'].mean())}")
    print(f"a class stated only by contrast with the other: {pct(np.mean(contrast))} of all responses")

    head("Section VII-C  Qwen3-VL-8B, label-only type responses")
    rows = data.raw("QWEN3-VL-8B", "label_only")
    prefix = sum((r["raw_type"] or "").startswith("Based on the visual evidence in the image") for r in rows)
    print(f'"Based on the visual evidence in the image...": {prefix} of {len(rows)}')

    head("Section VII-C  Type latency, label-only relative to descriptive (mean per query)")
    for model in ["MINICPM-LLAMA3-V-2.5", "INTERNVL2-8B"]:
        mean = {s: np.mean([r["latency_type_ms"] for r in data.raw(model, s)]) for s in ["label_only", "descriptive"]}
        print(f"{DISPLAY[model]:<16}{mean['label_only'] / mean['descriptive']:.1f}x")

    head("Section VII-C  Label-only: sweatshirts labelled t-shirt, t-shirts labelled sweatshirt")
    for model, display, _ in MODELS:
        d = data.main(model, "label_only")
        sw, ts = d["gt_type"] == "sweatshirt", d["gt_type"] == "tshirt"
        print(f"{display:<16}{pct((d['pred_type'][sw] == 'tshirt').mean()):>7}"
              f"{pct((d['pred_type'][ts] == 'sweatshirt').mean()):>8}")


# ── Section VIII ───────────────────────────────────────────────────────────

def section8(data):
    runs = data.runs()
    head("Section VIII-A  Order: strategy chosen before and after normalization")
    for model, display, _ in MODELS:
        raw = {s: data.main(model, s, normalize=False) for s in runs[model]}
        before = max(joint_scores(raw).items(), key=lambda kv: kv[1])[0]
        scores = joint_scores(runs[model])
        after = max(scores, key=scores.get)
        if before != after:
            print(f"{display:<16}before {before} ({scores[before]:.3f})   after {after} ({scores[after]:.3f})")

    head("Section VIII-A  Joint strategy against the per-task optimum (instruction-tuned models)")
    names = {"correct_type": "type", "correct_color": "color", "iou": "IoU"}
    for model in INSTRUCTION_TUNED:
        scores = joint_scores(runs[model])
        joint = max(scores, key=scores.get)
        tasks = supported_tasks(runs[model])
        print(f"{DISPLAY[model]} (joint {joint}): "
              + ", ".join(f"{names[k]} {runs[model][joint][k].mean():.3f}" for k in tasks))
        for key in tasks:
            best = max(runs[model], key=lambda s: runs[model][s][key].mean())
            if best != joint:
                print(f"   {names[key]} optimum {best}: "
                      + ", ".join(f"{names[k]} {runs[model][best][k].mean() - runs[model][joint][k].mean():+.3f}"
                                  for k in tasks))

    head("Section VIII-A  The procedure: held-out fold and a random starting strategy")
    gain = procedure_gain(data.args, data.segments, data.split)
    for model in ABLATION_MODELS:
        cells = []
        for task in ["type", "color"]:
            g = gain[(task, model, "fold")]
            start = np.mean([data.main(model, s, normalize=False)[f"correct_{task}"].mean() for s in runs[model]])
            cells.append(f"{task}: fold {g['gain']:+.3f}±{g['half_width']:.3f}, "
                         f"from a random strategy {gain[(task, model, 'full')]['stages'][3] - start:+.3f}")
        print(f"{DISPLAY[model]:<16}" + "   ".join(cells))
    print("isolation applied, type/colour (share of 1000 draws of 200 frames):")
    for m in ABLATION_MODELS:
        print(f"   {DISPLAY[m]:<16}{gain[('type', m, 'full')]['p_isol']:.2f} / {gain[('color', m, 'full')]['p_isol']:.2f}")


# ── Appendix A ─────────────────────────────────────────────────────────────

def appendix_a(data):
    head("Appendix A  Garment box of the detectors: ground-truth coverage by selection rule")
    for name in ["yolo-world-S", "grounding-dino"]:
        det = data.detector(name)
        cells = []
        for rule in ["conf", "polygon"]:
            cover = [overlap(r["gt_bbox_xyxy"], r[f"pred_{rule}"]) / (area(r["gt_bbox_xyxy"]) + 1e-9) for r in det]
            cells.append(f"{rule} {pct(np.mean(cover))}")
        print(f"{name:<16}" + "   ".join(cells))

    head("Appendix A  Configurations where normalization lowers adjusted accuracy")
    for (model, strategy) in sorted(discover(data.root / "main_benchmark")):
        on, off = data.main(model, strategy), data.main(model, strategy, normalize=False)
        for key in ["correct_type", "correct_color"]:
            delta = on[key].mean() - off[key].mean()
            if delta < 0:
                print(f"{DISPLAY[model]:<16}{strategy:<22}{key:<15}{delta:+.4f}")


SECTIONS = {"3": section3, "4": section4, "5b": section5b, "5d": section5d, "5e": section5e, "6": section6,
            "7a": section7a, "7b": section7b, "7c": section7c, "8": section8, "a": appendix_a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=list(SECTIONS), default=None)
    ap.add_argument("--predictions", default="predictions")
    ap.add_argument("--metadata", default="metadata")
    ap.add_argument("--n-boot", type=int, default=10000, dest="n_boot")
    args = ap.parse_args()
    data = Data(args)
    for name in [args.section] if args.section else SECTIONS:
        SECTIONS[name](data)
    print()


if __name__ == "__main__":
    main()
