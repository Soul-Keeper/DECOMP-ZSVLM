"""
Open-vocabulary detectors: the localization references of Table 4, the boxes of
the spatial ablation (YOLO-World-S, level L3) and the person boxes used to
measure the operator in the frame.

    python -m src.run_detectors --detector yolo-world-S   --data-dir ./data
    python -m src.run_detectors --detector yolo-world-L   --data-dir ./data
    python -m src.run_detectors --detector grounding-dino --data-dir ./data

YOLO-World runs in the environment of requirements/detectors.txt, Grounding
DINO in that of requirements/models-core.txt. Each detector is queried with the
two garment class names, a generic "folded garment" and "person". Person boxes
are not garment candidates; the most confident one is kept as the operator.

Detectors return several garment boxes per frame. All are stored, and one is
selected by each of five rules, so that the rules can be compared without a
second run:

    conf       most confident box
    area       largest box
    polygon    largest overlap with the working-area polygon (used throughout)
    poly_area  largest box whose centre lies inside the polygon
    union      box enclosing all garment boxes

The polygon is annotated once per camera position (metadata/working_area.json)
and uses no ground truth. Records are written to
predictions/detectors/<detector>_predictions.jsonl, one per frame.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASSES = ["t-shirt", "sweatshirt", "folded garment", "person"]
PERSON = "person"
DETECTORS = {
    "yolo-world-S": dict(kind="yolo", weights="yolov8s-worldv2.pt", conf=0.05),
    "yolo-world-L": dict(kind="yolo", weights="yolov8l-worldv2.pt", conf=0.05),
    "grounding-dino": dict(kind="gdino", weights="IDEA-Research/grounding-dino-base",
                           box_threshold=0.15, text_threshold=0.15),
}
RULES = ["conf", "area", "polygon", "poly_area", "union"]


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    if a is None or b is None:
        return 0.0
    inter = intersection(a, b)
    return inter / (area(a) + area(b) - inter + 1e-9)


def coverage(gt, box):
    """Share of the annotated box inside the selected box."""
    return 0.0 if box is None else intersection(gt, box) / (area(gt) + 1e-9)


def select(garments, rule, mask):
    """garments: list of (box, confidence)."""
    if not garments:
        return None
    if rule == "conf":
        return max(garments, key=lambda g: g[1])[0]
    if rule == "area":
        return max(garments, key=lambda g: area(g[0]))[0]
    if rule == "polygon":
        return max(garments, key=lambda g: float(
            mask[max(0, g[0][1]):g[0][3], max(0, g[0][0]):g[0][2]].sum()))[0]
    if rule == "poly_area":
        def centre_inside(box):
            x, y = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])
        inside = [g for g in garments if centre_inside(g[0])]
        return max(inside or garments, key=lambda g: area(g[0]))[0]
    if rule == "union":
        return [min(b[0] for b, _ in garments), min(b[1] for b, _ in garments),
                max(b[2] for b, _ in garments), max(b[3] for b, _ in garments)]
    raise ValueError(rule)


def yolo_world(config, device):
    from ultralytics import YOLO
    model = YOLO(config["weights"])
    model.set_classes(CLASSES)

    def detect(path):
        result = model.predict(str(path), conf=config["conf"], verbose=False, device=device)[0]
        if result.boxes is None or not len(result.boxes):
            return []
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        labels = result.boxes.cls.cpu().numpy().astype(int).tolist()
        scores = result.boxes.conf.cpu().numpy().tolist()
        return [([int(v) for v in b], float(s), CLASSES[k] == PERSON)
                for b, k, s in zip(boxes, labels, scores)]
    return detect


def grounding_dino(config, device):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    processor = AutoProcessor.from_pretrained(config["weights"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(config["weights"]).to(device).eval()
    prompt = ". ".join(CLASSES) + "."

    def detect(path):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=config["box_threshold"],
            text_threshold=config["text_threshold"], target_sizes=[image.size[::-1]])[0]
        labels = result.get("text_labels", result.get("labels", []))
        # A box belongs to the phrase it was matched with; only the person
        # phrase marks the operator.
        return [([int(v) for v in b.tolist()], float(s), PERSON in str(label).lower().split())
                for b, s, label in zip(result["boxes"], result["scores"], labels)]
    return detect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=list(DETECTORS), required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--metadata", default="metadata")
    ap.add_argument("--out-dir", default="predictions/detectors")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import cv2
    metadata = Path(args.metadata)
    frames = sorted(csv.DictReader(open(metadata / "frames.csv", encoding="utf-8")),
                    key=lambda f: f["file_name"])[:args.limit]
    area_json = json.load(open(metadata / "working_area.json", encoding="utf-8"))
    width, height = area_json["frame_size"]
    masks = {k: cv2.fillPoly(np.zeros((height, width), np.uint8), [np.array(v, np.int32)], 1)
             for k, v in area_json["polygons"].items()}
    camera = json.load(open(metadata / "polygon_assignment.json", encoding="utf-8"))

    config = DETECTORS[args.detector]
    detect = (yolo_world if config["kind"] == "yolo" else grounding_dino)(config, args.device)

    records = []
    for n, f in enumerate(frames, 1):
        boxes = detect(Path(args.data_dir) / f["file_name"])
        garments = [(b, s) for b, s, is_person in boxes if not is_person]
        persons = [(b, s) for b, s, is_person in boxes if is_person]
        gt = [int(f[k]) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        polygon = camera[f["file_name"]]
        record = {
            "file_name": f["file_name"],
            "polygon": polygon,
            "gt_bbox_xyxy": gt,
            "all_garment_boxes": [b for b, _ in garments],
            "all_garment_conf": [s for _, s in garments],
            "person_bbox_xyxy": max(persons, key=lambda p: p[1])[0] if persons else None,
            "n_garment_boxes": len(garments),
        }
        for rule in RULES:
            record[f"pred_{rule}"] = select(garments, rule, masks[polygon])
        record["pred_bbox_xyxy"] = record["pred_polygon"]
        record["iou"] = iou(record["pred_bbox_xyxy"], gt)
        record["gt_coverage"] = coverage(gt, record["pred_bbox_xyxy"])
        record["person_gt_iou"] = iou(record["person_bbox_xyxy"], gt)
        records.append(record)
        if n % 250 == 0:
            print(f"{n}/{len(frames)}", flush=True)

    out = Path(args.out_dir) / f"{args.detector}_predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    found = [r for r in records if r["pred_bbox_xyxy"]]
    print(f"{out}: {len(records)} frames, garment box in {len(found) / len(records):.1%}, "
          f"IoU_adj {np.mean([r['iou'] for r in records]):.3f}, "
          f"person in {np.mean([r['person_bbox_xyxy'] is not None for r in records]):.1%}")


if __name__ == "__main__":
    main()
