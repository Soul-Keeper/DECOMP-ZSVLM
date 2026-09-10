"""
Builds the masked image sets for the spatial ablation.

Three sets are written; the full frame is the original data and is not copied.

    polygon/      everything outside the working-area polygon is blacked out
    pred_bbox/    everything outside the YOLO-World box is blacked out
    gt_bbox/      everything outside the annotated box is blacked out

Images are masked, not cropped: the frame stays 1280x720 and the garment does
not grow, which rules out an increase in effective resolution as an explanation
for any accuracy change.

Coverage of the annotated box under each mask is written to coverage.json and
splits the analysis into a group where the target survives (>= 90%) and one
where it is partly lost.

    python -m src.build_roi_sets --data-dir ./data --out-dir ./data_roi
    python -m src.build_roi_sets --data-dir ./data --out-dir ./data_roi --preview 8
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LEVELS = ["polygon", "pred_bbox", "gt_bbox"]


def load_working_area(path):
    area = json.load(open(path, encoding="utf-8"))
    width, height = area["frame_size"]
    masks = {}
    for name, points in area["polygons"].items():
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.array(points, np.int32)], 1)
        masks[name] = mask
    return width, height, masks


def load_ground_truth(data_dir):
    data = json.load(open(Path(data_dir) / "_annotations.coco.json", encoding="utf-8"))
    boxes = {a["image_id"]: a["bbox"] for a in data.get("annotations", [])}
    out = {}
    for image in data.get("images", []):
        box = boxes.get(image["id"])
        if box:
            x, y, w, h = box
            out[image["file_name"]] = [int(x), int(y), int(x + w), int(y + h)]
    return out


def load_detector_boxes(path, rule):
    boxes = {}
    if not Path(path).exists():
        return boxes
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            boxes[r["file_name"]] = r.get(f"pred_{rule}") or r.get("pred_bbox_xyxy")
    return boxes


def box_mask(box, width, height):
    if box is None:
        return np.ones((height, width), np.uint8)
    mask = np.zeros((height, width), np.uint8)
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(int(box[2]), width), min(int(box[3]), height)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def coverage(box, mask, width, height):
    """Fraction of the annotated box left unmasked."""
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(box[2], width), min(box[3], height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].sum()) / ((x2 - x1) * (y2 - y1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./data_roi")
    ap.add_argument("--metadata", default="metadata")
    ap.add_argument("--detector", default="predictions/detectors/yolo-world-S_predictions.jsonl")
    ap.add_argument("--rule", default="polygon",
                    help="which YOLO-World selection rule to take the box from")
    ap.add_argument("--preview", type=int, default=0,
                    help="save N side-by-side strips for illustration")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    metadata = Path(args.metadata)

    width, height, polygon_masks = load_working_area(metadata / "working_area.json")
    assignment = json.load(open(metadata / "polygon_assignment.json", encoding="utf-8"))
    ground_truth = load_ground_truth(data_dir)
    detector = load_detector_boxes(args.detector, args.rule)

    print(f"working-area polygons: {sorted(polygon_masks)}")
    print(f"camera assignments: {len(assignment)}")
    print(f"detector boxes ({args.rule}): {len(detector)}")
    if not detector:
        print(f"[!] {args.detector} not found, the pred_bbox level will be skipped")

    files = sorted(p for p in data_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name in ground_truth)
    if args.limit:
        files = files[: args.limit]
    print(f"images: {len(files)}\n")

    for level in LEVELS:
        (out_dir / level).mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out_dir / "preview").mkdir(parents=True, exist_ok=True)

    coverages, missing_box = {}, 0

    for n, path in enumerate(files, 1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height))

        box = ground_truth[path.name]
        camera = assignment.get(path.name, "A")
        predicted = detector.get(path.name)
        if predicted is None:
            missing_box += 1

        masks = {
            "polygon": polygon_masks[camera],
            "pred_bbox": box_mask(predicted, width, height),
            "gt_bbox": box_mask(box, width, height),
        }

        record = {"camera": camera, "detector_box": predicted is not None}
        for level, mask in masks.items():
            if level == "pred_bbox" and not detector:
                continue
            cv2.imwrite(str(out_dir / level / path.name), image * mask[:, :, None])
            record[level] = round(coverage(box, mask, width, height), 4)
            record[f"{level}_retained"] = round(float(mask.sum()) / (width * height), 4)
        coverages[path.name] = record

        if args.preview and n <= args.preview:
            panels = [image] + [image * masks[l][:, :, None] for l in LEVELS if l in masks]
            strip = np.hstack(panels)
            strip = cv2.resize(strip, (strip.shape[1] // 2, strip.shape[0] // 2))
            for i, label in enumerate(["full frame"] + LEVELS):
                cv2.putText(strip, label, (10 + i * (width // 2), 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(out_dir / "preview" / f"strip_{path.stem}.png"), strip)

        if n % 250 == 0:
            print(f"  [{n}/{len(files)}]")

    (out_dir / "coverage.json").write_text(
        json.dumps(coverages, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nsets written to {out_dir}/{{{', '.join(LEVELS)}}}")
    print(f"coverage -> {out_dir / 'coverage.json'}")
    if missing_box:
        print(f"[i] no detector box on {missing_box} frames, full frame kept")

    print(f'\n{"level":<12}{"GT coverage":>13}{"median":>9}{">=90%":>9}{"retained":>11}')
    print("-" * 54)
    print(f'{"full frame":<12}{1.0:>12.1%}{1.0:>9.1%}{1.0:>9.1%}{1.0:>11.1%}')
    for level in LEVELS:
        values = np.array([v[level] for v in coverages.values() if level in v])
        retained = np.array([v[f"{level}_retained"] for v in coverages.values() if level in v])
        if not len(values):
            continue
        print(f'{level:<12}{values.mean():>12.1%}{np.median(values):>9.1%}'
              f'{(values >= 0.9).mean():>9.1%}{retained.mean():>11.1%}')

    print(f'\n{"level":<12}{"target intact":>15}{"partly lost":>14}')
    print("-" * 41)
    for level in LEVELS:
        values = np.array([v[level] for v in coverages.values() if level in v])
        if len(values):
            print(f'{level:<12}{(values >= 0.9).sum():>15}{(values < 0.9).sum():>14}')


if __name__ == "__main__":
    main()
