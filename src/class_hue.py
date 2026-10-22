"""
Colour of each annotated class as the camera records it (Section III): the mean
RGB over the central half of the annotated box in each dimension, away from the
hands and the operator, converted to hue, lightness and saturation, with the
median taken per class. Needs the production images.

    python -m src.class_hue --data-dir ./data
"""

import argparse
import colorsys
import csv
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["black", "brown", "nude", "white", "pink"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--metadata", default="metadata")
    args = ap.parse_args()

    frames = list(csv.DictReader(open(Path(args.metadata) / "frames.csv", encoding="utf-8")))
    per_class = {c: [] for c in CLASSES}
    for i, f in enumerate(frames, 1):
        image = np.asarray(Image.open(Path(args.data_dir) / f["file_name"]).convert("RGB"), dtype=float)
        x1, y1, x2, y2 = (int(float(f[k])) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
        w, h = x2 - x1, y2 - y1
        centre = image[y1 + h // 4: y2 - h // 4, x1 + w // 4: x2 - w // 4].reshape(-1, 3)
        if len(centre):
            hue, lightness, saturation = colorsys.rgb_to_hls(*(centre.mean(0) / 255))
            per_class[f["gt_color"]].append((hue * 360, lightness, saturation))
        print(f"\r{i}/{len(frames)}", end="", flush=True)
    print()

    print(f'{"class":<8}{"n":>6}{"hue":>8}{"lightness":>11}{"saturation":>12}')
    for cls in CLASSES:
        a = np.array(per_class[cls])
        if not len(a):
            continue
        print(f"{cls:<8}{len(a):>6}{np.median(a[:, 0]):>7.0f}°{np.median(a[:, 1]):>11.2f}{np.median(a[:, 2]):>12.2f}")


if __name__ == "__main__":
    main()
