"""
Fixes the coordinate unit of the models that write coordinates as numbers.

    python -m src.calibrate_bbox

No annotation is used. For each model and prompt, 200 frames are drawn
(seed 0) and the stored raw responses are read both as absolute pixels and as
positions on a 0-1000 grid. On a 1280x720 frame a vertical coordinate above 720
cannot be a pixel and a coordinate above 1000 cannot lie on the grid, so the
reading that leaves more boxes inside the frame is taken. Absolute pixels are
kept unless the grid raises the share of in-frame boxes by more than one point.

The result for the free-text models is the BBOX_GRID_1000 table of
src/parsing.py. The two Qwen models are listed as a check: their unit is set in
the model wrappers, and the same rule recovers it.
"""
import argparse
import json
import random
import re
from pathlib import Path

from src.parsing import (BBOX_GRID_1000, FREE_TEXT_BBOX_MODELS, bbox_text,
                         extract_bbox, parse_bbox)

QWEN = {"QWEN2.5-VL-7B": "absolute pixels", "QWEN3-VL-8B": "0-1000 grid"}


def qwen_text(record):
    """The bbox_2d field of a Qwen response, as a bracketed group."""
    raw = record.get("raw_detection") or record.get("raw_multitask") or ""
    match = re.search(r'"bbox_2d"\s*:\s*(\[[^\]]+\])', raw)
    return match.group(1) if match else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="predictions")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.01)
    args = ap.parse_args()

    print(f'{"configuration":<44}{"in frame as":>14}{"":>9}   {"reading":<17}{"in use"}')
    print(f'{"":<44}{"pixels":>14}{"grid":>9}')
    mismatch = 0
    for path in sorted((Path(args.predictions) / "main_benchmark").glob("*_predictions.jsonl")):
        name = path.name.replace("_predictions.jsonl", "")
        free_text = name.startswith(FREE_TEXT_BBOX_MODELS)
        if not free_text and not name.startswith(tuple(QWEN)):
            continue
        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        random.seed(args.seed)
        frames = random.sample(range(len(records)), args.n)
        text = bbox_text if free_text else qwen_text
        share = {grid: sum(parse_bbox(text(records[i]), grid) is not None for i in frames) / args.n
                 for grid in (False, True)}
        reading = "0-1000 grid" if share[True] - share[False] > args.margin else "absolute pixels"

        model = next(m for m in FREE_TEXT_BBOX_MODELS + tuple(QWEN) if name.startswith(m))
        strategy = name[len(model) + 1:]
        if free_text:
            in_use = "0-1000 grid" if (model, strategy) in BBOX_GRID_1000 else "absolute pixels"
        else:
            in_use = QWEN[model]
        flag = "" if reading == in_use else "   <-- differs"
        mismatch += reading != in_use
        print(f"{name:<44}{share[False]:>14.3f}{share[True]:>9.3f}   {reading:<17}{in_use}{flag}")

    print("\nall readings match the units in use" if not mismatch
          else f"\n{mismatch} configuration(s) differ from the units in use")


if __name__ == "__main__":
    main()
