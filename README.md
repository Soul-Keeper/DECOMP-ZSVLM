# Decomposing the Zero-Shot Gap of Vision-Language Models on a Garment Production Line

Code and released predictions for *Vision-Language Models for Zero-Shot Garment
Inspection: Decomposing the Gap to Supervised Baselines Under Production-Line
Conditions*.

The production images contain personal data and are not released. Everything
else is here: the per-image predictions of all model-strategy combinations and
of the open-vocabulary detectors, the ground truth, the parser, the prompts,
the profiling measurements and the sample of the parser check with its manual
reading. Every zero-shot result in the paper, in the tables and in the text,
can be recomputed from them without the images. Two analyses need the images
and are released as code only: the linear probe of Table 10 and the class colour
statistics of Section III. The fine-tuned baselines are released as reported.
The external catalog evaluation runs on a public dataset and can be reproduced
end to end.

## Recomputing the results

No GPU and no model weights are needed.

```bash
pip install -r requirements/analysis.txt
python -m src.build_metadata             # derives metadata/frames.csv and split.csv
python -m src.compute_tables             # Tables 4-9, 11 and 13
python -m src.compute_tables --table 6
python -m src.compute_tables --n-boot 1000   # faster, wider intervals
python -m src.compute_text               # figures quoted in the text, by section
python -m src.compute_text --section 7a  # one section: 3, 4, 5b, 5d, 5e, 6, 7a, 7b, 7c, 8, a
python -m src.parser_check score         # check of the parser against a human reading
python -m src.parser_check sample        # redraws that sample from the predictions
```

`build_metadata` prints the frame, segment and fold counts; if they match the
paper, the release is intact. Each script takes about half a minute with the
default 10000 bootstrap resamples.

| Table | Inputs |
|---|---|
| 4 | `main_benchmark/`, `detectors/`, `reference_points.csv` |
| 5 | `main_benchmark/`, `profiling.json` |
| 6 | `spatial_ablation/` |
| 7 | `spatial_ablation/`, `detectors/`, `working_area.json` |
| 8 | `main_benchmark/`, `catalog/` |
| 9 | `main_benchmark/`, `spatial_ablation/`, `split.csv` |
| 10 | image features, see below |
| 11 | `profiling.json` |
| 13 | `main_benchmark/` |

Tables 1, 2, 3, 12 and 14 describe the data, the model pool, the strategy grid,
the normalization rules and the inference settings and contain no measurements.

`compute_text` follows the sections of the paper (`5b` is Section V-B, `a` is
Appendix A) and prints each figure with what it is computed from, including
the text patterns used for the counts drawn from the responses.

## Analyses that need the images

```bash
python -m src.clip_probe extract --data-dir ./data   # GPU, CLIP image features
python -m src.clip_probe probe                       # CPU, prints Table 10
python -m src.class_hue --data-dir ./data            # class colour statistics, Section III
```

Both run in the environment of `requirements/models-core.txt`. The probe draws
its training frames from the training fold and is evaluated on the held-out
fold; the protocol is in the docstring of `src/clip_probe.py` and in Appendix B.

## Environments

The libraries required by different models conflict, so each group has its own
environment. Torch and flash-attention install lines are in the file headers.

| File | Used for |
|---|---|
| `requirements/analysis.txt` | recomputing the results, no models |
| `requirements/models-core.txt` | ViLT, BLIP, CLIP, Florence-2, PaliGemma, Qwen2.5-VL, Grounding DINO, the linear probe, class colours |
| `requirements/models-chat.txt` | LLaVA-1.6, InternVL2, MiniCPM-V-2.5 |
| `requirements/models-qwen3.txt` | Qwen3-VL |
| `requirements/detectors.txt` | YOLO-World |

flash-attention is optional; the wrappers fall back to eager attention, which
changes latency but not the outputs. Prebuilt wheels for other CUDA and torch
combinations: https://mjunya.com/flash-attention-prebuild-wheels/

Greedy decoding makes inference deterministic within one environment: two runs
of the same model, strategy and image set produced byte-identical outputs on a
200-image verification subset. It does not hold across environments, where a
library upgrade changed the output on 2.5% of that subset.

## Layout

```
predictions/
  main_benchmark/       36 single-task + 3 multi-task runs, 2746 frames each
  spatial_ablation/     6 models x 4 isolation levels
  catalog/type|color/   5 models x 4 strategies on the external subsets
  detectors/            YOLO-World-L, YOLO-World-S, Grounding DINO
metadata/
  frames.csv            frame -> segment, ground truth, annotated box
  split.csv             segment -> train | val | test
  working_area.json     working-area polygon per camera position
  polygon_assignment.json  frame -> camera position
  reference_points.csv  fine-tuned baselines, not recomputable without images
  profiling.json        latency and peak memory on both devices
  parser_check.json     sample of the parser check and its manual reading
  manual_labels.json    manual type labels for the catalog
  catalog/              frozen definitions of the two external subsets
src/
  parsing.py            normalization rules and label parser
  loading.py            reads predictions, attaches metadata
  bootstrap.py          cluster bootstrap over acquisition segments
  build_metadata.py     derives frames.csv and split.csv
  compute_tables.py     regenerates the tables
  compute_text.py       recomputes the figures quoted in the text
  parser_check.py       draws, reads and scores the check of the parser
  calibrate_bbox.py     label-free choice of the coordinate unit
  dataset.py            dataset loader with segment-level splitting
  base_model.py         model interface and run-time parser
  prompts.py            all prompt formulations
  models/               one wrapper per evaluated model
  run_benchmark.py      inference driver
  run_detectors.py      YOLO-World and Grounding DINO
  build_roi_sets.py     masked image sets for the spatial ablation
  clip_probe.py         linear probe on CLIP features (Table 10)
  class_hue.py          class colour statistics (Section III)
  prepare_catalog.py    builds the external subsets
  label_catalog.py      manual type annotation of catalog items
  measure_compute.py    latency and memory profiling
requirements/
```

## Prediction records

One JSON object per frame:

```json
{
  "model_name": "QWEN2.5-VL-7B", "prompt_strategy": "descriptive",
  "image_id": 0, "file_name": "00M40S_1678082860_9_jpg.rf.<hash>.jpg",
  "gt_type": "sweatshirt", "gt_color": "brown", "gt_bbox_xyxy": [0, 241, 1054, 720],
  "pred_type": "sweatshirt", "pred_color": "brown", "pred_bbox_xyxy": [0, 247, 1056, 728],
  "raw_type": "Sweatshirt", "raw_color": "brown", "raw_detection": "{\"bbox_2d\": ...}",
  "raw_multitask": null,
  "type_valid": true, "color_valid": true, "detection_valid": true,
  "latency_type_ms": 610.5, "latency_color_ms": 623.4, "latency_detection_ms": 1063.4
}
```

In a deployment the parser and the normalization are one stage, and the paper
puts normalization first in that pipeline. In the experiments the two are kept
apart, because the contribution of normalization is itself one of the
measurements: `run_benchmark` records the parse without it, and the reported
metrics come from re-parsing `raw_*` through `src/parsing.py`, which applies it.
Table 13 is the difference between the two.

So `pred_*` and `*_valid` are the parse **without** normalization. PaliGemma
answers `t-shirt`, which is rejected there (`pred_type: null`) and accepted
after normalization as `tshirt`; under the descriptive prompt its type valid
rate is 0.176 in the stored fields and 1.000 in Table 4. The metrics JSON
written by `run_benchmark` carries a `"parser"` field saying the same.
Bounding boxes are likewise re-parsed from `raw_*` with the coordinate unit of
each model and prompt (`src/parsing.py`, chosen by `src/calibrate_bbox.py`).

CLIP scores label embeddings instead of generating text, so `pred_*` is used
directly for it and no parsing applies. `raw_*` holds its score for every label.

Detector records hold every garment box with its confidence, the box selected
by each of five rules, and the most confident person box; `pred_bbox_xyxy` is
the box with the largest overlap with the working-area polygon, the rule used
throughout.

## Acquisition segments

Frames come from continuous recording and are grouped into 201 acquisition
segments, identified by the capture timestamp in the file name. Frames from one
segment are not independent observations, so all confidence intervals resample
segments and the split assigns whole segments to folds. Three frames follow a
different naming convention and are grouped as `unknown`, as in
`src/dataset.py`.

Distances to the supervised baselines are measured on the held-out fold:

```python
import glob
from src.loading import load, load_segments, load_split, fold_mask, subset
from src.bootstrap import cluster_ci, half_width

segments, split = load_segments(), load_split()
for path in sorted(glob.glob("predictions/main_benchmark/*_predictions.jsonl")):
    data = load(path, segments=segments)
    fold = subset(data, fold_mask(data, split))
    cells = []
    for key in ["correct_type", "correct_color", "iou"]:
        lo, hi = cluster_ci(fold[key], fold["segment"])
        cells.append(f'{fold[key].mean():.3f}+-{half_width(lo, hi):.3f}')
    print(f'{fold["model"]:<24}{fold["strategy"]:<22}' + "  ".join(cells))
```

## Reproducing the external evaluation

This part runs end to end on public data. Download the full release of the
[Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
(not the thumbnail version) and unpack it to `./fashion-dataset`.

```bash
pip install -r requirements/models-core.txt
python -m src.prepare_catalog --src ./fashion-dataset --out ./data_ext
```

By default this rebuilds the exact subsets of the paper from
`metadata/catalog/*.csv`: 568 images at 284 per class for type, 1000 at 200 per
class for color. `--resample --manual metadata/manual_labels.json` draws new
ones, `--dry-run` reports the class counts without writing.

Then run the strategies on each subset. Type metrics are read from the type
subset and color metrics from the color subset; the secondary label in each is
filled in formally and carries no meaning. The catalog has no boxes, so
detection is skipped.

```bash
for S in descriptive descriptive_cat constrained label_only; do
  python -m src.run_benchmark --models QWEN2.5-VL-7B --strategy $S \
    --data-dir ./data_ext/type  --output-dir ./outputs_ext/type_$S  --no-detection
  python -m src.run_benchmark --models QWEN2.5-VL-7B --strategy $S \
    --data-dir ./data_ext/color --output-dir ./outputs_ext/color_$S --no-detection
done
```

On Windows:

```powershell
foreach ($S in "descriptive","descriptive_cat","constrained","label_only") {
  python -m src.run_benchmark --models QWEN2.5-VL-7B --strategy $S `
    --data-dir ./data_ext/type  --output-dir ./outputs_ext/type_$S  --no-detection
  python -m src.run_benchmark --models QWEN2.5-VL-7B --strategy $S `
    --data-dir ./data_ext/color --output-dir ./outputs_ext/color_$S --no-detection
}
```

The same loop covers CLIP ViT-L/14, LLaVA-1.6-7B, PaliGemma-3B and Qwen3-VL-8B,
each in its own environment; CLIP takes `contextual` in place of `constrained`.
The manual type labels that verify the catalog metadata are in
`metadata/manual_labels.json`; agreement is 99.0% over all manual labels, and
98.9% over the 568 in the balanced subset.

## Protocol reference

These steps need the production images and cannot be run from this release.
They are listed so the protocol is unambiguous.

```bash
python -m src.run_benchmark --models QWEN2.5-VL-7B --data-dir ./data
python -m src.run_benchmark --models INTERNVL2-8B --ablation
python -m src.run_benchmark --models INTERNVL2-8B MINICPM-LLAMA3-V-2.5 --multitask
python -m src.run_detectors --detector yolo-world-S --data-dir ./data
python -m src.build_roi_sets --data-dir ./data --out-dir ./data_roi
python -m src.measure_compute --gpu-label A100 --n-measure 300
```

Without `--strategy`, the joint strategy of the model is used, the one reported
throughout the paper. `run_detectors` writes
`predictions/detectors/<detector>_predictions.jsonl` for `yolo-world-S`,
`yolo-world-L` or `grounding-dino`. `build_roi_sets` reads the YOLO-World-S
boxes and writes `polygon/`, `pred_bbox/` and `gt_bbox/`, each the full 1280x720
frame with everything outside the region blacked out, plus `coverage.json`; the
benchmark is then run on each directory. `measure_compute` produced
`metadata/profiling.json`, and can be pointed at `./data_ext` to check that it
runs, though catalog images have different dimensions and the latencies are not
comparable with the paper.

## Notes

Four strategies per model are released for the catalog subsets, one more than
Table 8 reports.

`src/parsing.py` is the only parser. It holds the normalization rules of
Table 12 and the label matcher, and every analysis script imports it, so the
rules cannot drift between them.
