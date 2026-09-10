# Decomposing the Zero-Shot Gap of Vision-Language Models on a Garment Production Line

Code and released predictions for *Vision-Language Models for Zero-Shot Garment
Inspection: Decomposing the Gap to Supervised Baselines Under Production-Line
Conditions*.

The production images contain personal data and are not released. Everything
needed to recompute every table is here: the per-image predictions of all
model-strategy combinations, the ground truth, the parser, the prompts and the
profiling measurements. The external catalog evaluation can be reproduced end
to end, since it runs on a public dataset.

## Recomputing the tables

No GPU and no model weights are needed.

```bash
pip install -r requirements/analysis.txt
python -m src.build_metadata      # writes metadata/frames.csv and split.csv
python -m src.compute_tables      # prints Tables 4, 5, 6, 7, 8, 9, 11
python -m src.compute_tables --table 6
python -m src.compute_tables --n-boot 1000    # faster, wider intervals
```

`build_metadata` prints the frame, segment and fold counts; if they match the
paper, the release is intact.

| Table | Inputs |
|---|---|
| 4 | `main_benchmark/`, `detectors/`, `reference_points.csv` |
| 5 | `main_benchmark/`, `profiling.json` |
| 6 | `spatial_ablation/` |
| 7 | `spatial_ablation/`, `detectors/`, `working_area.json`, `polygon_assignment.json` |
| 8 | `main_benchmark/`, `catalog/` |
| 9 | `profiling.json` |
| 11 | `main_benchmark/` |

Tables 1, 2, 3 and 10 describe the data, the model pool, the strategy grid and
the normalization rules and contain no measurements.

## Layout

```
predictions/
  main_benchmark/       36 single-task + 3 multi-task runs, 2746 frames each
  spatial_ablation/     5 models x 4 isolation levels
  catalog/type|color/   4 models x 4 strategies on the external subsets
  detectors/            YOLO-World-L, YOLO-World-S, Grounding DINO
metadata/
  frames.csv            frame -> segment, ground truth, annotated box
  split.csv             segment -> train | val | test
  working_area.json     working-area polygon per camera position
  polygon_assignment.json  frame -> camera position
  reference_points.csv  fine-tuned baselines, not recomputable without images
  profiling.json        latency and peak memory on both devices
  catalog/              frozen definitions of the two external subsets
src/
  parsing.py            normalization rules and label parser
  loading.py            reads predictions, attaches metadata
  bootstrap.py          cluster bootstrap over acquisition segments
  build_metadata.py     derives frames.csv and split.csv
  compute_tables.py     regenerates every table
  dataset.py            dataset loader with segment-level splitting
  base_model.py         model interface and run-time parser
  prompts.py            all prompt formulations
  models/               one wrapper per evaluated model
  run_benchmark.py      inference driver
  build_roi_sets.py     masked image sets for the spatial ablation
  prepare_catalog.py    builds the external subsets
  label_catalog.py      manual type annotation of catalog items
  measure_compute.py    latency and memory profiling
manual_labels.json      manual type labels for the catalog
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

**`pred_*` is the run-time parse, without surface-form normalization.**
Normalization is one of the three interventions the paper measures, so it runs
afterwards rather than during inference: the reported metrics come from
re-parsing `raw_*` through `src/parsing.py`. Comparing `pred_type` against
`gt_type` directly gives the unnormalized column of Table 11, not the benchmark.

PaliGemma answers `t-shirt`, which the run-time parser rejects (`pred_type:
null`); after normalization the same output parses as `tshirt`. Its type valid
rate is therefore 0.223 in the stored fields and 1.000 in Table 4. The same
applies to the metrics JSON written by `run_benchmark`, which carries a
`"parser"` field saying so.

CLIP scores label embeddings instead of generating text, so `pred_*` is used
directly for it and no parsing applies.

## Acquisition segments

Frames come from continuous recording and are grouped into 201 acquisition
segments, identified by the capture timestamp in the file name. Frames from one
segment are not independent observations, so all confidence intervals resample
segments and the split assigns whole segments to folds. Three frames follow a
different naming convention and are grouped as `unknown`, as in
`src/dataset.py`.

Metrics on the held-out fold:

```python
from src.loading import load, load_segments, load_split, fold_mask, subset
from src.bootstrap import cluster_ci

data = load("predictions/main_benchmark/QWEN2.5-VL-7B_descriptive_predictions.jsonl",
            segments=load_segments())
fold = subset(data, fold_mask(data, load_split()))
lo, hi = cluster_ci(fold["iou"], fold["segment"])
print(f'{fold["iou"].mean():.3f} +- {(hi - lo) / 2:.3f}')
```

## Reproducing the external evaluation

This part runs end to end on public data. Download the Fashion Product Images
Dataset (the full release, not the thumbnails) and unpack it to
`./fashion-dataset`.

```bash
pip install -r requirements/models-core.txt
python -m src.prepare_catalog --src ./fashion-dataset --out ./data_ext
```

By default this rebuilds the exact subsets of the paper from
`metadata/catalog/*.csv`: 568 images at 284 per class for type, 1000 at 200 per
class for color. Pass `--resample` to draw new ones, and `--dry-run` to see the
class counts without writing anything.

Then run the four strategies on each subset. Type metrics are read from the
type subset and color metrics from the color subset; the secondary label in
each is filled in formally and carries no meaning. The catalog has no boxes, so
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

CLIP takes `contextual` in place of `constrained`. The manual type labels used
to verify the catalog metadata are in `manual_labels.json`; agreement is 99.0%
over all manual labels, and 98.9% over the 568 that ended up in the balanced
subset.

## Re-running the production benchmark

Requires the production images, which are not released. The commands are given
so the protocol is unambiguous.

```bash
python -m src.run_benchmark --models QWEN2.5-VL-7B --data-dir ./data
python -m src.run_benchmark --models INTERNVL2-8B --ablation
python -m src.run_benchmark --models INTERNVL2-8B MINICPM-LLAMA3-V-2.5 --multitask
```

Without `--strategy`, the joint strategy of the model is used, the one reported
throughout the paper.

The spatial ablation needs the masked image sets:

```bash
pip install -r requirements/detectors.txt
python -m src.build_roi_sets --data-dir ./data --out-dir ./data_roi
```

This writes `polygon/`, `pred_bbox/` and `gt_bbox/`, each the full 1280x720
frame with everything outside the region blacked out, plus `coverage.json` with
the surviving fraction of the annotated box per frame. Then run the benchmark
on each directory with the model's joint strategy.

## Environments

The libraries required by different models conflict, so four environments are
needed. Each file lists the versions used, and the header gives the torch and
flash-attention install lines.

| File | Models |
|---|---|
| `requirements/analysis.txt` | none, recomputing tables only |
| `requirements/models-core.txt` | ViLT, BLIP, CLIP, Florence-2, PaliGemma, Qwen2.5-VL, Grounding DINO |
| `requirements/models-chat.txt` | LLaVA-1.6, InternVL2, MiniCPM-V-2.5 |
| `requirements/models-qwen3.txt` | Qwen3-VL |
| `requirements/detectors.txt` | YOLO-World |

flash-attention is optional; the wrappers fall back to eager attention, which
changes latency but not the outputs.

Greedy decoding makes inference deterministic within one environment: two runs
of the same model, strategy and image set produced byte-identical outputs on a
200-image verification subset. It does not hold across environments, where a
library upgrade changed the output on 2.5% of that subset.

## Profiling

`metadata/profiling.json` holds the measurements behind Tables 5 and 9. To
re-measure:

```bash
python -m src.measure_compute --gpu-label A100 --n-measure 300
```

## Notes

Four strategies per model are released for the catalog subsets, one more than
Table 8 reports.

`src/parsing.py` is the only parser. It holds the normalization rules of
Table 10 and the label matcher, and every analysis script imports it, so the
rules cannot drift between them.
