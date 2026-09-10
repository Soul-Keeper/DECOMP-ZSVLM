"""
Benchmark runner.

    python -m src.run_benchmark --models QWEN2.5-VL-7B
    python -m src.run_benchmark --models INTERNVL2-8B --ablation
    python -m src.run_benchmark --models INTERNVL2-8B MINICPM-LLAMA3-V-2.5 --multitask
    python -m src.run_benchmark --models VILT-B32-FINETUNED-VQA --limit 20

Without --strategy the joint strategy of the model is used, the one reported
throughout the paper. --ablation runs every strategy the model supports.

Writes to outputs/:
    predictions/<MODEL>_<STRATEGY>_predictions.jsonl
    metrics/<MODEL>_<STRATEGY>_metrics.json
    metrics/summary.json
    logs/benchmark_<timestamp>.log

The metrics written here use the run-time parser, without surface-form
normalization. The figures reported in the paper come from re-parsing the raw
outputs through src/parsing.py; see src/compute_tables.py.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.base_model import BenchmarkRecord, PredictionResult, Task
from src.parsing import compute_iou
from src.dataset import GarmentDataset
from src.prompts import (
    DEFAULT_STRATEGIES,
    PROMPT_REGISTRY,
    get_all_strategies,
    get_prompt,
)

logger = logging.getLogger(__name__)

# Models whose wrapper sets SUPPORTS_MULTITASK. Others fall back to three
# sequential calls, which is not a multi-task run.
MULTITASK_CAPABLE = {"INTERNVL2-8B", "MINICPM-LLAMA3-V-2.5", "QWEN2.5-VL-7B"}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CLASSES = {
    "VILT-B32-FINETUNED-VQA": ("src.models.vilt", "ViLTModel"),
    "BLIP-VQA-CAPFILT-LARGE": ("src.models.blip", "BLIPModel"),
    "FLORENCE-2-LARGE": ("src.models.florence2", "FlorenceModel"),
    "CLIP-VIT-L14": ("src.models.clip", "CLIPGarmentModel"),
    "PALIGEMMA": ("src.models.paligemma", "PaliGemmaModel"),
    "LLAVA-1.6-MISTRAL-7B": ("src.models.llava", "LLaVAModel"),
    "INTERNVL2-8B": ("src.models.internvl2", "InternVLModel"),
    "MINICPM-LLAMA3-V-2.5": ("src.models.minicpm", "MiniCPMModel"),
    "QWEN2.5-VL-7B": ("src.models.qwen25", "QwenVLModel"),
    "QWEN3-VL-8B": ("src.models.qwen3", "Qwen3VLModel"),
}


def _load_model_class(model_name: str):
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model '{model_name}'.")
    module, class_name = MODEL_CLASSES[model_name]
    return getattr(import_module(module), class_name)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(records: list[BenchmarkRecord]) -> dict:
    """
    Compute accuracy and IoU from a list of BenchmarkRecords.

    Reported metrics:
      accuracy        — correct / valid predictions   (denominator = parseable outputs)
      adjusted_accuracy — correct / all samples       (denominator = full dataset)
    """
    import numpy as np

    n = len(records)
    if n == 0:
        return {}

    type_preds  = [(r.pred_type,  r.gt_type)  for r in records if r.type_valid]
    color_preds = [(r.pred_color, r.gt_color) for r in records if r.color_valid]
    det_records = [r for r in records if r.detection_valid]

    type_acc  = sum(p == g for p, g in type_preds)  / len(type_preds)  if type_preds  else None
    color_acc = sum(p == g for p, g in color_preds) / len(color_preds) if color_preds else None

    type_valid_rate  = len(type_preds)  / n
    color_valid_rate = len(color_preds) / n
    det_valid_rate   = len(det_records) / n

    type_adj  = type_acc  * type_valid_rate  if type_acc  is not None else None
    color_adj = color_acc * color_valid_rate if color_acc is not None else None

    iou_scores = []
    for r in det_records:
        if r.pred_bbox_xyxy and r.gt_bbox_xyxy:
            iou_scores.append(compute_iou(r.gt_bbox_xyxy, r.pred_bbox_xyxy))
    mean_iou = float(np.mean(iou_scores)) if iou_scores else None

    type_per_class = {}
    for cls in ["tshirt", "sweatshirt"]:
        cls_preds = [(p, g) for p, g in type_preds if g == cls]
        if cls_preds:
            type_per_class[cls] = round(sum(p == g for p, g in cls_preds) / len(cls_preds), 4)

    color_per_class = {}
    for cls in ["black", "brown", "nude", "white", "pink"]:
        cls_preds = [(p, g) for p, g in color_preds if g == cls]
        if cls_preds:
            color_per_class[cls] = round(sum(p == g for p, g in cls_preds) / len(cls_preds), 4)

    return {
        "n_total":               n,
        "type_accuracy":         round(type_acc,  4) if type_acc  is not None else None,
        "color_accuracy":        round(color_acc, 4) if color_acc is not None else None,
        "detection_iou":         round(mean_iou,  4) if mean_iou  is not None else None,
        "type_valid_rate":       round(type_valid_rate,  4),
        "color_valid_rate":      round(color_valid_rate, 4),
        "detection_valid_rate":  round(det_valid_rate,   4),
        "type_accuracy_adj":     round(type_adj,  4) if type_adj  is not None else None,
        "color_accuracy_adj":    round(color_adj, 4) if color_adj is not None else None,
        "type_per_class":        type_per_class,
        "color_per_class":       color_per_class,
        "n_iou_samples":         len(iou_scores),
        "parser":                "run-time, no surface-form normalization",
    }


# ---------------------------------------------------------------------------
# Single-task run
# ---------------------------------------------------------------------------

def run_single(
    model_name: str,
    strategy: str,
    dataset: GarmentDataset,
    output_dir: Path,
    device: str,
    limit: int | None = None,
    skip_detection: bool = False,
) -> dict:
    """Run one model with one prompt strategy, three separate calls per image."""
    logger.info(f"\n{'='*60}")
    logger.info(f"[SINGLE-TASK] Model: {model_name}  Strategy: {strategy}")
    logger.info(f"Samples: {limit or len(dataset)}")
    logger.info(f"{'='*60}")

    tag          = f"{model_name}_{strategy}"
    pred_path    = output_dir / "predictions" / f"{tag}_predictions.jsonl"
    metrics_path = output_dir / "metrics"     / f"{tag}_metrics.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    ModelClass = _load_model_class(model_name)
    model = ModelClass(device=device)
    model.load()
    model.reset_peak_memory()

    prompt_config = get_prompt(model_name, strategy)
    records: list[BenchmarkRecord] = []

    n = min(limit, len(dataset)) if limit else len(dataset)
    t_start = time.perf_counter()

    with open(pred_path, "w") as pred_file:
        for i in range(n):
            item = dataset[i]
            image      = item["image"]
            image_size = (item.get("width", 1280), item.get("height", 720))

            r_type  = model.predict_type(image, prompt_config)
            r_color = model.predict_color(image, prompt_config)
            if skip_detection:
                r_bbox = PredictionResult(
                    task=Task.DETECTION, raw_output="", is_valid=False)
            else:
                r_bbox = model.predict_bbox(image, prompt_config, image_size)

            rec = BenchmarkRecord(
                model_name=model_name,
                prompt_strategy=strategy,
                image_id=item["image_id"],
                file_name=item["file_name"],
                gt_type=item["garment_type"],
                gt_color=item["color"],
                gt_bbox_xyxy=item["bbox_xyxy"],
                pred_type=r_type.parsed_label,
                pred_color=r_color.parsed_label,
                pred_bbox_xyxy=r_bbox.bbox_xyxy,
                raw_type=r_type.raw_output,
                raw_color=r_color.raw_output,
                raw_detection=r_bbox.raw_output,
                type_valid=r_type.is_valid,
                color_valid=r_color.is_valid,
                detection_valid=r_bbox.is_valid,
                latency_type_ms=r_type.latency_ms,
                latency_color_ms=r_color.latency_ms,
                latency_detection_ms=r_bbox.latency_ms,
            )
            records.append(rec)
            pred_file.write(json.dumps(rec.__dict__) + "\n")

            if (i + 1) % 100 == 0 or (i + 1) == n:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f"  [{i+1}/{n}] elapsed={elapsed:.1f}s  "
                    f"avg={elapsed/(i+1)*1000:.0f}ms/img  "
                    f"gpu_peak={model.gpu_memory_peak_mb() or 0:.0f}MB"
                )

    return _finalise(model, records, model_name, strategy, "single", metrics_path, t_start, n)


# ---------------------------------------------------------------------------
# Multi-task run
# ---------------------------------------------------------------------------

def _parse_multitask_response(raw: str, model) -> tuple[str | None, str | None, list | None]:
    """
    Parse structured multi-task response.

    Expected format (from multitask_prompt):
        Type: tshirt
        Color: black
        Box: [x_min, y_min, x_max, y_max]

    Returns (pred_type, pred_color, bbox_xyxy) — any can be None if not parsed.
    """
    import re

    pred_type  = None
    pred_color = None
    bbox_xyxy  = None

    for line in raw.splitlines():
        line = line.strip()
        low  = line.lower()

        if low.startswith("type:"):
            value = line.split(":", 1)[1].strip()
            pred_type = model._parse_classification(value, model.VALID_TYPES)

        elif low.startswith("color:"):
            value = line.split(":", 1)[1].strip()
            pred_color = model._parse_classification(value, model.VALID_COLORS)

        elif low.startswith("box:"):
            value = line.split(":", 1)[1].strip()
            bbox_xyxy = model._parse_bbox(value)

    return pred_type, pred_color, bbox_xyxy


def run_multitask(
    model_name: str,
    dataset: GarmentDataset,
    output_dir: Path,
    device: str,
    limit: int | None = None,
) -> dict:
    """
    Run multi-task inference: one model.predict_multitask() call per image.

    Uses the multitask_prompt from the model's descriptive strategy.
    Models outside MULTITASK_CAPABLE fall back to three sequential calls.
    """
    if model_name not in MULTITASK_CAPABLE:
        logger.warning(
            f"[MULTI-TASK] {model_name} is not in MULTITASK_CAPABLE. "
            f"Will use sequential fallback — results are NOT a true multi-task experiment."
        )

    # Multi-task always uses the descriptive strategy (it contains multitask_prompt)
    strategy = "descriptive"

    logger.info(f"\n{'='*60}")
    logger.info(f"[MULTI-TASK] Model: {model_name}  Strategy: {strategy}")
    logger.info(f"Samples: {limit or len(dataset)}")
    logger.info(f"{'='*60}")

    tag          = f"{model_name}_{strategy}_multitask"
    pred_path    = output_dir / "predictions" / f"{tag}_predictions.jsonl"
    metrics_path = output_dir / "metrics"     / f"{tag}_metrics.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    ModelClass = _load_model_class(model_name)
    model = ModelClass(device=device)
    model.load()
    model.reset_peak_memory()

    prompt_config = get_prompt(model_name, strategy)

    # Verify multitask_prompt is set
    if not prompt_config.multitask_prompt:
        raise ValueError(
            f"{model_name} / {strategy} has no multitask_prompt in src/prompts.py"
        )

    records: list[BenchmarkRecord] = []
    n = min(limit, len(dataset)) if limit else len(dataset)
    t_start = time.perf_counter()

    with open(pred_path, "w") as pred_file:
        for i in range(n):
            item = dataset[i]
            image      = item["image"]
            image_size = (item.get("width", 1280), item.get("height", 720))

            # Single call — model decides internally whether to do 1 or 3 passes
            r_multi = model.predict_multitask(image, prompt_config, image_size)

            # Parse structured output (Type:/Color:/Box: lines)
            pred_type, pred_color, bbox_xyxy = _parse_multitask_response(
                r_multi.raw_output, model
            )

            # If the model already parsed (e.g. fallback fills parsed_type), prefer that
            pred_type  = pred_type  or r_multi.parsed_type
            pred_color = pred_color or r_multi.parsed_color
            bbox_xyxy  = bbox_xyxy  or r_multi.bbox_xyxy

            rec = BenchmarkRecord(
                model_name=model_name,
                prompt_strategy=tag,          # includes "multitask" suffix for clarity
                image_id=item["image_id"],
                file_name=item["file_name"],
                gt_type=item["garment_type"],
                gt_color=item["color"],
                gt_bbox_xyxy=item["bbox_xyxy"],
                pred_type=pred_type,
                pred_color=pred_color,
                pred_bbox_xyxy=bbox_xyxy,
                raw_multitask=r_multi.raw_output,
                # Single-task fields are empty — this is a different experiment
                raw_type=None,
                raw_color=None,
                raw_detection=None,
                type_valid=(pred_type is not None),
                color_valid=(pred_color is not None),
                detection_valid=(bbox_xyxy is not None),
                latency_type_ms=r_multi.latency_ms / 3,    # total split evenly for reporting
                latency_color_ms=r_multi.latency_ms / 3,
                latency_detection_ms=r_multi.latency_ms / 3,
            )
            records.append(rec)
            pred_file.write(json.dumps(rec.__dict__) + "\n")

            if (i + 1) % 100 == 0 or (i + 1) == n:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f"  [{i+1}/{n}] elapsed={elapsed:.1f}s  "
                    f"avg={elapsed/(i+1)*1000:.0f}ms/img  "
                    f"gpu_peak={model.gpu_memory_peak_mb() or 0:.0f}MB"
                )

    return _finalise(model, records, model_name, tag, "multitask", metrics_path, t_start, n)


# ---------------------------------------------------------------------------
# Shared finalisation (metrics + unload)
# ---------------------------------------------------------------------------

def _finalise(
    model,
    records: list[BenchmarkRecord],
    model_name: str,
    strategy_tag: str,
    mode: str,
    metrics_path: Path,
    t_start: float,
    n: int,
) -> dict:
    total_time = time.perf_counter() - t_start
    metrics = compute_metrics(records)
    metrics["model_name"]      = model_name
    metrics["prompt_strategy"] = strategy_tag
    metrics["mode"]            = mode          # "single" | "multitask"
    metrics["total_time_s"]    = round(total_time, 2)
    metrics["avg_latency_ms"]  = round(total_time / n * 1000, 1)
    metrics["gpu_peak_mb"]     = model.gpu_memory_peak_mb()

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nResults [{mode}] {model_name} / {strategy_tag}:")
    logger.info(f"  Type  accuracy (adj): {metrics.get('type_accuracy')} ({metrics.get('type_accuracy_adj')})")
    logger.info(f"  Color accuracy (adj): {metrics.get('color_accuracy')} ({metrics.get('color_accuracy_adj')})")
    logger.info(f"  Detection IoU       : {metrics.get('detection_iou')}")
    logger.info(f"  Total time          : {total_time:.1f}s  ({metrics['avg_latency_ms']}ms/img)")
    logger.info(f"  Metrics -> {metrics_path}")

    model.unload()
    return metrics


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def save_summary(all_metrics: list[dict], output_dir: Path) -> None:
    summary_path = output_dir / "metrics" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "results": all_metrics}, f, indent=2)

    header = f"\n{'Model':<35} {'Strategy':<30} {'Mode':<12} {'Type':>8} {'Color':>8} {'IoU':>8} {'ms/img':>8}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(
            f"{m['model_name']:<35} "
            f"{m['prompt_strategy']:<30} "
            f"{m.get('mode', '-'):<12} "
            f"{str(m.get('type_accuracy', '-')):>8} "
            f"{str(m.get('color_accuracy', '-')):>8} "
            f"{str(m.get('detection_iou', '-')):>8} "
            f"{str(m.get('avg_latency_ms', '-')):>8}"
        )
    print(f"\nSummary -> {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Garment VLM Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=list(PROMPT_REGISTRY.keys()) + ["ALL"],
        help="Models to evaluate.",
    )

    # --- Experiment mode (mutually exclusive) ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ablation", action="store_true",
        help="Run every prompt strategy the model supports.",
    )
    mode_group.add_argument(
        "--multitask", action="store_true",
        help=("One predict_multitask() call per image. Genuine single-pass "
              f"only for {sorted(MULTITASK_CAPABLE)}."),
    )

    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Override prompt strategy (single-task only).",
    )
    parser.add_argument(
        "--split", default="all",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--data-dir",   type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device",     type=str, default="cuda")

    parser.add_argument(
        "--no-detection", action="store_true",
        help="Skip box prediction, for data where detection is not evaluated.",
    )

    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"benchmark_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(f"Logging to {log_path}")


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    setup_logging(output_dir)

    model_names = list(PROMPT_REGISTRY.keys()) if "ALL" in args.models else args.models

    data_dir = ROOT / args.data_dir
    logger.info(f"Loading dataset from {data_dir} (split={args.split})")
    dataset = GarmentDataset(data_dir=data_dir, split=args.split)
    logger.info(repr(dataset))

    all_metrics = []

    for model_name in model_names:

        if args.multitask:
            # ---- Multi-task experiment ----
            try:
                metrics = run_multitask(
                    model_name=model_name,
                    dataset=dataset,
                    output_dir=output_dir,
                    device=args.device,
                    limit=args.limit,
                )
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"FAILED multitask: {model_name} — {e}", exc_info=True)

        else:
            # ---- Single-task experiment (default / ablation) ----
            if args.ablation:
                strategies = [pc.strategy for pc in get_all_strategies(model_name)]
            elif args.strategy:
                strategies = [args.strategy]
            else:
                strategies = [DEFAULT_STRATEGIES[model_name]]

            for strategy in strategies:
                try:
                    metrics = run_single(
                        model_name=model_name,
                        strategy=strategy,
                        dataset=dataset,
                        output_dir=output_dir,
                        device=args.device,
                        limit=args.limit,
                        skip_detection=args.no_detection,
                    )
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"FAILED: {model_name} [{strategy}] — {e}", exc_info=True)

    save_summary(all_metrics, output_dir)


if __name__ == "__main__":
    main()
