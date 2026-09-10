"""
measure_compute.py — Computational benchmarking for all models.

Measures inference latency and GPU memory for each model on the current device.
Run this script SEPARATELY on each GPU to get comparable numbers:

    # On Tesla A100 (all models)
    python src/measure_compute.py --gpu-label A100

    # On RTX 4060 Ti (consumer models only — InternVL2/MiniCPM will OOM)
    python src/measure_compute.py --gpu-label 4060Ti --skip-large

Results are saved to:
    outputs/compute/compute_{gpu_label}.json   — raw per-model numbers
    outputs/compute/compute_combined.json      — merged A100 + 4060Ti table
                                                 (run after both measurements)

Protocol
--------
- N images measured after a 10-image warmup (--n-measure, default 500)
- Each model loaded fresh, GPU cache cleared before and after
- Latency  = wall-clock time; CUDA sync BEFORE starting the timer and after
             the call, so queued work is not charged to the measurement
- VRAM     = peak GPU memory allocated during inference (max_memory_allocated)
- Throughput = images/second
- Images sampled uniformly from the full dataset (seed=42)
- Prompt   = per-model joint strategy (JOINT_STRATEGIES), the same one used
             in every table of the paper. Latency depends strongly on the
             prompt, so this must match the analysis.

Run in EXCLUSIVE mode: concurrent processes compete for SMs and distort
the numbers.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import GarmentDataset
from src.prompt_configs import DEFAULT_STRATEGIES, get_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models and their expected VRAM — used to skip OOM candidates
# ---------------------------------------------------------------------------

# (module_path, class_name, approx_vram_gb)
MODEL_REGISTRY = {
    "VILT-B32-FINETUNED-VQA":   ("src.models.vilt_model",     "ViLTModel",        2),
    "BLIP-VQA-CAPFILT-LARGE":   ("src.models.blip_model",     "BLIPModel",        4),
    "FLORENCE-2-LARGE":         ("src.models.florence_model", "FlorenceModel",    6),
    "CLIP-VIT-L14":             ("src.models.clip_model",     "CLIPGarmentModel", 2),
    "PALIGEMMA":                ("src.models.paligemma_model","PaliGemmaModel",   8),
    "LLAVA-1.6-MISTRAL-7B":     ("src.models.llava_model",    "LLaVAModel",       10),
    "INTERNVL2-8B":             ("src.models.internvl_model", "InternVLModel",    18),
    "MINICPM-LLAMA3-V-2.5":     ("src.models.minicpm_model",  "MiniCPMModel",     20),
    "QWEN2.5-VL-7B":            ("src.models.qwen_model",     "QwenVLModel",      18),
    "QWEN3-VL-8B":              ("src.models.qwen3_model",    "Qwen3VLModel",     18),
}

LARGE_MODELS = {"INTERNVL2-8B", "MINICPM-LLAMA3-V-2.5",
                "QWEN2.5-VL-7B", "QWEN3-VL-8B"}

# Модели, поддерживающие multi-task инференс в один проход
MULTITASK_MODELS = {"INTERNVL2-8B", "MINICPM-LLAMA3-V-2.5",
                    "QWEN2.5-VL-7B", "QWEN3-VL-8B"}

# Supervised baselines — handled separately (no prompt_config needed)
BASELINE_REGISTRY = {
    "resnet50": "outputs/baselines/resnet50_best.pt",
    "vit_b16":  "outputs/baselines/vit_b16_best.pt",
    "yolo11n":  "outputs/yolo_runs/det/weights/best.pt",
}

N_WARMUP  = 10
N_MEASURE = 500          # значение по умолчанию, переопределяется --n-measure

# ---------------------------------------------------------------------------
# Единая стратегия на модель — та же, что используется во всех таблицах
# статьи (правило: максимум среднего adjusted accuracy по доступным задачам).
# DEFAULT_STRATEGIES из prompt_configs расходится с ней у BLIP, Florence-2,
# PaliGemma и LLaVA, поэтому замер вёлся бы под другими промптами, чем анализ.
# Латентность сильно зависит от промпта, так что это принципиально.
# ---------------------------------------------------------------------------

JOINT_STRATEGIES = {
    "VILT-B32-FINETUNED-VQA":   "constrained",
    "BLIP-VQA-CAPFILT-LARGE":   "label_only",
    "FLORENCE-2-LARGE":         "label_only",
    "CLIP-VIT-L14":             "descriptive",
    "PALIGEMMA":                "descriptive",
    "LLAVA-1.6-MISTRAL-7B":     "descriptive",
    "INTERNVL2-8B":             "descriptive",
    "MINICPM-LLAMA3-V-2.5":     "descriptive",
    "QWEN2.5-VL-7B":            "descriptive",
    "QWEN3-VL-8B":              "descriptive",
}


# ---------------------------------------------------------------------------
# VLM measurement
# ---------------------------------------------------------------------------

def measure_vlm(model_name: str, images: list, device: str,
                n_measure: int = N_MEASURE, strategy: str | None = None,
                modes: tuple = ("single",)) -> list:
    """
    Load model once, measure the requested inference modes, return one result
    dict per mode.

    single: three separate forward passes (type, color, detection)
    multi:  one forward pass resolving all three tasks via multitask_prompt
    """
    import importlib

    module_path, class_name, _ = MODEL_REGISTRY[model_name]
    module   = importlib.import_module(module_path)
    ModelCls = getattr(module, class_name)

    model = ModelCls(device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model.load()
    load_time_s = time.perf_counter() - t_load
    vram_after_load_mb = torch.cuda.memory_allocated() / 1024**2

    strat = strategy or JOINT_STRATEGIES.get(model_name) \
        or DEFAULT_STRATEGIES[model_name]
    prompt_config = get_prompt(model_name, strat)
    logger.info(f"  Strategy: {strat}")

    results = []

    for mode in modes:
        if mode == "multi":
            if model_name not in MULTITASK_MODELS:
                logger.info(f"  Skipping multi-task: {model_name} does not support it")
                continue
            if not getattr(prompt_config, "multitask_prompt", None):
                logger.info(f"  Skipping multi-task: no multitask_prompt for {strat}")
                continue

        def one_pass(img):
            if mode == "multi":
                model.predict_multitask(img, prompt_config)
            else:
                model.predict_type(img, prompt_config)
                model.predict_color(img, prompt_config)
                model.predict_bbox(img, prompt_config)

        logger.info(f"  [{mode}] Warmup ({N_WARMUP} images)...")
        for img in images[:N_WARMUP]:
            try:
                one_pass(img)
            except Exception:
                pass

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        logger.info(f"  [{mode}] Measuring ({n_measure} images)...")
        latencies, errors = [], 0
        for img in images[:n_measure]:
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                one_pass(img)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                errors += 1
                logger.warning(f"  Error on image: {e}")

        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        if not latencies:
            results.append({"model_name": model_name, "mode": mode,
                            "error": "all inference calls failed"})
            continue

        lat = np.array(latencies)
        result = {
            "model_name":        model_name,
            "mode":              mode,
            "strategy":          strat,
            "n_measured":        len(latencies),
            "n_errors":          errors,
            "latency_mean_ms":   round(float(lat.mean()), 1),
            "latency_median_ms": round(float(np.median(lat)), 1),
            "latency_p95_ms":    round(float(np.percentile(lat, 95)), 1),
            "latency_std_ms":    round(float(lat.std()), 1),
            "throughput_img_s":  round(1000 / float(lat.mean()), 2),
            "vram_load_mb":      round(vram_after_load_mb, 0),
            "vram_peak_mb":      round(peak_vram_mb, 0),
            "load_time_s":       round(load_time_s, 1),
        }
        results.append(result)
        logger.info(
            f"  {model_name} [{mode}]: {result['latency_median_ms']}ms/img "
            f"(median)  peak={result['vram_peak_mb']}MB"
        )

    model.unload()
    torch.cuda.empty_cache()

    # speedup, если измерены оба режима
    by_mode = {r["mode"]: r for r in results if "latency_median_ms" in r}
    if "single" in by_mode and "multi" in by_mode:
        sp = by_mode["single"]["latency_median_ms"] / by_mode["multi"]["latency_median_ms"]
        for r in results:
            r["speedup"] = round(sp, 2)
        logger.info(f"  {model_name}: speedup {sp:.2f}x")

    return results


# ---------------------------------------------------------------------------
# Supervised baseline measurement
# ---------------------------------------------------------------------------

def measure_baseline_cls(tag: str, ckpt_path: Path, images: list,
                          device: str) -> dict:
    """Measure ResNet-50 or ViT-B/16 inference."""
    import torchvision.models as tv_models
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Rebuild model architecture to load checkpoint
    if tag == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
    else:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        backbone = vit_b_16(weights=None)
        feat_dim = backbone.heads.head.in_features
        backbone.heads = torch.nn.Identity()

    # Reconstruct DualHeadClassifier structure
    class DualHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone   = backbone
            self.type_head  = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat_dim, 2))
            self.color_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(feat_dim, 5))
        def forward(self, x):
            f = self.backbone(x)
            return self.type_head(f), self.color_head(f)

    model = DualHead().to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    dummy = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(N_WARMUP):
        with torch.inference_mode():
            model(dummy)

    torch.cuda.reset_peak_memory_stats()

    # Measure — single image at a time (same protocol as VLMs)
    latencies = []
    for img in images[:N_MEASURE]:
        t_img = transform(img).unsqueeze(0).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(t_img)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    del model
    torch.cuda.empty_cache()

    lat = np.array(latencies)
    result = {
        "model_name":        tag,
        "n_measured":        len(latencies),
        "latency_mean_ms":   round(float(lat.mean()), 1),
        "latency_median_ms": round(float(np.median(lat)), 1),
        "latency_p95_ms":    round(float(np.percentile(lat, 95)), 1),
        "latency_std_ms":    round(float(lat.std()), 1),
        "throughput_img_s":  round(1000 / float(lat.mean()), 2),
        "vram_load_mb":      round(peak_vram_mb, 0),
        "vram_peak_mb":      round(peak_vram_mb, 0),
        "load_time_s":       0.0,
    }
    logger.info(
        f"  {tag}: {result['latency_mean_ms']}ms/img  "
        f"peak={result['vram_peak_mb']}MB"
    )
    return result


def measure_yolo(ckpt_path: Path, images: list, device: str) -> dict:
    """Measure YOLO11n detection inference."""
    from ultralytics import YOLO

    model = YOLO(str(ckpt_path))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for img in images[:N_WARMUP]:
        model.predict(img, device=device, verbose=False)

    torch.cuda.reset_peak_memory_stats()

    latencies = []
    for img in images[:N_MEASURE]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(img, device=device, verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    lat = np.array(latencies)
    result = {
        "model_name":        "yolo11n_det",
        "n_measured":        len(latencies),
        "latency_mean_ms":   round(float(lat.mean()), 1),
        "latency_median_ms": round(float(np.median(lat)), 1),
        "latency_p95_ms":    round(float(np.percentile(lat, 95)), 1),
        "latency_std_ms":    round(float(lat.std()), 1),
        "throughput_img_s":  round(1000 / float(lat.mean()), 2),
        "vram_load_mb":      round(peak_vram_mb, 0),
        "vram_peak_mb":      round(peak_vram_mb, 0),
        "load_time_s":       0.0,
    }
    logger.info(
        f"  yolo11n: {result['latency_mean_ms']}ms/img  "
        f"peak={result['vram_peak_mb']}MB"
    )
    return result


# ---------------------------------------------------------------------------
# Combine two GPU results into one table
# ---------------------------------------------------------------------------

def combine_results(compute_dir: Path) -> None:
    """Merge A100 and 4060Ti JSONs into a single comparison table."""
    # Exclude the combined output file itself
    files = [f for f in compute_dir.glob("compute_*.json")
             if "combined" not in f.name]
    if len(files) < 2:
        logger.info("Need results from both GPUs to combine. Run on second GPU first.")
        return

    by_gpu     = {}
    all_models = set()
    for f in files:
        gpu_label = f.stem.replace("compute_", "")
        with open(f) as fp:
            results = json.load(fp)
        # Only include successful measurements (skip skipped/error entries)
        by_gpu[gpu_label] = {
            r["model_name"]: r
            for r in results["measurements"]
            if not r.get("skipped") and not r.get("error")
        }
        # Track ALL model names including skipped (to show OOM in combined table)
        for r in results["measurements"]:
            all_models.add(r["model_name"])

    gpu_labels = sorted(by_gpu.keys())
    combined   = []

    for model in sorted(all_models):
        row = {"model_name": model}
        for gpu in gpu_labels:
            data = by_gpu[gpu].get(model)
            if data:
                row[f"{gpu}_latency_ms"]  = data.get("latency_mean_ms", "—")
                row[f"{gpu}_vram_mb"]     = data.get("vram_peak_mb",    "—")
                row[f"{gpu}_throughput"]  = data.get("throughput_img_s","—")
            else:
                row[f"{gpu}_latency_ms"]  = "OOM"
                row[f"{gpu}_vram_mb"]     = "OOM"
                row[f"{gpu}_throughput"]  = "OOM"
        combined.append(row)

    out_path = compute_dir / "compute_combined.json"
    with open(out_path, "w") as f:
        json.dump({"gpu_labels": gpu_labels, "rows": combined}, f, indent=2)

    # Print table
    col_w  = 14
    header = f"{'Model':<30}"
    for gpu in gpu_labels:
        header += f"  {(gpu+' ms/img').center(col_w)}  {(gpu+' VRAM').center(col_w)}"
    print(f"\n{header}")
    print("-" * len(header))
    for row in combined:
        line = f"{row['model_name']:<30}"
        for gpu in gpu_labels:
            line += f"  {str(row.get(f'{gpu}_latency_ms', '—')).center(col_w)}"
            line += f"  {str(row.get(f'{gpu}_vram_mb',    '—')).center(col_w)}"
        print(line)

    logger.info(f"\nCombined table saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Measure inference compute for all models")
    parser.add_argument(
        "--gpu-label", type=str, required=True,
        help="Label for this GPU, e.g. 'A100' or '4060Ti'. Used in output filename.",
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help="Skip InternVL2 and MiniCPM (>16GB VRAM). Use on consumer GPUs.",
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip ResNet-50, ViT-B/16, YOLO (e.g. if checkpoints not ready yet).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Measure specific models only. Default: all.",
    )
    parser.add_argument(
        "--n-measure", type=int, default=N_MEASURE,
        help=f"Number of images to measure (default: {N_MEASURE}).",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Force one strategy for all models. Default: per-model "
             "JOINT_STRATEGIES (same prompts as in the paper's tables).",
    )
    parser.add_argument(
    "--mode", choices=["single", "multi", "both"], default="single",
    help="Inference mode. 'both' measures single- and multi-task in one "
            "model load; multi is skipped for models without multitask support.",
    )
    parser.add_argument("--data-dir",   type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device",     type=str, default="cuda")
    return parser.parse_args()


def setup_logging(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"compute_{ts}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    args       = parse_args()
    output_dir = ROOT / args.output_dir
    data_dir   = ROOT / args.data_dir
    setup_logging(output_dir)

    compute_dir = output_dir / "compute"
    compute_dir.mkdir(parents=True, exist_ok=True)

    gpu_info = ""
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
    logger.info(f"GPU: {gpu_info} (label: {args.gpu_label})")
    logger.info(f"Measuring {args.n_measure} images per model after {N_WARMUP}-image warmup")

    # Load dataset — sample N_MEASURE + N_WARMUP images uniformly
    dataset  = GarmentDataset(data_dir, split="all")
    rng      = np.random.default_rng(42)
    n_needed = args.n_measure + N_WARMUP
    indices  = rng.choice(len(dataset), size=min(n_needed, len(dataset)), replace=False)

    images = []
    for idx in indices:
        item = dataset[int(idx)]
        images.append(item["image"])   # PIL Image

    logger.info(f"Loaded {len(images)} images for measurement")

    measurements = []

    # --- VLM models ---
    vlm_names = args.models if args.models else list(MODEL_REGISTRY.keys())

    for model_name in vlm_names:
        if model_name not in MODEL_REGISTRY:
            continue
        if args.skip_large and model_name in LARGE_MODELS:
            logger.info(f"Skipping {model_name} (--skip-large)")
            measurements.append({
                "model_name": model_name,
                "skipped": True,
                "reason": "OOM on consumer GPU (>16GB VRAM required)",
            })
            continue

        logger.info(f"\n[{model_name}]")
        modes = ("single", "multi") if args.mode == "both" else (args.mode,)
        try:
            results = measure_vlm(model_name, images, args.device,
                                  n_measure=args.n_measure,
                                  strategy=args.strategy,
                                  modes=modes)
            measurements.extend(results)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"  OOM — {model_name} does not fit on this GPU")
            measurements.append({"model_name": model_name, "skipped": True,
                                 "reason": "OOM"})
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            measurements.append({"model_name": model_name, "error": str(e)})

    # --- Supervised baselines ---
    if not args.skip_baselines:
        resnet_ckpt = ROOT / BASELINE_REGISTRY["resnet50"]
        vit_ckpt    = ROOT / BASELINE_REGISTRY["vit_b16"]
        yolo_ckpt   = ROOT / BASELINE_REGISTRY["yolo11n"]

        for tag, ckpt in [("resnet50", resnet_ckpt), ("vit_b16", vit_ckpt)]:
            if ckpt.exists():
                logger.info(f"\n[{tag}]")
                try:
                    result = measure_baseline_cls(tag, ckpt, images, args.device)
                    measurements.append(result)
                except Exception as e:
                    logger.error(f"  FAILED: {e}", exc_info=True)
            else:
                logger.warning(f"Checkpoint not found, skipping {tag}: {ckpt}")

        if yolo_ckpt.exists():
            logger.info("\n[yolo11n_det]")
            try:
                result = measure_yolo(yolo_ckpt, images, args.device)
                measurements.append(result)
            except Exception as e:
                logger.error(f"  FAILED: {e}", exc_info=True)
        else:
            logger.warning(f"YOLO checkpoint not found, skipping: {yolo_ckpt}")

    # --- Save results ---
    out = {
        "gpu_label":    args.gpu_label,
        "gpu_name":     gpu_info,
        "n_measure":    args.n_measure,
        "n_warmup":     N_WARMUP,
        "generated_at": datetime.now().isoformat(),
        "measurements": measurements,
    }
    out_path = compute_dir / f"compute_{args.gpu_label}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nResults saved -> {out_path}")

    # Print summary
    print(f"\n{'Model':<35} {'ms/img':>8} {'VRAM MB':>10} {'img/s':>8}")
    print("-" * 65)
    for m in measurements:
        if m.get("skipped") or m.get("error"):
            status = m.get("reason", m.get("error", "skip"))
            print(f"{m['model_name']:<35} {'—':>8} {status[:20]:>10}")
        else:
            print(
                f"{m['model_name']:<35} "
                f"{m['latency_mean_ms']:>8.1f} "
                f"{m['vram_peak_mb']:>10.0f} "
                f"{m['throughput_img_s']:>8.2f}"
            )

    # Try to combine if both GPU results exist
    combine_results(compute_dir)


if __name__ == "__main__":
    main()
