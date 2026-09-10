"""
Latency and peak-memory profiling.

    python -m src.measure_compute --gpu-label A100 --n-measure 300
    python -m src.measure_compute --gpu-label 4070S --skip-large
    python -m src.measure_compute --gpu-label A100 --mode both --models INTERNVL2-8B

Each model is loaded fresh, warmed up, and timed with CUDA synchronization on
images sampled uniformly from the dataset. One measurement covers every task
the model supports, so single-task figures stay comparable with multi-task ones
taken in the same run.

Writes outputs/compute/compute_<label>.json, and a combined table once results
from more than one device are present.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import GarmentDataset
from src.prompts import DEFAULT_STRATEGIES, get_prompt

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
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

# Do not fit on a consumer card.
LARGE_MODELS = {"INTERNVL2-8B", "MINICPM-LLAMA3-V-2.5",
                "QWEN2.5-VL-7B", "QWEN3-VL-8B", "LLAVA-1.6-MISTRAL-7B"}

# Wrappers whose SUPPORTS_MULTITASK is True.
MULTITASK_MODELS = {"INTERNVL2-8B", "MINICPM-LLAMA3-V-2.5",
                    "QWEN2.5-VL-7B", "QWEN3-VL-8B"}

N_WARMUP = 10
N_MEASURE = 300


def measure_vlm(model_name, images, device, n_measure=N_MEASURE,
                strategy=None, modes=("single",)) -> list:
    """
    Load the model once and time the requested modes.

    single: three forward passes, one per task
    multi:  one forward pass resolving all three tasks
    """
    module_path, class_name = MODEL_REGISTRY[model_name]
    ModelCls = getattr(import_module(module_path), class_name)
    model = ModelCls(device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model.load()
    load_time_s = time.perf_counter() - t_load
    vram_after_load_mb = torch.cuda.memory_allocated() / 1024 ** 2

    # Latency depends strongly on the prompt, so the deployable strategy of the
    # model is used, the one reported in the tables.
    chosen = strategy or DEFAULT_STRATEGIES[model_name]
    prompt_config = get_prompt(model_name, chosen)
    logger.info(f"  strategy: {chosen}")

    results = []

    for mode in modes:
        if mode == "multi":
            if model_name not in MULTITASK_MODELS:
                logger.info(f"  skipping multi: {model_name} does not support it")
                continue
            if not getattr(prompt_config, "multitask_prompt", None):
                logger.info(f"  skipping multi: no multitask_prompt for {chosen}")
                continue

        def one_pass(img, mode=mode):
            if mode == "multi":
                model.predict_multitask(img, prompt_config)
            else:
                model.predict_type(img, prompt_config)
                model.predict_color(img, prompt_config)
                model.predict_bbox(img, prompt_config)

        logger.info(f"  [{mode}] warmup on {N_WARMUP} images")
        for img in images[:N_WARMUP]:
            try:
                one_pass(img)
            except Exception:
                pass

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        logger.info(f"  [{mode}] measuring {n_measure} images")
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
                logger.warning(f"  error on image: {e}")

        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

        if not latencies:
            results.append({"model_name": model_name, "mode": mode,
                            "error": "all inference calls failed"})
            continue

        lat = np.array(latencies)
        result = {
            "model_name": model_name,
            "mode": mode,
            "strategy": chosen,
            "n_measured": len(latencies),
            "n_errors": errors,
            "latency_mean_ms": round(float(lat.mean()), 1),
            "latency_median_ms": round(float(np.median(lat)), 1),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
            "latency_std_ms": round(float(lat.std()), 1),
            "throughput_img_s": round(1000 / float(np.median(lat)), 2),
            "vram_load_mb": round(vram_after_load_mb, 0),
            "vram_peak_mb": round(peak_vram_mb, 0),
            "load_time_s": round(load_time_s, 1),
        }
        results.append(result)
        logger.info(f"  {model_name} [{mode}]: {result['latency_median_ms']} ms "
                    f"median, peak {result['vram_peak_mb']} MB")

    model.unload()
    torch.cuda.empty_cache()

    by_mode = {r["mode"]: r for r in results if "latency_median_ms" in r}
    if "single" in by_mode and "multi" in by_mode:
        speedup = by_mode["single"]["latency_median_ms"] / by_mode["multi"]["latency_median_ms"]
        for r in results:
            r["speedup"] = round(speedup, 2)
        logger.info(f"  {model_name}: speedup {speedup:.2f}x")

    return results


def combine_results(compute_dir: Path) -> None:
    """Merge per-device JSONs into one table. Medians, as in the paper."""
    files = [f for f in compute_dir.glob("compute_*.json") if "combined" not in f.name]
    if len(files) < 2:
        logger.info("need results from more than one device to combine")
        return

    by_device, all_models = {}, set()
    for path in files:
        label = path.stem.replace("compute_", "")
        results = json.load(open(path))["measurements"]
        by_device[label] = {r["model_name"]: r for r in results
                            if not r.get("skipped") and not r.get("error")}
        all_models.update(r["model_name"] for r in results)

    labels = sorted(by_device)
    rows = []
    for model in sorted(all_models):
        row = {"model_name": model}
        for label in labels:
            data = by_device[label].get(model)
            row[f"{label}_latency_ms"] = data.get("latency_median_ms", "—") if data else "OOM"
            row[f"{label}_vram_mb"] = data.get("vram_peak_mb", "—") if data else "OOM"
        rows.append(row)

    out_path = compute_dir / "compute_combined.json"
    out_path.write_text(json.dumps({"devices": labels, "rows": rows}, indent=2))

    header = f"{'Model':<30}" + "".join(
        f"  {(d + ' ms').center(12)}  {(d + ' MB').center(12)}" for d in labels)
    print(f"\n{header}\n" + "-" * len(header))
    for row in rows:
        print(f"{row['model_name']:<30}" + "".join(
            f"  {str(row[f'{d}_latency_ms']).center(12)}"
            f"  {str(row[f'{d}_vram_mb']).center(12)}" for d in labels))
    logger.info(f"combined table -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure inference latency and memory")
    parser.add_argument("--gpu-label", required=True,
                        help="device label used in the output filename, e.g. A100")
    parser.add_argument("--skip-large", action="store_true",
                        help=f"skip models that need a datacenter card: "
                             f"{sorted(LARGE_MODELS)}")
    parser.add_argument("--models", nargs="+", default=None,
                        help="measure these models only")
    parser.add_argument("--n-measure", type=int, default=N_MEASURE)
    parser.add_argument("--strategy", default=None,
                        help="force one strategy; default is the model's joint strategy")
    parser.add_argument("--mode", choices=["single", "multi", "both"], default="single")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_dir / f"compute_{timestamp}.log"),
                  logging.StreamHandler(sys.stdout)],
    )


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    setup_logging(output_dir)

    compute_dir = output_dir / "compute"
    compute_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    logger.info(f"GPU: {gpu_name} (label {args.gpu_label})")
    logger.info(f"{args.n_measure} images per model after a {N_WARMUP}-image warmup")

    dataset = GarmentDataset(ROOT / args.data_dir, split="all")
    rng = np.random.default_rng(42)
    n_needed = min(args.n_measure + N_WARMUP, len(dataset))
    indices = rng.choice(len(dataset), size=n_needed, replace=False)
    images = [dataset[int(i)]["image"] for i in indices]
    logger.info(f"loaded {len(images)} images")

    measurements = []
    modes = ("single", "multi") if args.mode == "both" else (args.mode,)

    for model_name in (args.models or MODEL_REGISTRY):
        if model_name not in MODEL_REGISTRY:
            continue
        if args.skip_large and model_name in LARGE_MODELS:
            logger.info(f"skipping {model_name} (--skip-large)")
            measurements.append({"model_name": model_name, "skipped": True,
                                 "reason": "does not fit on a consumer card"})
            continue

        logger.info(f"\n[{model_name}]")
        try:
            measurements.extend(measure_vlm(
                model_name, images, args.device,
                n_measure=args.n_measure, strategy=args.strategy, modes=modes))
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"  out of memory on {model_name}")
            measurements.append({"model_name": model_name, "skipped": True, "reason": "OOM"})
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"  failed: {e}", exc_info=True)
            measurements.append({"model_name": model_name, "error": str(e)})

    out_path = compute_dir / f"compute_{args.gpu_label}.json"
    out_path.write_text(json.dumps({
        "gpu_label": args.gpu_label,
        "gpu_name": gpu_name,
        "n_measure": args.n_measure,
        "n_warmup": N_WARMUP,
        "generated_at": datetime.now().isoformat(),
        "measurements": measurements,
    }, indent=2))
    logger.info(f"\nresults -> {out_path}")

    print(f"\n{'Model':<30}{'median ms':>12}{'VRAM MB':>10}{'img/s':>8}")
    print("-" * 60)
    for m in measurements:
        if m.get("skipped") or m.get("error"):
            status = m.get("reason", m.get("error", "skipped"))
            print(f"{m['model_name']:<30}{status[:28]:>30}")
        else:
            print(f"{m['model_name']:<30}{m['latency_median_ms']:>12.1f}"
                  f"{m['vram_peak_mb']:>10.0f}{m['throughput_img_s']:>8.2f}")

    combine_results(compute_dir)


if __name__ == "__main__":
    main()
