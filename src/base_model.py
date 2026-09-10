import re
import time
import torch
import logging

from PIL import Image
from enum import Enum
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

 
class Task(str, Enum):
    TYPE_CLASSIFICATION = "type_classification"
    COLOR_CLASSIFICATION = "color_classification"
    DETECTION            = "detection"
    MULTITASK            = "multitask"
 

@dataclass
class PromptConfig:
    """
    Holds prompt strings for each task and a strategy label.
 
    strategy: short identifier used in result filenames and ablation tables,
              e.g. "label_only", "question", "constrained", "descriptive"
    """
    strategy: str
    type_prompt: str
    color_prompt: str
    detection_prompt: str
    multitask_prompt: Optional[str] = None
 
 
@dataclass
class PredictionResult:
    """
    Unified result container returned by every predict_* call.
 
    Fields
    ------
    task          : which task this result belongs to
    raw_output    : unmodified string output from the model
    parsed_label  : cleaned string label after post-processing, or None
    bbox_xyxy     : [x_min, y_min, x_max, y_max] for detection, or None
    is_valid      : False if model output could not be parsed into a label
    error         : exception message if something went wrong, else None
    latency_ms    : wall-clock inference time in milliseconds
    """
    task: Task
    raw_output: str
    parsed_label: Optional[str]       = None
    bbox_xyxy: Optional[list[int]]    = None
    is_valid: bool                    = True
    error: Optional[str]              = None
    latency_ms: float                 = 0.0
 
    # Multitask convenience fields (populated when task == MULTITASK)
    parsed_type: Optional[str]        = None
    parsed_color: Optional[str]       = None
 
 
@dataclass
class BenchmarkRecord:
    """
    One complete record stored to disk per image per model per prompt strategy.
    Designed to be JSON-serialisable (no PIL/tensor objects).
    """
    # Identifiers
    model_name: str
    prompt_strategy: str
    image_id: int
    file_name: str
 
    # Ground truth
    gt_type: str
    gt_color: str
    gt_bbox_xyxy: list[int]
 
    # Predictions
    pred_type: Optional[str]       = None
    pred_color: Optional[str]      = None
    pred_bbox_xyxy: Optional[list[int]] = None
 
    # Raw outputs (for error analysis)
    raw_type: Optional[str]        = None
    raw_color: Optional[str]       = None
    raw_detection: Optional[str]   = None
    raw_multitask: Optional[str]   = None
 
    # Validity flags
    type_valid: bool               = False
    color_valid: bool              = False
    detection_valid: bool          = False
 
    # Latency per task (ms)
    latency_type_ms: float         = 0.0
    latency_color_ms: float        = 0.0
    latency_detection_ms: float    = 0.0
 
 
class BaseGarmentModel(ABC):
    """
    Abstract base class for all VLM wrappers in the garment benchmark.
 
    Subclasses must implement:
        load()           — load model weights onto device
        predict_type()   — garment type classification
        predict_color()  — color classification
        predict_bbox()   — bounding box detection
 
    Subclasses may optionally override:
        predict_multitask() — run all three tasks in one forward pass
        unload()            — free GPU memory after benchmark run
 
    Usage
    -----
        model = ConcreteModel(device="cuda")
        model.load()
 
        result = model.predict_type(image, prompt_config)
        print(result.parsed_label, result.latency_ms)
 
        model.unload()
    """
 
    # Subclasses must set these
    MODEL_NAME: str = "base"
    SUPPORTS_MULTITASK: bool = False
 
    VALID_TYPES  = {"tshirt", "sweatshirt"}
    VALID_COLORS = {"black", "brown", "nude", "white", "pink"}
 
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._loaded = False
        self._model = None
        self._processor = None
 
    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor/tokenizer onto self.device."""
        ...
 
    def unload(self) -> None:
        """
        Free GPU memory. Override if model requires special cleanup.
        Default implementation deletes model/processor and empties cache.
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info(f"[{self.MODEL_NAME}] Unloaded and GPU cache cleared")
 
    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                f"[{self.MODEL_NAME}] Model not loaded. Call .load() first."
            )
 
    @abstractmethod
    def predict_type(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
    ) -> PredictionResult:
        """
        Predict garment type (tshirt | sweatshirt).
 
        Must return PredictionResult with:
          - task = Task.TYPE_CLASSIFICATION
          - raw_output = unmodified model string output
          - parsed_label = one of VALID_TYPES, or None if parsing failed
          - is_valid = True only if parsed_label is in VALID_TYPES
          - latency_ms = wall-clock inference time
        """
        ...
 
    @abstractmethod
    def predict_color(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
    ) -> PredictionResult:
        """
        Predict garment color (black | brown | nude | white | pink).
 
        Same contract as predict_type but for VALID_COLORS.
        """
        ...
 
    @abstractmethod
    def predict_bbox(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
        image_size: tuple[int, int] = (1280, 720),
    ) -> PredictionResult:
        """
        Predict bounding box of the garment.
 
        Must return PredictionResult with:
          - task = Task.DETECTION
          - raw_output = unmodified model string output
          - bbox_xyxy = [x_min, y_min, x_max, y_max] in absolute pixels, or None
          - is_valid = True only if bbox_xyxy was successfully parsed
          - latency_ms = wall-clock inference time
 
        image_size: (width, height) of the original image — needed by models
                    that output normalised or relative coordinates.
        """
        ...
 
    def predict_multitask(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
        image_size: tuple[int, int] = (1280, 720),
    ) -> PredictionResult:
        """
        Run type, color and detection in a single forward pass.
 
        Default fallback: runs the three tasks sequentially.
        Override in models that genuinely support single-pass multitask
        (InternVL2, MiniCPM) and set SUPPORTS_MULTITASK = True.
 
        Returns PredictionResult with task=MULTITASK and populated
        parsed_type, parsed_color, bbox_xyxy fields.
        """
        if self.SUPPORTS_MULTITASK:
            raise NotImplementedError(
                f"[{self.MODEL_NAME}] SUPPORTS_MULTITASK=True but "
                "predict_multitask() is not implemented."
            )
 
        # Sequential fallback
        t0 = time.perf_counter()
        r_type  = self.predict_type(image, prompt_config)
        r_color = self.predict_color(image, prompt_config)
        r_bbox  = self.predict_bbox(image, prompt_config, image_size)
        total_ms = (time.perf_counter() - t0) * 1000
 
        combined_raw = (
            f"[type] {r_type.raw_output} | "
            f"[color] {r_color.raw_output} | "
            f"[bbox] {r_bbox.raw_output}"
        )
 
        return PredictionResult(
            task=Task.MULTITASK,
            raw_output=combined_raw,
            parsed_type=r_type.parsed_label,
            parsed_color=r_color.parsed_label,
            bbox_xyxy=r_bbox.bbox_xyxy,
            is_valid=(r_type.is_valid and r_color.is_valid and r_bbox.is_valid),
            latency_ms=total_ms,
        )
 
    def _parse_classification(
        self,
        raw: str,
        valid_labels: set[str],
    ) -> Optional[str]:
        """
        Extract a valid label from model output string.
 
        Strategy (in order):
          1. Exact match after lowercasing and stripping
          2. Check if any valid label appears as a substring
          3. Return None — caller marks result as invalid
        """
        cleaned = raw.lower().strip().rstrip(".,!?")
 
        # 1. Exact match
        if cleaned in valid_labels:
            return cleaned
 
        # 2. Word-boundary scan — find all matching labels
        found = []
        for label in valid_labels:
            pattern = r'\b' + re.escape(label) + r'\b'
            if re.search(pattern, cleaned):
                found.append(label)
 
        if len(found) == 1:
            return found[0]
 
        # Multiple labels found — ambiguous, reject
        return None
 
    def _parse_bbox(
        self,
        raw: str,
        image_size: tuple[int, int] = (1280, 720),
    ) -> Optional[list[int]]:
        """
        Extract [x_min, y_min, x_max, y_max] in absolute pixel coordinates.
 
        Handles common output formats:
          - "[123, 45, 678, 234]"        plain list
          - "123, 45, 678, 234"          comma-separated
          - "123 45 678 234"             space-separated
          - normalised floats 0..1       scaled by image_size
          - PaliGemma <loc0123> tokens   decoded to 0..1 range then scaled
 
        Returns None if parsing fails.
        """
        import re
 
        raw = raw.strip()
        w, h = image_size
 
        # Try <loc####> token format (PaliGemma / similar)
        loc_tokens = re.findall(r"<loc(\d{4})>", raw)
        if len(loc_tokens) >= 4:
            # PaliGemma encodes y_min, x_min, y_max, x_max normalised to 0..1024
            try:
                vals = [int(t) / 1024.0 for t in loc_tokens[:4]]
                y_min, x_min, y_max, x_max = vals
                return [
                    int(x_min * w), int(y_min * h),
                    int(x_max * w), int(y_max * h),
                ]
            except Exception:
                pass
 
        # Extract numeric tokens (int or float)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", raw)
        if len(numbers) < 4:
            return None
 
        try:
            vals = [float(n) for n in numbers[:4]]
        except ValueError:
            return None
 
        # Normalised floats (0..1 range)
        if all(0.0 <= v <= 1.0 for v in vals):
            x_min, y_min, x_max, y_max = vals
            return [
                int(x_min * w), int(y_min * h),
                int(x_max * w), int(y_max * h),
            ]
 
        # Absolute pixel coordinates
        x_min, y_min, x_max, y_max = [int(v) for v in vals]
 
        # Sanity check
        if x_min >= x_max or y_min >= y_max:
            return None
        if x_max > w * 1.1 or y_max > h * 1.1:   # allow 10% slack
            return None
 
        return [x_min, y_min, x_max, y_max]
 
    @staticmethod
    def _timed(fn, *args, **kwargs) -> tuple:
        """
        Run fn(*args, **kwargs) and return (result, elapsed_ms).
        GPU-aware: syncs CUDA before stopping the clock.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return result, elapsed_ms
 
    def gpu_memory_mb(self) -> Optional[float]:
        """Return current GPU memory allocated in MB, or None if no CUDA."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 2
        return None
 
    def gpu_memory_peak_mb(self) -> Optional[float]:
        """Return peak GPU memory allocated in MB since last reset."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
        return None
 
    def reset_peak_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
 
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"{self.__class__.__name__}(device={self.device}, status={status})"