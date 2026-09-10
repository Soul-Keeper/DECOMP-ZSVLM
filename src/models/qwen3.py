import json
import logging
import re

import torch
from PIL import Image
from transformers import AutoProcessor

from src.base_model import BaseGarmentModel, PredictionResult, PromptConfig, Task

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# Qwen3-VL emits box coordinates in a different convention from its
# predecessor, and reading them with the wrong one produces well-formed but
# wrong boxes rather than a parsing failure. The convention was fixed by
# calibration against annotated boxes before the evaluation run.
#   norm_1000   normalized to 0..1000, what Qwen3-VL uses
#   norm_1      normalized to 0..1
#   abs_orig    pixels of the original frame, what Qwen2.5-VL uses
BBOX_CONVENTION = "norm_1000"


def _try_flash_attn() -> bool:
    try:
        import flash_attn  # noqa
        return True
    except ImportError:
        return False


class Qwen3VLModel(BaseGarmentModel):
    """Qwen3-VL-8B. Native JSON boxes; multi-task is a single pass."""

    MODEL_NAME = "QWEN3-VL-8B"
    SUPPORTS_MULTITASK = True

    def load(self) -> None:
        use_flash = _try_flash_attn()
        logger.info(f"[{self.MODEL_NAME}] loading {MODEL_ID} (flash_attn={use_flash})")

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is unavailable; "
                "install transformers>=4.57"
            )

        self._model = ModelCls.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash else "eager",
            device_map=self.device,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] loaded ({self._model.num_parameters():,} params, "
                    f"bbox={BBOX_CONVENTION})")

    def _infer(self, image: Image.Image, question: str,
               max_new_tokens: int) -> tuple[str, float]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        # process_vision_info is deliberately not used: its default patch_size
        # is meant for Qwen2.5-VL.
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
        ).to(self.device)

        def run():
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            return self._processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        return self._timed(run)

    @staticmethod
    def _extract_numbers(raw: str) -> list[float] | None:
        """Four coordinates: the bbox_2d field first, any four numbers second."""
        match = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', raw)
        if match:
            found = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
            if len(found) >= 4:
                return [float(x) for x in found[:4]]

        try:
            match = re.search(r"\{[^{}]*bbox_2d[^{}]*\}", raw)
            if match:
                coords = json.loads(match.group()).get("bbox_2d")
                if coords and len(coords) >= 4:
                    return [float(v) for v in coords[:4]]
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

        found = re.findall(r"-?\d+(?:\.\d+)?", raw)
        return [float(x) for x in found[:4]] if len(found) >= 4 else None

    def _parse_bbox_qwen3(self, raw: str, image: Image.Image) -> list[int] | None:
        numbers = self._extract_numbers(raw)
        if numbers is None:
            return None

        width, height = image.size
        x1, y1, x2, y2 = numbers

        if BBOX_CONVENTION == "norm_1000":
            box = [x1 / 1000 * width, y1 / 1000 * height,
                   x2 / 1000 * width, y2 / 1000 * height]
        elif BBOX_CONVENTION == "norm_1":
            box = [x1 * width, y1 * height, x2 * width, y2 * height]
        else:
            box = [x1, y1, x2, y2]

        box = [int(v) for v in box]
        if box[0] >= box[2] or box[1] >= box[3]:
            return None
        if box[2] > width * 1.1 or box[3] > height * 1.1:
            return None
        return box

    def _run(self, image, prompt, task, max_new_tokens, finish) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._infer(image, prompt, max_new_tokens)
            return finish(raw, latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] {task.value} failed: {e}")
            return PredictionResult(task=task, raw_output="", is_valid=False, error=str(e))

    def _classify(self, image, prompt, task, labels) -> PredictionResult:
        def finish(raw, latency):
            parsed = self._parse_classification(raw, labels)
            return PredictionResult(
                task=task, raw_output=raw, parsed_label=parsed,
                is_valid=parsed is not None, latency_ms=latency,
            )

        return self._run(image, prompt, task, 32, finish)

    def predict_type(self, image: Image.Image,
                     prompt_config: PromptConfig) -> PredictionResult:
        return self._classify(image, prompt_config.type_prompt,
                              Task.TYPE_CLASSIFICATION, self.VALID_TYPES)

    def predict_color(self, image: Image.Image,
                      prompt_config: PromptConfig) -> PredictionResult:
        return self._classify(image, prompt_config.color_prompt,
                              Task.COLOR_CLASSIFICATION, self.VALID_COLORS)

    def predict_bbox(self, image: Image.Image, prompt_config: PromptConfig,
                     image_size: tuple[int, int] = (1280, 720)) -> PredictionResult:
        def finish(raw, latency):
            bbox = self._parse_bbox_qwen3(raw, image) or self._parse_bbox(raw, image_size)
            return PredictionResult(
                task=Task.DETECTION, raw_output=raw, bbox_xyxy=bbox,
                is_valid=bbox is not None, latency_ms=latency,
            )

        return self._run(image, prompt_config.detection_prompt, Task.DETECTION, 64, finish)

    def predict_multitask(self, image: Image.Image, prompt_config: PromptConfig,
                          image_size: tuple[int, int] = (1280, 720)) -> PredictionResult:
        """
        One forward pass for all three tasks. The prompt asks for a structured
        response; raw_output keeps it whole and run_benchmark re-parses it.
        """
        if not prompt_config.multitask_prompt:
            raise ValueError(f"[{self.MODEL_NAME}] multitask_prompt is not set")

        def finish(raw, latency):
            parsed_type = parsed_color = bbox = None
            for line in raw.splitlines():
                head = line.strip().lower()
                value = line.split(":", 1)[1].strip() if ":" in line else ""
                if head.startswith("type:"):
                    parsed_type = self._parse_classification(value, self.VALID_TYPES)
                elif head.startswith("color:"):
                    parsed_color = self._parse_classification(value, self.VALID_COLORS)
                elif head.startswith("box:"):
                    bbox = (self._parse_bbox_qwen3(value, image)
                            or self._parse_bbox(value, image_size))
            return PredictionResult(
                task=Task.MULTITASK, raw_output=raw,
                parsed_type=parsed_type, parsed_color=parsed_color, bbox_xyxy=bbox,
                is_valid=parsed_type is not None and parsed_color is not None,
                latency_ms=latency,
            )

        return self._run(image, prompt_config.multitask_prompt, Task.MULTITASK, 256, finish)
