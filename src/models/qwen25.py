import json
import logging
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.base_model import BaseGarmentModel, PredictionResult, PromptConfig, Task

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Dynamic resolution bounds. The processor resizes the frame to fit between
# them, with both sides a multiple of 28.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
PATCH = 28


def _try_flash_attn() -> bool:
    try:
        import flash_attn  # noqa
        return True
    except ImportError:
        return False


class QwenVLModel(BaseGarmentModel):
    """
    Qwen2.5-VL-7B. Boxes come back as a native JSON field, in pixels of the
    resized frame rather than the original, so they are scaled back here.
    Multi-task is a single pass.
    """

    MODEL_NAME = "QWEN2.5-VL-7B"
    SUPPORTS_MULTITASK = True

    def load(self) -> None:
        use_flash = _try_flash_attn()
        logger.info(f"[{self.MODEL_NAME}] loading {MODEL_ID} (flash_attn={use_flash})")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash else "eager",
            device_map=self.device,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] loaded ({self._model.num_parameters():,} params)")

    def _infer(self, image: Image.Image, question: str,
               max_new_tokens: int) -> tuple[str, float]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("qwen-vl-utils is required: pip install qwen-vl-utils")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
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
    def _processed_size(image: Image.Image) -> tuple[int, int]:
        """The size the processor resizes to: within the pixel bounds, sides multiple of 28."""
        width, height = image.size
        total = width * height
        if total < MIN_PIXELS:
            scale = (MIN_PIXELS / total) ** 0.5
        elif total > MAX_PIXELS:
            scale = (MAX_PIXELS / total) ** 0.5
        else:
            scale = 1.0
        return (max(PATCH, round(width * scale / PATCH) * PATCH),
                max(PATCH, round(height * scale / PATCH) * PATCH))

    @staticmethod
    def _extract_coords(raw: str) -> list[float] | None:
        """The bbox_2d JSON field first, any four numbers second."""
        try:
            match = re.search(r"\{[^}]*bbox_2d[^}]*\}", raw)
            if match:
                coords = json.loads(match.group()).get("bbox_2d")
                if coords and len(coords) >= 4:
                    return [float(v) for v in coords[:4]]
        except (json.JSONDecodeError, AttributeError):
            pass

        found = re.findall(r"-?\d+(?:\.\d+)?", raw)
        return [float(n) for n in found[:4]] if len(found) >= 4 else None

    def _parse_bbox_qwen(self, raw: str, image: Image.Image,
                         image_size: tuple[int, int]) -> list[int] | None:
        coords = self._extract_coords(raw)
        if coords is None:
            return None

        width, height = image_size
        proc_width, proc_height = self._processed_size(image)
        scale_x = width / proc_width if proc_width > 0 else 1.0
        scale_y = height / proc_height if proc_height > 0 else 1.0

        x1, y1, x2, y2 = coords
        box = [int(x1 * scale_x), int(y1 * scale_y),
               int(x2 * scale_x), int(y2 * scale_y)]

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
            bbox = (self._parse_bbox_qwen(raw, image, image_size)
                    or self._parse_bbox(raw, image_size))
            return PredictionResult(
                task=Task.DETECTION, raw_output=raw, bbox_xyxy=bbox,
                is_valid=bbox is not None, latency_ms=latency,
            )

        return self._run(image, prompt_config.detection_prompt, Task.DETECTION, 64, finish)

    def predict_multitask(self, image: Image.Image, prompt_config: PromptConfig,
                          image_size: tuple[int, int] = (1280, 720)) -> PredictionResult:
        """
        One forward pass for all three tasks. The prompt asks for:

            Type: tshirt
            Color: black
            Box: {"bbox_2d": [x1, y1, x2, y2]}

        raw_output keeps the full response; run_benchmark re-parses it.
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
                    bbox = (self._parse_bbox_qwen(value, image, image_size)
                            or self._parse_bbox(value, image_size))
            return PredictionResult(
                task=Task.MULTITASK, raw_output=raw,
                parsed_type=parsed_type, parsed_color=parsed_color, bbox_xyxy=bbox,
                is_valid=parsed_type is not None and parsed_color is not None,
                latency_ms=latency,
            )

        return self._run(image, prompt_config.multitask_prompt, Task.MULTITASK, 256, finish)
