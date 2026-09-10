"""
Qwen2.5-VL-7B-Instruct wrapper.

Key differences from other models in this benchmark:

1. NATIVE BBOX OUTPUT — модель обучена выдавать структурированные координаты
   в формате {"bbox_2d": [x1, y1, x2, y2]} без хаков с парсингом текста.
   Координаты привязаны к размеру после dynamic resolution ресайза,
   поэтому нужно пересчитывать в оригинальные пиксели.

2. DYNAMIC RESOLUTION — изображение ресайзится под ближайший размер кратный 28px.
   min_pixels=256*28*28, max_pixels=1280*28*28 — оптимальный диапазон для 720p.

3. PIPELINE — использует AutoProcessor с apply_chat_template + process_vision_info,
   не model.chat() как InternVL/MiniCPM.

4. ЗАВИСИМОСТИ — требует transformers >= 4.49.0 и qwen-vl-utils:
   pip install "transformers>=4.49.0" qwen-vl-utils

Поддерживаемые режимы:
  - Single-task : три отдельных вызова _infer() (E1, E2)
  - Multi-task  : один вызов predict_multitask() с multitask_prompt (E5)
"""

import re
import json
import torch
import logging
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from src.base_model import BaseGarmentModel, PredictionResult, PromptConfig, Task

logger = logging.getLogger(__name__)

MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"
MIN_PIXELS = 256 * 28 * 28    # ~200K pixels — минимум для чёткого распознавания
MAX_PIXELS = 1280 * 28 * 28   # ~1M pixels  — оптимально для 1280x720


def _try_flash_attn() -> bool:
    try:
        import flash_attn  # noqa
        return True
    except ImportError:
        return False


class QwenVLModel(BaseGarmentModel):
    MODEL_NAME = "QWEN2.5-VL-7B"
    SUPPORTS_MULTITASK = True

    def load(self):
        use_flash = _try_flash_attn()
        logger.info(f"[{self.MODEL_NAME}] Loading {MODEL_ID} (flash_attn={use_flash})")

        attn_impl = "flash_attention_2" if use_flash else "eager"

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=self.device,
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] Loaded ({self._model.num_parameters():,} params)")

    # ==================== CORE INFERENCE ====================

    def _infer(self, image: Image.Image, question: str,
               max_new_tokens: int = 64) -> tuple[str, float]:
        """
        Single image + question → text response.

        Uses the standard Qwen2.5-VL pipeline:
          apply_chat_template → process_vision_info → generate → decode
        """
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "qwen-vl-utils is required for Qwen2.5-VL. "
                "Install with: pip install qwen-vl-utils"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        def run():
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            # Trim input tokens — keep only newly generated part
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            return self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        return self._timed(run)

    # ==================== BBOX PARSING ====================

    def _parse_bbox_qwen(self, raw: str, image: Image.Image,
                         image_size: tuple[int, int]) -> list[int] | None:
        """
        Parse Qwen2.5-VL native bbox output.

        The model outputs coordinates relative to the dynamically resized image,
        not the original. We need to scale back to original pixel coordinates.

        Qwen bbox formats (in order of preference):
          1. JSON: {"bbox_2d": [x1, y1, x2, y2]}
          2. Plain list: [x1, y1, x2, y2]
          3. Fallback to base class _parse_bbox
        """
        orig_w, orig_h = image_size

        # Compute the actual processed image size (same logic as AutoProcessor)
        # Image is resized so total pixels are within [MIN_PIXELS, MAX_PIXELS]
        # and dimensions are multiples of 28
        proc_w, proc_h = self._get_processed_size(image)

        # Scale factors from processed → original
        scale_x = orig_w / proc_w if proc_w > 0 else 1.0
        scale_y = orig_h / proc_h if proc_h > 0 else 1.0

        coords = None

        # Try JSON format {"bbox_2d": [...]}
        try:
            # Handle both with and without outer braces
            json_match = re.search(r'\{[^}]*bbox_2d[^}]*\}', raw)
            if json_match:
                data   = json.loads(json_match.group())
                coords = data.get("bbox_2d")
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try plain list [x1, y1, x2, y2]
        if coords is None:
            nums = re.findall(r'-?\d+(?:\.\d+)?', raw)
            if len(nums) >= 4:
                coords = [float(n) for n in nums[:4]]

        if coords is None or len(coords) < 4:
            return None

        x1, y1, x2, y2 = coords

        # Scale to original image coordinates
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Sanity check
        if x1 >= x2 or y1 >= y2:
            return None
        if x2 > orig_w * 1.1 or y2 > orig_h * 1.1:
            return None

        return [x1, y1, x2, y2]

    def _get_processed_size(self, image: Image.Image) -> tuple[int, int]:
        """
        Estimate the size Qwen processor will resize the image to.
        Processor resizes to fit within [MIN_PIXELS, MAX_PIXELS],
        rounding dimensions to nearest multiple of 28.
        """
        orig_w, orig_h = image.size
        total_pixels   = orig_w * orig_h

        if total_pixels < MIN_PIXELS:
            scale = (MIN_PIXELS / total_pixels) ** 0.5
        elif total_pixels > MAX_PIXELS:
            scale = (MAX_PIXELS / total_pixels) ** 0.5
        else:
            scale = 1.0

        proc_w = max(28, round(orig_w * scale / 28) * 28)
        proc_h = max(28, round(orig_h * scale / 28) * 28)
        return proc_w, proc_h

    # ==================== SINGLE-TASK (E1, E2) ====================

    def predict_type(self, image: Image.Image, prompt_config: PromptConfig) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._infer(image, prompt_config.type_prompt, max_new_tokens=32)
            parsed = self._parse_classification(raw, self.VALID_TYPES)
            return PredictionResult(
                task=Task.TYPE_CLASSIFICATION, raw_output=raw,
                parsed_label=parsed, is_valid=(parsed is not None), latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_type failed: {e}")
            return PredictionResult(task=Task.TYPE_CLASSIFICATION, raw_output="",
                                    is_valid=False, error=str(e))

    def predict_color(self, image: Image.Image, prompt_config: PromptConfig) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._infer(image, prompt_config.color_prompt, max_new_tokens=32)
            parsed = self._parse_classification(raw, self.VALID_COLORS)
            return PredictionResult(
                task=Task.COLOR_CLASSIFICATION, raw_output=raw,
                parsed_label=parsed, is_valid=(parsed is not None), latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_color failed: {e}")
            return PredictionResult(task=Task.COLOR_CLASSIFICATION, raw_output="",
                                    is_valid=False, error=str(e))

    def predict_bbox(self, image: Image.Image, prompt_config: PromptConfig,
                     image_size=(1280, 720)) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._infer(image, prompt_config.detection_prompt, max_new_tokens=64)
            bbox_xyxy = self._parse_bbox_qwen(raw, image, image_size)
            # Fallback to base parser if native parsing failed
            if bbox_xyxy is None:
                bbox_xyxy = self._parse_bbox(raw, image_size)
            return PredictionResult(
                task=Task.DETECTION, raw_output=raw,
                bbox_xyxy=bbox_xyxy, is_valid=(bbox_xyxy is not None), latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_bbox failed: {e}")
            return PredictionResult(task=Task.DETECTION, raw_output="",
                                    is_valid=False, error=str(e))

    # ==================== MULTI-TASK (E5) ====================

    def predict_multitask(self, image: Image.Image, prompt_config: PromptConfig,
                          image_size: tuple[int, int] = (1280, 720)) -> PredictionResult:
        """
        Single-pass multi-task inference using multitask_prompt.
        Expected structured response format:
            Type: tshirt
            Color: black
            Box: {"bbox_2d": [x1, y1, x2, y2]}
        """
        self._require_loaded()
        if not prompt_config.multitask_prompt:
            raise ValueError(f"[{self.MODEL_NAME}] multitask_prompt not set in PromptConfig")

        try:
            raw, latency = self._infer(
                image, prompt_config.multitask_prompt, max_new_tokens=256
            )

            parsed_type  = None
            parsed_color = None
            bbox_xyxy    = None

            for line in raw.splitlines():
                line_low = line.strip().lower()
                if line_low.startswith("type:"):
                    parsed_type = self._parse_classification(
                        line.split(":", 1)[1].strip(), self.VALID_TYPES)
                elif line_low.startswith("color:"):
                    parsed_color = self._parse_classification(
                        line.split(":", 1)[1].strip(), self.VALID_COLORS)
                elif line_low.startswith("box:"):
                    box_str   = line.split(":", 1)[1].strip()
                    bbox_xyxy = self._parse_bbox_qwen(box_str, image, image_size)
                    if bbox_xyxy is None:
                        bbox_xyxy = self._parse_bbox(box_str, image_size)

            return PredictionResult(
                task=Task.MULTITASK,
                raw_output=raw,
                parsed_type=parsed_type,
                parsed_color=parsed_color,
                bbox_xyxy=bbox_xyxy,
                is_valid=(parsed_type is not None and parsed_color is not None),
                latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_multitask failed: {e}")
            return PredictionResult(task=Task.MULTITASK, raw_output="",
                                    is_valid=False, error=str(e))

    def unload(self):
        super().unload()
