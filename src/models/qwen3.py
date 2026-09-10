"""
Qwen3-VL-8B-Instruct wrapper.

Добавляется в пул как ЧЕТВЁРТОЕ поколение относительно Qwen2.5-VL-7B:
меняется только поколение модели, промпты и протокол те же. Это делает
сравнение контролируемым — разница в числах относится к модели, а не
к формулировкам.

ОТЛИЧИЯ ОТ Qwen2.5-VL, критичные для кода
------------------------------------------
1. КООРДИНАТЫ. В Qwen3-VL 2D-grounding переведён с абсолютных пиксельных
   координат на ОТНОСИТЕЛЬНЫЕ. Парсер от Qwen2.5-VL даст систематически
   смещённые рамки. Конвенция задаётся BBOX_CONVENTION и определяется
   заранее скриптом calibrate_qwen3_bbox.py — не угадывать.

2. TRANSFORMERS >= 4.57. Отдельное окружение: обновление до 4.57 ломает
   InternVL2 и MiniCPM, которые грузятся через trust_remote_code.

3. PATCH_SIZE. В qwen-vl-utils по умолчанию 14 — значение для Qwen2.5-VL.
   С Qwen3-VL это даёт не ошибку, а ТИХО неверные счётчики токенов.
   Поэтому здесь process_vision_info не используется, изображение
   передаётся в processor напрямую.

4. INSTRUCT, НЕ THINKING. У Thinking-версии взрывается латентность,
   и сравнение с существующими замерами станет некорректным.
"""

import re
import json
import torch
import logging
from PIL import Image

from transformers import AutoProcessor

from src.base_model import BaseGarmentModel, PredictionResult, PromptConfig, Task

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# Определяется скриптом calibrate_qwen3_bbox.py. Варианты:
#   "norm_1000"    координаты нормированы в 0..1000   (ожидаемо для Qwen3-VL)
#   "abs_orig"     пиксели исходника                  (как было в Qwen2.5-VL)
#   "norm_1"       нормировано в 0..1
#   "abs_resized"  пиксели после ресайза процессором
BBOX_CONVENTION = "norm_1000"


def _try_flash_attn() -> bool:
    try:
        import flash_attn  # noqa
        return True
    except ImportError:
        return False


class Qwen3VLModel(BaseGarmentModel):
    MODEL_NAME = "QWEN3-VL-8B"
    SUPPORTS_MULTITASK = True

    def load(self):
        use_flash = _try_flash_attn()
        logger.info(f"[{self.MODEL_NAME}] Loading {MODEL_ID} (flash_attn={use_flash})")

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:
            raise ImportError(
                "Qwen3VLForConditionalGeneration недоступен. "
                "Нужен transformers >= 4.57: pip install 'transformers>=4.57'"
            )

        self._model = ModelCls.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash else "eager",
            device_map=self.device,
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] Loaded "
                    f"({self._model.num_parameters():,} params, "
                    f"bbox={BBOX_CONVENTION})")

    # ==================== CORE INFERENCE ====================

    def _infer(self, image: Image.Image, question: str,
               max_new_tokens: int = 64) -> tuple[str, float]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # process_vision_info намеренно не используется: его patch_size
        # по умолчанию рассчитан на Qwen2.5-VL
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
        ).to(self.device)

        def run():
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
            trimmed = [out[len(inp):]
                       for inp, out in zip(inputs.input_ids, generated)]
            return self._processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        return self._timed(run)

    # ==================== BBOX PARSING ====================

    def _processed_size(self, inputs) -> tuple[int, int]:
        """Размер изображения после ресайза процессором, из image_grid_thw."""
        try:
            grid = inputs["image_grid_thw"][0]
            return int(grid[2]) * 14, int(grid[1]) * 14      # (w, h)
        except Exception:
            return 0, 0

    def _parse_bbox_qwen3(self, raw: str, image: Image.Image,
                          proc_wh: tuple[int, int] | None = None
                          ) -> list[int] | None:
        """
        Разобрать рамку с учётом конвенции координат Qwen3-VL.

        Порядок: сначала JSON-поле bbox_2d, затем первые четыре числа.
        Далее пересчёт в пиксели исходника по BBOX_CONVENTION.
        """
        nums = None
        m = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', raw)
        if m:
            found = re.findall(r'-?\d+(?:\.\d+)?', m.group(1))
            if len(found) >= 4:
                nums = [float(x) for x in found[:4]]
        if nums is None:
            # запасной путь: JSON-объект целиком
            try:
                jm = re.search(r'\{[^{}]*bbox_2d[^{}]*\}', raw)
                if jm:
                    coords = json.loads(jm.group()).get("bbox_2d")
                    if coords and len(coords) >= 4:
                        nums = [float(v) for v in coords[:4]]
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        if nums is None:
            found = re.findall(r'-?\d+(?:\.\d+)?', raw)
            if len(found) >= 4:
                nums = [float(x) for x in found[:4]]
        if nums is None:
            return None

        ow, oh = image.size
        x1, y1, x2, y2 = nums

        if BBOX_CONVENTION == "norm_1000":
            box = [x1 / 1000 * ow, y1 / 1000 * oh, x2 / 1000 * ow, y2 / 1000 * oh]
        elif BBOX_CONVENTION == "norm_1":
            box = [x1 * ow, y1 * oh, x2 * ow, y2 * oh]
        elif BBOX_CONVENTION == "abs_resized" and proc_wh and all(proc_wh):
            pw, ph = proc_wh
            box = [x1 * ow / pw, y1 * oh / ph, x2 * ow / pw, y2 * oh / ph]
        else:                                    # abs_orig
            box = [x1, y1, x2, y2]

        box = [int(v) for v in box]
        if box[0] >= box[2] or box[1] >= box[3]:
            return None
        if box[2] > ow * 1.1 or box[3] > oh * 1.1:
            return None
        return box

    # ==================== SINGLE-TASK ====================

    def predict_type(self, image: Image.Image,
                     prompt_config: PromptConfig) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._infer(image, prompt_config.type_prompt,
                                       max_new_tokens=32)
            parsed = self._parse_classification(raw, self.VALID_TYPES)
            return PredictionResult(
                task=Task.TYPE_CLASSIFICATION, raw_output=raw,
                parsed_label=parsed, is_valid=(parsed is not None),
                latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_type failed: {e}")
            return PredictionResult(task=Task.TYPE_CLASSIFICATION,
                                    raw_output="", is_valid=False, error=str(e))

    def predict_color(self, image: Image.Image,
                      prompt_config: PromptConfig) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._infer(image, prompt_config.color_prompt,
                                       max_new_tokens=32)
            parsed = self._parse_classification(raw, self.VALID_COLORS)
            return PredictionResult(
                task=Task.COLOR_CLASSIFICATION, raw_output=raw,
                parsed_label=parsed, is_valid=(parsed is not None),
                latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_color failed: {e}")
            return PredictionResult(task=Task.COLOR_CLASSIFICATION,
                                    raw_output="", is_valid=False, error=str(e))

    def predict_bbox(self, image: Image.Image, prompt_config: PromptConfig,
                     image_size=(1280, 720)) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._infer(image, prompt_config.detection_prompt,
                                       max_new_tokens=64)
            bbox = self._parse_bbox_qwen3(raw, image)
            if bbox is None:
                bbox = self._parse_bbox(raw, image_size)
            return PredictionResult(
                task=Task.DETECTION, raw_output=raw, bbox_xyxy=bbox,
                is_valid=(bbox is not None), latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_bbox failed: {e}")
            return PredictionResult(task=Task.DETECTION, raw_output="",
                                    is_valid=False, error=str(e))

    # ==================== MULTI-TASK ====================

    def predict_multitask(self, image: Image.Image, prompt_config: PromptConfig,
                          image_size: tuple[int, int] = (1280, 720)
                          ) -> PredictionResult:
        self._require_loaded()
        if not prompt_config.multitask_prompt:
            raise ValueError(f"[{self.MODEL_NAME}] multitask_prompt не задан")

        try:
            raw, latency = self._infer(image, prompt_config.multitask_prompt,
                                       max_new_tokens=256)
            parsed_type = parsed_color = bbox = None
            for line in raw.splitlines():
                low = line.strip().lower()
                if low.startswith("type:"):
                    parsed_type = self._parse_classification(
                        line.split(":", 1)[1].strip(), self.VALID_TYPES)
                elif low.startswith("color:"):
                    parsed_color = self._parse_classification(
                        line.split(":", 1)[1].strip(), self.VALID_COLORS)
                elif low.startswith("box:"):
                    part = line.split(":", 1)[1].strip()
                    bbox = self._parse_bbox_qwen3(part, image) \
                        or self._parse_bbox(part, image_size)

            return PredictionResult(
                task=Task.MULTITASK, raw_output=raw,
                parsed_type=parsed_type, parsed_color=parsed_color,
                bbox_xyxy=bbox,
                is_valid=(parsed_type is not None and parsed_color is not None),
                latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_multitask failed: {e}")
            return PredictionResult(task=Task.MULTITASK, raw_output="",
                                    is_valid=False, error=str(e))

    def unload(self):
        super().unload()
