"""
MiniCPM-Llama3-V-2.5 wrapper.

Поддерживает два режима:
  - Single-task : три отдельных вызова _chat() (E1 основной бенчмарк, E2 ablation)
  - Multi-task  : один вызов predict_multitask() с multitask_prompt (E5)

Оптимизации по сравнению с начальной версией:
  - max_new_tokens=128 для classification (было неограничено)
  - torch.inference_mode() явно оборачивает инференс
  - low_cpu_mem_usage=True при загрузке
"""

import torch
import logging
from PIL import Image

from transformers import AutoModel, AutoTokenizer

from src.base_model import BaseGarmentModel, PredictionResult, PromptConfig, Task

logger = logging.getLogger(__name__)

MODEL_ID = "openbmb/MiniCPM-Llama3-V-2_5"
REVISION = "fd7f352fac0e06d0d818b23f98e3ec8c64267a57"


class MiniCPMModel(BaseGarmentModel):
    MODEL_NAME = "MINICPM-LLAMA3-V-2.5"
    SUPPORTS_MULTITASK = True           # genuine single-pass via multitask_prompt

    def load(self):
        logger.info(f"[{self.MODEL_NAME}] Loading {MODEL_ID}")
        self._model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval().to(self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] Loaded ({self._model.num_parameters():,} params)")

    # ==================== CORE INFERENCE ====================

    def _chat(self, image: Image.Image, question: str,
              max_new_tokens: int = 128) -> tuple[str, float]:
        """Single question → single answer."""
        msgs = [{'role': 'user', 'content': question}]

        def run():
            with torch.inference_mode():
                response = self._model.chat(
                    image=image,
                    msgs=msgs,
                    tokenizer=self._tokenizer,
                    sampling=False,
                    max_new_tokens=max_new_tokens,
                )
            return response.strip()

        return self._timed(run)

    # ==================== SINGLE-TASK (E1, E2) ====================

    def predict_type(self, image: Image.Image, prompt_config: PromptConfig) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._chat(image, prompt_config.type_prompt, max_new_tokens=16)
            parsed = self._parse_classification(raw, self.VALID_TYPES)
            return PredictionResult(task=Task.TYPE_CLASSIFICATION, raw_output=raw,
                                    parsed_label=parsed, is_valid=(parsed is not None),
                                    latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_type failed: {e}")
            return PredictionResult(task=Task.TYPE_CLASSIFICATION, raw_output="",
                                    is_valid=False, error=str(e))

    def predict_color(self, image: Image.Image, prompt_config: PromptConfig) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._chat(image, prompt_config.color_prompt, max_new_tokens=32)
            parsed = self._parse_classification(raw, self.VALID_COLORS)
            return PredictionResult(task=Task.COLOR_CLASSIFICATION, raw_output=raw,
                                    parsed_label=parsed, is_valid=(parsed is not None),
                                    latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_color failed: {e}")
            return PredictionResult(task=Task.COLOR_CLASSIFICATION, raw_output="",
                                    is_valid=False, error=str(e))

    def predict_bbox(self, image: Image.Image, prompt_config: PromptConfig,
                     image_size=(1280, 720)) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._chat(image, prompt_config.detection_prompt, max_new_tokens=256)
            bbox_xyxy = self._parse_bbox(raw, image_size)
            return PredictionResult(task=Task.DETECTION, raw_output=raw,
                                    bbox_xyxy=bbox_xyxy, is_valid=(bbox_xyxy is not None),
                                    latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_bbox failed: {e}")
            return PredictionResult(task=Task.DETECTION, raw_output="",
                                    is_valid=False, error=str(e))

    # ==================== MULTI-TASK (E5) ====================

    def predict_multitask(self, image: Image.Image, prompt_config: PromptConfig,
                          image_size: tuple[int, int] = (1280, 720)) -> PredictionResult:
        """
        Genuine single-pass multi-task inference.

        One model.chat() call with multitask_prompt — image is encoded once.
        The prompt instructs the model to respond in structured format:
            Type: <type>
            Color: <color>
            Box: [x_min, y_min, x_max, y_max]

        This is ~3x faster than three separate predict_* calls and is a
        distinct experimental condition (section 8.5 of the paper).
        """
        self._require_loaded()
        if not prompt_config.multitask_prompt:
            raise ValueError(f"[{self.MODEL_NAME}] multitask_prompt is not set in PromptConfig")

        try:
            raw, latency = self._chat(
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
                    bbox_xyxy = self._parse_bbox(
                        line.split(":", 1)[1].strip(), image_size)

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
        if hasattr(self, '_tokenizer'):
            del self._tokenizer
