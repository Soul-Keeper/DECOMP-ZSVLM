import torch
import logging
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from src.base_model import (
    BaseGarmentModel,
    PredictionResult,
    PromptConfig,
    Task,
)

logger = logging.getLogger(__name__)

MODEL_ID = "google/paligemma-3b-mix-448"

# PaliGemma — мультимодальная модель Google на базе SigLIP + Gemma-2B.
# Особенности важные для benchmark:
#
# 1. Detection через префикс "detect <label>" — модель возвращает
#    специальные <loc####> токены в формате y_min x_min y_max x_max,
#    нормализованные к диапазону 0..1024. _parse_bbox в базовом классе
#    уже умеет их обрабатывать.
#
# 2. Гейтированная модель на HuggingFace — нужен токен:
#    huggingface-cli login
#
# 3. Процессор принимает изображение и текст вместе, токены изображения
#    подставляются автоматически — не нужен явный <image> тег в промпте.
#
# 4. При генерации нужно обрезать входные токены из ответа — модель
#    возвращает prompt + ответ, а нам нужен только ответ.


class PaliGemmaModel(BaseGarmentModel):

    MODEL_NAME = "PALIGEMMA"
    SUPPORTS_MULTITASK = False

    # ------------------------------------------------

    def load(self) -> None:

        logger.info(f"[{self.MODEL_NAME}] Loading {MODEL_ID}")

        self._processor = AutoProcessor.from_pretrained(MODEL_ID)

        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )

        self._model.eval()
        self._loaded = True

        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(f"[{self.MODEL_NAME}] Loaded — {n_params:.0f}M params")

    # ------------------------------------------------

    def _generate(self, image: Image.Image, prompt: str) -> str:
        """
        Запускает инференс и возвращает только сгенерированный ответ
        (без входного промпта).
        """
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        # Обрезаем входные токены — оставляем только сгенерированные
        generated_tokens = outputs[0][input_len:]
        decoded = self._processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return decoded.strip()

    # ------------------------------------------------
    # TYPE
    # ------------------------------------------------

    def predict_type(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
    ) -> PredictionResult:

        self._require_loaded()

        def run():
            raw = self._generate(image, prompt_config.type_prompt)
            parsed = self._parse_classification(raw, self.VALID_TYPES)
            return raw, parsed

        try:
            (raw, parsed), latency = self._timed(run)

            return PredictionResult(
                task=Task.TYPE_CLASSIFICATION,
                raw_output=raw,
                parsed_label=parsed,
                is_valid=(parsed is not None),
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_type failed: {e}")
            return PredictionResult(
                task=Task.TYPE_CLASSIFICATION,
                raw_output="",
                is_valid=False,
                error=str(e),
            )

    # ------------------------------------------------
    # COLOR
    # ------------------------------------------------

    def predict_color(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
    ) -> PredictionResult:

        self._require_loaded()

        def run():
            raw = self._generate(image, prompt_config.color_prompt)
            parsed = self._parse_classification(raw, self.VALID_COLORS)
            return raw, parsed

        try:
            (raw, parsed), latency = self._timed(run)

            return PredictionResult(
                task=Task.COLOR_CLASSIFICATION,
                raw_output=raw,
                parsed_label=parsed,
                is_valid=(parsed is not None),
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_color failed: {e}")
            return PredictionResult(
                task=Task.COLOR_CLASSIFICATION,
                raw_output="",
                is_valid=False,
                error=str(e),
            )

    # ------------------------------------------------
    # DETECTION
    # ------------------------------------------------

    def predict_bbox(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
        image_size: tuple[int, int] = (1280, 720),
    ) -> PredictionResult:
        """
        PaliGemma detection через префикс "detect <label>".

        Модель возвращает <loc####> токены в формате:
            <loc0045><loc0123><loc0567><loc0789> tshirt

        Порядок: y_min x_min y_max x_max, нормализованные к 0..1024.
        _parse_bbox в базовом классе обрабатывает этот формат.
        """

        self._require_loaded()

        def run():
            raw = self._generate(image, prompt_config.detection_prompt)
            bbox = self._parse_bbox(raw, image_size)
            return raw, bbox

        try:
            (raw, bbox), latency = self._timed(run)

            return PredictionResult(
                task=Task.DETECTION,
                raw_output=raw,
                bbox_xyxy=bbox,
                is_valid=(bbox is not None),
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_bbox failed: {e}")
            return PredictionResult(
                task=Task.DETECTION,
                raw_output="",
                is_valid=False,
                error=str(e),
            )
