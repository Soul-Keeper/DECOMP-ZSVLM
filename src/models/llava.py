import torch
import logging
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

from src.base_model import (
    BaseGarmentModel,
    PredictionResult,
    PromptConfig,
    Task,
)

logger = logging.getLogger(__name__)

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

# LLaVA-1.6 (LLaVA-NeXT) — instruction-tuned MLLM на базе Mistral-7B.
# Особенности важные для benchmark:
#
# 1. Промпт должен быть в формате chat template Mistral:
#    "[INST] <image>\n{вопрос} [/INST]"
#    Без этого модель игнорирует изображение или отвечает некорректно.
#
# 2. LlavaNextProcessor автоматически разбивает изображение на тайлы
#    (до 6 патчей) для высокого разрешения — это особенность v1.6.
#    Важно: передавать оригинальное изображение без ресайза.
#
# 3. Как и PaliGemma, возвращает prompt + ответ — нужно обрезать
#    входные токены перед декодированием.
#
# 4. pad_token нужно явно установить — у Mistral его нет по умолчанию,
#    без этого processor падает при batch обработке.


class LLaVAModel(BaseGarmentModel):

    MODEL_NAME = "LLAVA-1.6-MISTRAL-7B"
    SUPPORTS_MULTITASK = False

    # ------------------------------------------------

    def load(self) -> None:

        logger.info(f"[{self.MODEL_NAME}] Loading {MODEL_ID}")

        self._processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Mistral не имеет pad_token — используем eos_token
        if self._processor.tokenizer.pad_token is None:
            self._processor.tokenizer.pad_token = (
                self._processor.tokenizer.eos_token
            )

        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )

        self._model.eval()
        self._loaded = True

        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(f"[{self.MODEL_NAME}] Loaded — {n_params:.0f}M params")

    # ------------------------------------------------

    @staticmethod
    def _apply_chat_template(prompt: str) -> str:
        """
        Оборачивает промпт в Mistral chat template.
        <image> должен быть внутри [INST]...[/INST].
        Если промпт уже содержит <image> — используем как есть,
        иначе добавляем в начало.
        """
        if "<image>" not in prompt:
            prompt = f"<image>\n{prompt}"
        return f"[INST] {prompt} [/INST]"

    def _generate(self, image: Image.Image, prompt: str) -> str:
        """
        Запускает инференс и возвращает только сгенерированный ответ.
        """
        formatted = self._apply_chat_template(prompt)

        inputs = self._processor(
            text=formatted,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.eos_token_id,
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
