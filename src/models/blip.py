import logging

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

from src.base_model import (
    BaseGarmentModel,
    PredictionResult,
    PromptConfig,
    Task,
)

logger = logging.getLogger(__name__)

MODEL_ID = "ybelkada/blip-vqa-capfilt-large"


class BLIPModel(BaseGarmentModel):
    """BLIP-VQA. Generates free-form text; no spatial output."""

    MODEL_NAME = "BLIP-VQA-CAPFILT-LARGE"
    SUPPORTS_MULTITASK = False

    def load(self) -> None:
        logger.info(f"[{self.MODEL_NAME}] loading {MODEL_ID}")
        self._processor = BlipProcessor.from_pretrained(MODEL_ID)
        self._model = BlipForQuestionAnswering.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        ).to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] loaded "
                    f"({self._model.num_parameters():,} params on {self.device})")

    def _ask(self, image: Image.Image, question: str) -> tuple[str, float]:
        inputs = self._processor(image, question, return_tensors="pt").to(
            self.device, torch.float16)

        def run():
            with torch.no_grad():
                out = self._model.generate(**inputs)
            return self._processor.decode(out[0], skip_special_tokens=True)

        return self._timed(run)

    def _classify(self, image, question, task, labels) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._ask(image, question)
            parsed = self._parse_classification(raw, labels)
            return PredictionResult(
                task=task, raw_output=raw, parsed_label=parsed,
                is_valid=parsed is not None, latency_ms=latency,
            )
        except Exception as e:
            logger.warning(f"[{self.MODEL_NAME}] {task.value} failed: {e}")
            return PredictionResult(task=task, raw_output="", is_valid=False, error=str(e))

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
        return PredictionResult(
            task=Task.DETECTION, raw_output="", bbox_xyxy=None, is_valid=False,
            error="BLIP has no spatial output",
        )
