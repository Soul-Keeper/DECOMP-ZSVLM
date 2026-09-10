import logging

import torch
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor

from src.base_model import (
    BaseGarmentModel,
    PredictionResult,
    PromptConfig,
    Task,
)

logger = logging.getLogger(__name__)

MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"


class ViLTModel(BaseGarmentModel):
    """
    ViLT-B/32 VQA. The head is a classifier over a closed answer vocabulary of
    about 3000 entries, so the model cannot emit a label outside it. No spatial
    output.
    """

    MODEL_NAME = "VILT-B32-FINETUNED-VQA"
    SUPPORTS_MULTITASK = False

    def load(self) -> None:
        logger.info(f"[{self.MODEL_NAME}] loading {MODEL_ID}")
        self._processor = ViltProcessor.from_pretrained(MODEL_ID)
        self._model = ViltForQuestionAnswering.from_pretrained(MODEL_ID).to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] loaded "
                    f"({self._model.num_parameters():,} params on {self.device})")

    def _ask(self, image: Image.Image, question: str) -> tuple[str, float]:
        encoding = self._processor(image, question, return_tensors="pt").to(self.device)

        def run():
            with torch.no_grad():
                logits = self._model(**encoding).logits
            return self._model.config.id2label[logits.argmax(-1).item()]

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
            error="ViLT has no spatial output",
        )
