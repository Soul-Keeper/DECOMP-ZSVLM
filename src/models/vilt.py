import time
import torch
import logging

from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor
 
from src.base_model import (
    BaseGarmentModel,
    PromptConfig,
    PredictionResult,
    Task,
)
 
logger = logging.getLogger(__name__)
 
MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"

class ViLTModel(BaseGarmentModel):
 
    MODEL_NAME = "VILT-B32-FINETUNED-VQA"
    SUPPORTS_MULTITASK = False
 
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
 
    def load(self) -> None:
        logger.info(f"[{self.MODEL_NAME}] Loading from {MODEL_ID} ...")
        self._processor = ViltProcessor.from_pretrained(MODEL_ID)
        self._model = ViltForQuestionAnswering.from_pretrained(MODEL_ID).to(self.device)
        self._model.eval()
        self._loaded = True
        params = self._model.num_parameters()
        logger.info(f"[{self.MODEL_NAME}] Loaded — {params/1e6:.2f}M parameters on {self.device}")
 
    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
 
    def _ask(self, image: Image.Image, question: str) -> tuple[str, float]:
        """
        Run VQA inference and return (raw_answer_string, latency_ms).
 
        ViLT VQA head outputs logits over a fixed vocabulary (~3000 tokens).
        We take argmax and decode via id2label.
        """
        encoding = self._processor(image, question, return_tensors="pt").to(self.device)
 
        def _forward():
            with torch.no_grad():
                outputs = self._model(**encoding)
            idx = outputs.logits.argmax(-1).item()
            return self._model.config.id2label[idx]
 
        raw_answer, latency_ms = self._timed(_forward)
        return raw_answer, latency_ms
 
    # ------------------------------------------------------------------
    # predict_type
    # ------------------------------------------------------------------
 
    def predict_type(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
    ) -> PredictionResult:
        self._require_loaded()
 
        try:
            raw, latency_ms = self._ask(image, prompt_config.type_prompt)
            parsed = self._parse_classification(raw, self.VALID_TYPES)
            return PredictionResult(
                task=Task.TYPE_CLASSIFICATION,
                raw_output=raw,
                parsed_label=parsed,
                is_valid=parsed is not None,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.warning(f"[{self.MODEL_NAME}] predict_type failed: {e}")
            return PredictionResult(
                task=Task.TYPE_CLASSIFICATION,
                raw_output="",
                is_valid=False,
                error=str(e),
            )
 
    # ------------------------------------------------------------------
    # predict_color
    # ------------------------------------------------------------------
 
    def predict_color(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
    ) -> PredictionResult:
        self._require_loaded()
 
        try:
            raw, latency_ms = self._ask(image, prompt_config.color_prompt)
            parsed = self._parse_classification(raw, self.VALID_COLORS)
            return PredictionResult(
                task=Task.COLOR_CLASSIFICATION,
                raw_output=raw,
                parsed_label=parsed,
                is_valid=parsed is not None,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.warning(f"[{self.MODEL_NAME}] predict_color failed: {e}")
            return PredictionResult(
                task=Task.COLOR_CLASSIFICATION,
                raw_output="",
                is_valid=False,
                error=str(e),
            )
 
    # ------------------------------------------------------------------
    # predict_bbox  — not supported
    # ------------------------------------------------------------------
 
    def predict_bbox(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
        image_size: tuple[int, int] = (1280, 720),
    ) -> PredictionResult:
        """
        ViLT has no spatial output capability.
        Returns is_valid=False with an explanatory message.
        """
        return PredictionResult(
            task=Task.DETECTION,
            raw_output="",
            bbox_xyxy=None,
            is_valid=False,
            error="ViLT does not support object detection",
        )