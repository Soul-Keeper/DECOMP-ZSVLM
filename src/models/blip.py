"""
blip_model.py — BLIP wrapper for the garment benchmark

Model:   ybelkada/blip-vqa-capfilt-large
Params:  ~384M
VRAM:    ~4GB
Device:  RTX 4060 Ti / 4070 Super

BLIP uses a ViT-based vision encoder with a multimodal fusion transformer
and text decoder, trained with contrastive + generative objectives.
Inference is framed as VQA — same as ViLT, but BLIP has a generative
decoder so it can produce arbitrary tokens, not just VQA vocabulary labels.

Key difference from ViLT:
  - Output is generated text, not argmax over fixed vocab
  - Can in principle output any word including "tshirt", "sweatshirt", "nude"
  - Still no spatial output → detection returns is_valid=False
"""

import logging

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

from src.base_model import (
    BaseGarmentModel,
    PromptConfig,
    PredictionResult,
    Task,
)

logger = logging.getLogger(__name__)

MODEL_ID = "ybelkada/blip-vqa-capfilt-large"


class BLIPModel(BaseGarmentModel):

    MODEL_NAME = "BLIP-VQA-CAPFILT-LARGE"
    SUPPORTS_MULTITASK = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        logger.info(f"[{self.MODEL_NAME}] Loading from {MODEL_ID} ...")
        self._processor = BlipProcessor.from_pretrained(MODEL_ID)
        self._model = BlipForQuestionAnswering.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        ).to(self.device)
        self._model.eval()
        self._loaded = True
        params = self._model.num_parameters()
        logger.info(f"[{self.MODEL_NAME}] Loaded — {params/1e6:.2f}M parameters on {self.device}")

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    def _ask(self, image: Image.Image, question: str) -> tuple[str, float]:
        """
        Run VQA inference and return (raw_answer_string, latency_ms).

        BLIP generates text autoregressively — output is a free-form string,
        not constrained to a fixed vocabulary like ViLT.
        """
        inputs = self._processor(
            image, question, return_tensors="pt"
        ).to(self.device, torch.float16)

        def _forward():
            with torch.no_grad():
                out = self._model.generate(**inputs)
            return self._processor.decode(out[0], skip_special_tokens=True)

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
    # predict_bbox — not supported
    # ------------------------------------------------------------------

    def predict_bbox(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
        image_size: tuple[int, int] = (1280, 720),
    ) -> PredictionResult:
        """BLIP has no spatial output capability."""
        return PredictionResult(
            task=Task.DETECTION,
            raw_output="",
            bbox_xyxy=None,
            is_valid=False,
            error="BLIP does not support object detection",
        )
