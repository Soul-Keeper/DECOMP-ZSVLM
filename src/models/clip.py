import json
import logging

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.base_model import (
    BaseGarmentModel,
    PredictionResult,
    PromptConfig,
    Task,
)

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-large-patch14"


class CLIPGarmentModel(BaseGarmentModel):
    """
    CLIP scores candidate label embeddings against the image embedding.

    Prompts are JSON lists of candidates rather than questions, and the
    candidate wording is what the strategies vary. raw_output holds the
    probability of every candidate, so the margin between them stays available.
    No spatial output.
    """

    MODEL_NAME = "CLIP-VIT-L14"
    SUPPORTS_MULTITASK = False

    def load(self) -> None:
        logger.info(f"[{self.MODEL_NAME}] loading {MODEL_ID}")
        self._processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self._model = CLIPModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self._model.eval()
        self._loaded = True
        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(f"[{self.MODEL_NAME}] loaded ({n_params:,} params on {self.device})")

    @staticmethod
    def _parse_candidates(prompt: str) -> list[str]:
        try:
            candidates = json.loads(prompt)
            if isinstance(candidates, list) and len(candidates) >= 2:
                return candidates
        except (json.JSONDecodeError, TypeError):
            pass
        raise ValueError(f"CLIP prompt must be a JSON list of candidates, got {prompt!r}")

    def _score(self, image: Image.Image, candidates: list[str]) -> tuple[str, str]:
        inputs = self._processor(
            text=candidates, images=image, return_tensors="pt", padding=True,
        ).to(self.device)

        with torch.no_grad():
            probs = self._model(**inputs).logits_per_image.softmax(dim=-1)[0]

        winner = candidates[probs.argmax().item()]
        scores = {c: round(probs[i].item(), 4) for i, c in enumerate(candidates)}
        return winner, json.dumps(scores)

    def _classify(self, image, prompt, task, labels) -> PredictionResult:
        self._require_loaded()

        def run():
            winner, raw = self._score(image, self._parse_candidates(prompt))
            # The winner may be a phrase such as "a photo of a tshirt".
            return raw, self._parse_classification(winner, labels)

        try:
            (raw, parsed), latency = self._timed(run)
            return PredictionResult(
                task=task, raw_output=raw, parsed_label=parsed,
                is_valid=parsed is not None, latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] {task.value} failed: {e}")
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
            task=Task.DETECTION, raw_output="detection_not_supported",
            bbox_xyxy=None, is_valid=False, latency_ms=0.0,
        )
