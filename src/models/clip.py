import json
import torch
import logging
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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
    CLIP wrapper for the garment benchmark.

    Принципиальное отличие от других моделей:
    CLIP не генерирует текст — он вычисляет cosine similarity между
    эмбеддингом изображения и эмбеддингами текстовых кандидатов,
    затем выбирает наиболее похожий.

    Ablation стратегии задаются через PromptConfig как JSON-список кандидатов:
        label_only:   '["tshirt", "sweatshirt"]'
        descriptive:  '["a photo of a tshirt", "a photo of a sweatshirt"]'
        contextual:   '["a tshirt lying flat on a conveyor belt",
                        "a sweatshirt lying flat on a conveyor belt"]'

    Detection: не поддерживается — CLIP не имеет spatial output.
    """

    MODEL_NAME = "CLIP-VIT-L14"
    SUPPORTS_MULTITASK = False

    # ------------------------------------------------

    def load(self) -> None:

        logger.info(f"[{self.MODEL_NAME}] Loading {MODEL_ID}")

        self._processor = CLIPProcessor.from_pretrained(MODEL_ID)

        self._model = CLIPModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self._model.eval()
        self._loaded = True

        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(f"[{self.MODEL_NAME}] Loaded — {n_params:.0f}M params")

    # ------------------------------------------------

    def _classify(
        self,
        image: Image.Image,
        candidates: list[str],
    ) -> tuple[str, str]:
        """
        Возвращает (winning_label, raw_output).
        raw_output — строка с вероятностями всех кандидатов для сохранения.
        """
        inputs = self._processor(
            text=candidates,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # logits_per_image: [1, n_candidates]
            probs = outputs.logits_per_image.softmax(dim=-1)[0]

        best_idx = probs.argmax().item()
        best_label = candidates[best_idx]

        # Сохраняем все вероятности для error analysis
        scores = {c: round(probs[i].item(), 4) for i, c in enumerate(candidates)}
        raw = json.dumps(scores)

        return best_label, raw

    @staticmethod
    def _parse_candidates(prompt: str) -> list[str]:
        """
        Парсит JSON-список кандидатов из строки промпта.
        Пример: '["tshirt", "sweatshirt"]' -> ["tshirt", "sweatshirt"]
        """
        try:
            candidates = json.loads(prompt)
            if isinstance(candidates, list) and len(candidates) >= 2:
                return candidates
        except (json.JSONDecodeError, TypeError):
            pass
        raise ValueError(
            f"CLIP prompt must be a JSON list of candidates, got: {prompt!r}"
        )

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
            candidates = self._parse_candidates(prompt_config.type_prompt)
            winner, raw = self._classify(image, candidates)
            # Маппинг: кандидат может быть "a photo of a tshirt" -> "tshirt"
            parsed = self._parse_classification(winner, self.VALID_TYPES)
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
            candidates = self._parse_candidates(prompt_config.color_prompt)
            winner, raw = self._classify(image, candidates)
            parsed = self._parse_classification(winner, self.VALID_COLORS)
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
    # DETECTION — не поддерживается
    # ------------------------------------------------

    def predict_bbox(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
        image_size: tuple[int, int] = (1280, 720),
    ) -> PredictionResult:

        # CLIP не имеет spatial output — detection не поддерживается.
        # is_valid=False корректно отразится в метриках (IoU = 0).
        return PredictionResult(
            task=Task.DETECTION,
            raw_output="detection_not_supported",
            bbox_xyxy=None,
            is_valid=False,
            latency_ms=0.0,
        )
