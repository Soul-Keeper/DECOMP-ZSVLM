import torch
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from src.base_model import (
    BaseGarmentModel,
    PredictionResult,
    PromptConfig,
    Task,
)

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/Florence-2-large"


class FlorenceModel(BaseGarmentModel):

    MODEL_NAME = "FLORENCE-2-LARGE"

    # ------------------------------------------------

    def load(self):

        logger.info(f"[{self.MODEL_NAME}] Loading {MODEL_ID}")

        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager",  # отключает flash-attn, Florence с ним нестабильна
            low_cpu_mem_usage=True,
        ).to(self.device)

        self._model.eval()
        self._loaded = True

        logger.info(f"[{self.MODEL_NAME}] Model loaded")

    # ------------------------------------------------

    def _generate(self, image: Image.Image, prompt: str):

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        # Явный каст: pixel_values должны быть float16 на cuda,
        # иначе Florence падает с dtype mismatch
        inputs = {
            k: (v.to(self.device, dtype=torch.float16)
                if v.dtype == torch.float32 and self.device == "cuda"
                else v.to(self.device))
            for k, v in inputs.items()
        }

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,  # нужно для post_process_generation
        )[0]

        return generated_ids, generated_text

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
            _, text = self._generate(image, prompt_config.type_prompt)
            return text

        try:
            raw, latency = self._timed(run)

            parsed = self._parse_classification(raw, self.VALID_TYPES)

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
            _, text = self._generate(image, prompt_config.color_prompt)
            return text

        try:
            raw, latency = self._timed(run)

            parsed = self._parse_classification(raw, self.VALID_COLORS)

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
        image_size=(1280, 720),
    ) -> PredictionResult:

        self._require_loaded()

        def run():

            ids, text = self._generate(image, prompt_config.detection_prompt)

            result = self._processor.post_process_generation(
                text,
                task="<OD>",
                image_size=image_size,
            )

            bbox = None

            if "<OD>" in result and len(result["<OD>"]["bboxes"]) > 0:
                # Florence возвращает абсолютные координаты [x1, y1, x2, y2]
                raw_bbox = result["<OD>"]["bboxes"][0]
                bbox = [int(c) for c in raw_bbox]

            return text, bbox

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
