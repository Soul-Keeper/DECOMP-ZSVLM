import logging

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from src.base_model import (
    BaseGarmentModel,
    PredictionResult,
    PromptConfig,
    Task,
)

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/Florence-2-large"


class FlorenceModel(BaseGarmentModel):
    """Florence-2. Boxes are decoded from <loc> tokens by the processor."""

    MODEL_NAME = "FLORENCE-2-LARGE"
    SUPPORTS_MULTITASK = False

    def load(self) -> None:
        logger.info(f"[{self.MODEL_NAME}] loading {MODEL_ID}")
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager",  # Florence-2 is unstable with flash-attn
            low_cpu_mem_usage=True,
        ).to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] loaded on {self.device}")

    def _generate(self, image: Image.Image, prompt: str) -> str:
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        # pixel_values must be float16 on cuda or generation fails on a dtype mismatch.
        inputs = {
            k: (v.to(self.device, dtype=torch.float16)
                if v.dtype == torch.float32 and self.device == "cuda"
                else v.to(self.device))
            for k, v in inputs.items()
        }

        generated = self._model.generate(**inputs, max_new_tokens=64, do_sample=False)
        # Special tokens are kept because post_process_generation needs them.
        return self._processor.batch_decode(generated, skip_special_tokens=False)[0]

    def _classify(self, image, prompt, task, labels) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._timed(self._generate, image, prompt)
            parsed = self._parse_classification(raw, labels)
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
        self._require_loaded()

        def run():
            text = self._generate(image, prompt_config.detection_prompt)
            result = self._processor.post_process_generation(
                text, task="<OD>", image_size=image_size)
            boxes = result.get("<OD>", {}).get("bboxes", [])
            # Florence-2 returns absolute [x1, y1, x2, y2].
            return text, [int(c) for c in boxes[0]] if boxes else None

        try:
            (raw, bbox), latency = self._timed(run)
            return PredictionResult(
                task=Task.DETECTION, raw_output=raw, bbox_xyxy=bbox,
                is_valid=bbox is not None, latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_bbox failed: {e}")
            return PredictionResult(task=Task.DETECTION, raw_output="",
                                    is_valid=False, error=str(e))
