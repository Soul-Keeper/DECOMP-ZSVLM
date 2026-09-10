import logging

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from src.base_model import BaseGarmentModel, PredictionResult, PromptConfig, Task

logger = logging.getLogger(__name__)

MODEL_ID = "openbmb/MiniCPM-Llama3-V-2_5"
REVISION = "fd7f352fac0e06d0d818b23f98e3ec8c64267a57"


class MiniCPMModel(BaseGarmentModel):
    """MiniCPM-V-2.5. Coordinates are free text; multi-task is a single pass."""

    MODEL_NAME = "MINICPM-LLAMA3-V-2.5"
    SUPPORTS_MULTITASK = True

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._tokenizer = None

    def load(self) -> None:
        logger.info(f"[{self.MODEL_NAME}] loading {MODEL_ID}")
        self._model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval().to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True)
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] loaded ({self._model.num_parameters():,} params)")

    def unload(self) -> None:
        super().unload()
        self._tokenizer = None

    def _chat(self, image: Image.Image, question: str,
              max_new_tokens: int) -> tuple[str, float]:
        messages = [{"role": "user", "content": question}]

        def run():
            with torch.inference_mode():
                return self._model.chat(
                    image=image,
                    msgs=messages,
                    tokenizer=self._tokenizer,
                    sampling=False,
                    max_new_tokens=max_new_tokens,
                ).strip()

        return self._timed(run)

    def _run(self, image, prompt, task, max_new_tokens, finish) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._chat(image, prompt, max_new_tokens)
            return finish(raw, latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] {task.value} failed: {e}")
            return PredictionResult(task=task, raw_output="", is_valid=False, error=str(e))

    def _classify(self, image, prompt, task, labels, max_new_tokens) -> PredictionResult:
        def finish(raw, latency):
            parsed = self._parse_classification(raw, labels)
            return PredictionResult(
                task=task, raw_output=raw, parsed_label=parsed,
                is_valid=parsed is not None, latency_ms=latency,
            )

        return self._run(image, prompt, task, max_new_tokens, finish)

    def predict_type(self, image: Image.Image,
                     prompt_config: PromptConfig) -> PredictionResult:
        return self._classify(image, prompt_config.type_prompt,
                              Task.TYPE_CLASSIFICATION, self.VALID_TYPES, 16)

    def predict_color(self, image: Image.Image,
                      prompt_config: PromptConfig) -> PredictionResult:
        return self._classify(image, prompt_config.color_prompt,
                              Task.COLOR_CLASSIFICATION, self.VALID_COLORS, 32)

    def predict_bbox(self, image: Image.Image, prompt_config: PromptConfig,
                     image_size: tuple[int, int] = (1280, 720)) -> PredictionResult:
        def finish(raw, latency):
            bbox = self._parse_bbox(raw, image_size)
            return PredictionResult(
                task=Task.DETECTION, raw_output=raw, bbox_xyxy=bbox,
                is_valid=bbox is not None, latency_ms=latency,
            )

        return self._run(image, prompt_config.detection_prompt, Task.DETECTION, 256, finish)

    def predict_multitask(self, image: Image.Image, prompt_config: PromptConfig,
                          image_size: tuple[int, int] = (1280, 720)) -> PredictionResult:
        """
        One chat() call for all three tasks. The image is encoded once and the
        prompt asks for a structured response:

            Type: <type>
            Color: <color>
            Box: [x_min, y_min, x_max, y_max]

        raw_output keeps the full response; run_benchmark re-parses it.
        """
        if not prompt_config.multitask_prompt:
            raise ValueError(f"[{self.MODEL_NAME}] multitask_prompt is not set")

        def finish(raw, latency):
            parsed_type = parsed_color = bbox = None
            for line in raw.splitlines():
                head = line.strip().lower()
                value = line.split(":", 1)[1].strip() if ":" in line else ""
                if head.startswith("type:"):
                    parsed_type = self._parse_classification(value, self.VALID_TYPES)
                elif head.startswith("color:"):
                    parsed_color = self._parse_classification(value, self.VALID_COLORS)
                elif head.startswith("box:"):
                    bbox = self._parse_bbox(value, image_size)
            return PredictionResult(
                task=Task.MULTITASK, raw_output=raw,
                parsed_type=parsed_type, parsed_color=parsed_color, bbox_xyxy=bbox,
                is_valid=parsed_type is not None and parsed_color is not None,
                latency_ms=latency,
            )

        return self._run(image, prompt_config.multitask_prompt, Task.MULTITASK, 256, finish)
