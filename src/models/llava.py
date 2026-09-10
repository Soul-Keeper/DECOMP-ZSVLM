import logging

import torch
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


class LLaVAModel(BaseGarmentModel):
    """
    LLaVA-1.6 (NeXT) on Mistral-7B. Coordinates are free text.

    Prompts from src/prompts.py carry no chat markup; it is added here, since
    the model ignores the image without it. The processor tiles the image
    itself, so the original is passed through unresized.
    """

    MODEL_NAME = "LLAVA-1.6-MISTRAL-7B"
    SUPPORTS_MULTITASK = False

    def load(self) -> None:
        logger.info(f"[{self.MODEL_NAME}] loading {MODEL_ID}")
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Mistral has no pad token and the processor needs one.
        if self._processor.tokenizer.pad_token is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token

        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        self._loaded = True
        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(f"[{self.MODEL_NAME}] loaded ({n_params:,} params on {self.device})")

    @staticmethod
    def _apply_chat_template(prompt: str) -> str:
        """Wrap in the Mistral template; <image> has to sit inside [INST]."""
        if "<image>" not in prompt:
            prompt = f"<image>\n{prompt}"
        return f"[INST] {prompt} [/INST]"

    def _generate(self, image: Image.Image, prompt: str) -> str:
        inputs = self._processor(
            text=self._apply_chat_template(prompt), images=image, return_tensors="pt",
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        # The model echoes the prompt, so drop the input tokens.
        return self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=True).strip()

    def _run(self, image, prompt, task, finish) -> PredictionResult:
        self._require_loaded()
        try:
            raw, latency = self._timed(self._generate, image, prompt)
            return finish(raw, latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] {task.value} failed: {e}")
            return PredictionResult(task=task, raw_output="", is_valid=False, error=str(e))

    def _classify(self, image, prompt, task, labels) -> PredictionResult:
        def finish(raw, latency):
            parsed = self._parse_classification(raw, labels)
            return PredictionResult(
                task=task, raw_output=raw, parsed_label=parsed,
                is_valid=parsed is not None, latency_ms=latency,
            )

        return self._run(image, prompt, task, finish)

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
        def finish(raw, latency):
            bbox = self._parse_bbox(raw, image_size)
            return PredictionResult(
                task=Task.DETECTION, raw_output=raw, bbox_xyxy=bbox,
                is_valid=bbox is not None, latency_ms=latency,
            )

        return self._run(image, prompt_config.detection_prompt, Task.DETECTION, finish)
