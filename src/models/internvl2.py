import logging

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from src.base_model import BaseGarmentModel, PredictionResult, PromptConfig, Task

logger = logging.getLogger(__name__)

MODEL_ID = "OpenGVLab/InternVL2-8B"
TILE = 448


def _try_flash_attn() -> bool:
    try:
        import flash_attn  # noqa
        return True
    except ImportError:
        return False


def _is_usable(tokenizer) -> bool:
    """A tokenizer is usable if it has the methods chat() calls on it."""
    return (tokenizer is not None
            and not isinstance(tokenizer, bool)
            and callable(getattr(tokenizer, "convert_tokens_to_ids", None)))


def _load_tokenizer(model_id: str):
    """
    AutoTokenizer resolves to something unusable for some revisions of this
    model, so four loading paths are tried in order and the first usable
    tokenizer wins. Failing here rather than on the first inference keeps the
    error a minute and a half earlier.
    """
    attempts = []

    for kwargs in ({"trust_remote_code": True},
                   {"trust_remote_code": True, "use_fast": False}):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
            if _is_usable(tokenizer):
                logger.info(f"  tokenizer: AutoTokenizer {kwargs} -> {type(tokenizer).__name__}")
                return tokenizer
            attempts.append(f"AutoTokenizer {kwargs}: returned {type(tokenizer).__name__}")
        except Exception as e:
            attempts.append(f"AutoTokenizer {kwargs}: {type(e).__name__} {e}")

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        for ref in ("tokenization_internlm2.InternLM2Tokenizer",
                    "tokenization_internlm2_fast.InternLM2TokenizerFast"):
            try:
                cls = get_class_from_dynamic_module(ref, model_id)
                tokenizer = cls.from_pretrained(model_id, trust_remote_code=True)
                if _is_usable(tokenizer):
                    logger.info(f"  tokenizer: {ref} -> {type(tokenizer).__name__}")
                    return tokenizer
                attempts.append(f"{ref}: returned {type(tokenizer).__name__}")
            except Exception as e:
                attempts.append(f"{ref}: {type(e).__name__} {e}")
    except ImportError as e:
        attempts.append(f"get_class_from_dynamic_module unavailable: {e}")

    raise RuntimeError("could not load the InternVL2 tokenizer:\n  " + "\n  ".join(attempts))


class InternVLModel(BaseGarmentModel):
    """InternVL2-8B. Coordinates are free text; multi-task is a single pass."""

    MODEL_NAME = "INTERNVL2-8B"
    SUPPORTS_MULTITASK = True

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._tokenizer = None

    def load(self) -> None:
        use_flash = _try_flash_attn()
        logger.info(f"[{self.MODEL_NAME}] loading {MODEL_ID} (flash_attn={use_flash})")
        self._model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=use_flash,
        ).eval().to(self.device)
        self._tokenizer = _load_tokenizer(MODEL_ID)
        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] loaded ({self._model.num_parameters():,} params)")

    def unload(self) -> None:
        super().unload()
        self._tokenizer = None

    def _transform(self):
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((TILE, TILE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def _preprocess(self, image: Image.Image, max_tiles: int = 12) -> torch.Tensor:
        """Tile the image to the closest aspect ratio; add a thumbnail if tiled."""
        width, height = image.size
        aspect = width / height

        ratios = sorted(
            {(i, j)
             for n in range(1, max_tiles + 1)
             for i in range(1, n + 1)
             for j in range(1, n + 1)
             if i * j <= max_tiles},
            key=lambda x: x[0] * x[1],
        )
        cols, rows = min(ratios, key=lambda r: abs(aspect - r[0] / r[1]))

        resized = image.resize((TILE * cols, TILE * rows))
        transform = self._transform()
        tiles = [
            transform(resized.crop((c * TILE, r * TILE, (c + 1) * TILE, (r + 1) * TILE)))
            for r in range(rows)
            for c in range(cols)
        ]
        if len(tiles) > 1:
            tiles.append(transform(image.resize((TILE, TILE))))

        return torch.stack(tiles).to(torch.bfloat16).to(self.device)

    def _chat(self, pixel_values: torch.Tensor, question: str,
              max_new_tokens: int = 32) -> tuple[str, float]:
        config = dict(max_new_tokens=max_new_tokens, do_sample=False)

        def run():
            with torch.inference_mode():
                # Keyword arguments: the parameter order in the remote code can
                # change when the model repository is updated.
                return self._model.chat(
                    tokenizer=self._tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=config,
                ).strip()

        return self._timed(run)

    def _run(self, image, prompt, task, max_new_tokens, finish):
        self._require_loaded()
        try:
            raw, latency = self._chat(self._preprocess(image), prompt, max_new_tokens)
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

        return self._run(image, prompt, task, 32, finish)

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

        return self._run(image, prompt_config.detection_prompt, Task.DETECTION, 128, finish)

    def predict_multitask(self, image: Image.Image, prompt_config: PromptConfig,
                          image_size: tuple[int, int] = (1280, 720)) -> PredictionResult:
        """
        One forward pass for all three tasks. The image is encoded once and the
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
