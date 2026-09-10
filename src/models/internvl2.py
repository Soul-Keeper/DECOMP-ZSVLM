"""
InternVL2-8B wrapper.

Поддерживает два режима:
  - Single-task : три отдельных вызова _chat() (E1 основной бенчмарк, E2 ablation)
  - Multi-task  : один вызов predict_multitask() с multitask_prompt (E5)

Препроцессинг dynamic_preprocess выполняется один раз на изображение —
в single-task режиме это не экономит время (три отдельных вызова chat()),
в multi-task режиме весь инференс — один forward pass.

ЗАГРУЗКА ТОКЕНИЗАТОРА
---------------------
На transformers 4.49 AutoTokenizer.from_pretrained для этой модели
возвращает bool вместо объекта токенизатора — резолвер не справляется
с auto_map удалённого кода. Ошибка тихая: модель грузится, падает только
первый вызов chat() с 'bool' object has no attribute convert_tokens_to_ids.

Поэтому реализована цепочка способов с проверкой результата на входе,
а не на первом инференсе.
"""

import torch
import logging
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer

from src.base_model import BaseGarmentModel, PredictionResult, PromptConfig, Task

logger = logging.getLogger(__name__)

MODEL_ID = "OpenGVLab/InternVL2-8B"


def _try_flash_attn() -> bool:
    try:
        import flash_attn  # noqa
        return True
    except ImportError:
        return False


def _is_usable_tokenizer(tok) -> bool:
    """Токенизатор годен, если у него есть методы, которые вызывает chat()."""
    return (tok is not None
            and not isinstance(tok, bool)
            and hasattr(tok, "convert_tokens_to_ids")
            and callable(getattr(tok, "convert_tokens_to_ids", None)))


def _load_tokenizer(model_id: str):
    """
    Цепочка способов загрузки. Возвращает первый годный токенизатор.

    1. AutoTokenizer без use_fast   — штатный путь
    2. AutoTokenizer с use_fast=False
    3. Класс напрямую из удалённого модуля, минуя резолвер AutoTokenizer
    4. Fast-версия класса из удалённого модуля
    """
    attempts = []

    for kwargs in ({"trust_remote_code": True},
                   {"trust_remote_code": True, "use_fast": False}):
        try:
            tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
            if _is_usable_tokenizer(tok):
                logger.info(f"  токенизатор: AutoTokenizer {kwargs} -> "
                            f"{type(tok).__name__}")
                return tok
            attempts.append(f"AutoTokenizer {kwargs}: вернул {type(tok).__name__}")
        except Exception as e:
            attempts.append(f"AutoTokenizer {kwargs}: {type(e).__name__} {e}")

    # Обход резолвера: берём класс прямо из динамического модуля
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        for ref in ("tokenization_internlm2.InternLM2Tokenizer",
                    "tokenization_internlm2_fast.InternLM2TokenizerFast"):
            try:
                cls = get_class_from_dynamic_module(ref, model_id)
                tok = cls.from_pretrained(model_id, trust_remote_code=True)
                if _is_usable_tokenizer(tok):
                    logger.info(f"  токенизатор: {ref} -> {type(tok).__name__}")
                    return tok
                attempts.append(f"{ref}: вернул {type(tok).__name__}")
            except Exception as e:
                attempts.append(f"{ref}: {type(e).__name__} {e}")
    except ImportError as e:
        attempts.append(f"get_class_from_dynamic_module недоступен: {e}")

    raise RuntimeError(
        "Не удалось загрузить токенизатор InternVL2. Попытки:\n  "
        + "\n  ".join(attempts)
    )


class InternVLModel(BaseGarmentModel):
    MODEL_NAME = "INTERNVL2-8B"
    SUPPORTS_MULTITASK = True           # genuine single-pass via multitask_prompt

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._tokenizer = None

    def load(self):
        use_flash = _try_flash_attn()
        logger.info(f"[{self.MODEL_NAME}] Loading {MODEL_ID} (flash_attn={use_flash})")

        self._model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=use_flash,
        ).eval().to(self.device)

        # Падаем здесь, а не на первом инференсе через полторы минуты
        self._tokenizer = _load_tokenizer(MODEL_ID)

        self._loaded = True
        logger.info(f"[{self.MODEL_NAME}] Loaded ({self._model.num_parameters():,} params)")

    # ==================== PREPROCESSING ====================

    def _build_transform(self):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def _dynamic_preprocess(self, image: Image.Image, max_num: int = 12):
        """Tile image into patches matching the aspect ratio. Adds thumbnail if >1 tile."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = sorted(
            {(i, j)
             for n in range(1, max_num + 1)
             for i in range(1, n + 1)
             for j in range(1, n + 1)
             if i * j <= max_num},
            key=lambda x: x[0] * x[1],
        )

        cols, rows = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0] / r[1]))
        resized = image.resize((448 * cols, 448 * rows))
        transform = self._build_transform()

        patches = [
            transform(resized.crop((c * 448, r * 448, (c + 1) * 448, (r + 1) * 448)))
            for r in range(rows)
            for c in range(cols)
        ]
        if len(patches) > 1:
            patches.append(transform(image.resize((448, 448))))

        return torch.stack(patches).to(torch.bfloat16).to(self.device)

    # ==================== CORE INFERENCE ====================

    def _chat(self, pixel_values: torch.Tensor, question: str,
              max_new_tokens: int = 32) -> tuple[str, float]:
        """Single question → single answer. pixel_values already on device."""
        gen_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

        def run():
            with torch.inference_mode():
                # Именованные аргументы: порядок параметров в удалённом коде
                # может измениться при обновлении репозитория модели
                return self._model.chat(
                    tokenizer=self._tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=gen_config,
                ).strip()

        return self._timed(run)

    # ==================== SINGLE-TASK (E1, E2) ====================

    def predict_type(self, image: Image.Image, prompt_config: PromptConfig) -> PredictionResult:
        self._require_loaded()
        try:
            pv = self._dynamic_preprocess(image)
            raw, latency = self._chat(pv, prompt_config.type_prompt, max_new_tokens=32)
            parsed = self._parse_classification(raw, self.VALID_TYPES)
            return PredictionResult(task=Task.TYPE_CLASSIFICATION, raw_output=raw,
                                    parsed_label=parsed, is_valid=(parsed is not None),
                                    latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_type failed: {e}")
            return PredictionResult(task=Task.TYPE_CLASSIFICATION, raw_output="",
                                    is_valid=False, error=str(e))

    def predict_color(self, image: Image.Image, prompt_config: PromptConfig) -> PredictionResult:
        self._require_loaded()
        try:
            pv = self._dynamic_preprocess(image)
            raw, latency = self._chat(pv, prompt_config.color_prompt, max_new_tokens=32)
            parsed = self._parse_classification(raw, self.VALID_COLORS)
            return PredictionResult(task=Task.COLOR_CLASSIFICATION, raw_output=raw,
                                    parsed_label=parsed, is_valid=(parsed is not None),
                                    latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_color failed: {e}")
            return PredictionResult(task=Task.COLOR_CLASSIFICATION, raw_output="",
                                    is_valid=False, error=str(e))

    def predict_bbox(self, image: Image.Image, prompt_config: PromptConfig,
                     image_size=(1280, 720)) -> PredictionResult:
        self._require_loaded()
        try:
            pv = self._dynamic_preprocess(image)
            raw, latency = self._chat(pv, prompt_config.detection_prompt, max_new_tokens=128)
            bbox_xyxy = self._parse_bbox(raw, image_size)
            return PredictionResult(task=Task.DETECTION, raw_output=raw,
                                    bbox_xyxy=bbox_xyxy, is_valid=(bbox_xyxy is not None),
                                    latency_ms=latency)
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_bbox failed: {e}")
            return PredictionResult(task=Task.DETECTION, raw_output="",
                                    is_valid=False, error=str(e))

    # ==================== MULTI-TASK (E5) ====================

    def predict_multitask(self, image: Image.Image, prompt_config: PromptConfig,
                          image_size: tuple[int, int] = (1280, 720)) -> PredictionResult:
        """
        Genuine single-pass multi-task inference.

        Encodes the image once, asks all three questions in one prompt.
        The multitask_prompt instructs the model to respond in structured format:
            Type: <type>
            Color: <color>
            Box: [x_min, y_min, x_max, y_max]

        raw_output contains the full model response — parsed upstream in run_benchmark.
        """
        self._require_loaded()
        if not prompt_config.multitask_prompt:
            raise ValueError(f"[{self.MODEL_NAME}] multitask_prompt is not set in PromptConfig")

        try:
            pv = self._dynamic_preprocess(image)
            raw, latency = self._chat(pv, prompt_config.multitask_prompt, max_new_tokens=256)

            parsed_type  = None
            parsed_color = None
            bbox_xyxy    = None

            for line in raw.splitlines():
                line_low = line.strip().lower()
                if line_low.startswith("type:"):
                    parsed_type = self._parse_classification(
                        line.split(":", 1)[1].strip(), self.VALID_TYPES)
                elif line_low.startswith("color:"):
                    parsed_color = self._parse_classification(
                        line.split(":", 1)[1].strip(), self.VALID_COLORS)
                elif line_low.startswith("box:"):
                    bbox_xyxy = self._parse_bbox(
                        line.split(":", 1)[1].strip(), image_size)

            return PredictionResult(
                task=Task.MULTITASK,
                raw_output=raw,
                parsed_type=parsed_type,
                parsed_color=parsed_color,
                bbox_xyxy=bbox_xyxy,
                is_valid=(parsed_type is not None and parsed_color is not None),
                latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"[{self.MODEL_NAME}] predict_multitask failed: {e}")
            return PredictionResult(task=Task.MULTITASK, raw_output="",
                                    is_valid=False, error=str(e))

    def unload(self):
        super().unload()
        self._tokenizer = None
