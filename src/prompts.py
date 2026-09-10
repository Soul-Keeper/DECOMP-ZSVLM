"""
Central registry of prompt strategies.

Each model maps strategy name -> PromptConfig. Four strategies are evaluated:

    label_only    candidate classes as bare strings, no question
    question      open query, no label set
    constrained   open query with explicit label enumeration and a length limit
    descriptive   production scene context before the query

Two further strategies are not part of the four-way grid. CLIP scores label
embeddings rather than answering a question, so it uses contextual in place of
question and constrained. descriptive_cat mirrors descriptive clause by clause
and replaces only the scene, and is used on the external catalog subsets.

DEFAULT_STRATEGIES holds the joint strategy of each model: the one maximizing
mean adjusted accuracy across the tasks the model supports. These are the
configurations reported throughout the paper.
"""

from src.base_model import PromptConfig


VILT_PROMPTS: dict[str, PromptConfig] = {
    "label_only": PromptConfig(
        strategy="label_only",
        type_prompt="tshirt or sweatshirt?",
        color_prompt="black, brown, nude, white or pink?",
        detection_prompt="",  # not supported
    ),
    "question": PromptConfig(
        strategy="question",
        type_prompt="What type of clothing is this?",
        color_prompt="What color is this clothing?",
        detection_prompt="",
    ),
    "constrained": PromptConfig(
        strategy="constrained",
        type_prompt="Is this a tshirt or sweatshirt?",
        color_prompt="Is this clothing black, brown, nude, white or pink?",
        detection_prompt="",
    ),
}

VILT_DEFAULT = "constrained"

BLIP_PROMPTS: dict[str, PromptConfig] = {
    "label_only": PromptConfig(
        strategy="label_only",
        type_prompt="tshirt or sweatshirt?",
        color_prompt="black, brown, nude, white or pink?",
        detection_prompt="",
    ),
    "question": PromptConfig(
        strategy="question",
        type_prompt="What type of garment is on the table?",
        color_prompt="What color is the garment on the table?",
        detection_prompt="",
    ),
    "constrained": PromptConfig(
        strategy="constrained",
        type_prompt="What is the person working with, a tshirt or sweatshirt?",
        color_prompt="What is the color of the clothing? Choose from: black, brown, nude, white, pink.",
        detection_prompt="",
    ),
}

BLIP_DEFAULT = "label_only"

FLORENCE_PROMPTS: dict[str, PromptConfig] = {
    "label_only": PromptConfig(
        strategy="label_only",
        type_prompt="tshirt or sweatshirt",
        color_prompt="black brown nude white pink",
        detection_prompt="the biggest piece of clothing",
    ),
    "question": PromptConfig(
        strategy="question",
        type_prompt="What type of garment is this?",
        color_prompt="What color is the garment?",
        detection_prompt="clothing item",
    ),
    "constrained": PromptConfig(
        strategy="constrained",
        type_prompt="Is this a tshirt or sweatshirt?",
        color_prompt="Is the color black, brown, nude, white or pink?",
        detection_prompt="the garment lying flat on the surface",
    ),
}

FLORENCE_DEFAULT = "label_only"

# CLIP takes a JSON list of candidates rather than a question, and picks the
# candidate whose text embedding is closest to the image embedding. The
# strategies vary how each candidate is described, not the question structure.

CLIP_PROMPTS: dict[str, PromptConfig] = {
    "label_only": PromptConfig(
        strategy="label_only",
        type_prompt='["tshirt", "sweatshirt"]',
        color_prompt='["black", "brown", "nude", "white", "pink"]',
        detection_prompt="",  # not supported
    ),
    "descriptive": PromptConfig(
        strategy="descriptive",
        type_prompt='["a photo of a tshirt", "a photo of a sweatshirt"]',
        color_prompt='["a photo of a black garment", "a photo of a brown garment", '
                     '"a photo of a nude garment", "a photo of a white garment", '
                     '"a photo of a pink garment"]',
        detection_prompt="",
    ),
    "contextual": PromptConfig(
        strategy="contextual",
        type_prompt='["a tshirt being ironed by a worker on an industrial table", '
                    '"a sweatshirt being ironed by a worker on an industrial table"]',
        color_prompt='["a black garment being ironed by a worker on an industrial table", '
                     '"a brown garment being ironed by a worker on an industrial table", '
                     '"a nude garment being ironed by a worker on an industrial table", '
                     '"a white garment being ironed by a worker on an industrial table", '
                     '"a pink garment being ironed by a worker on an industrial table"]',
        detection_prompt="",
    ),
    "descriptive_cat": PromptConfig(
        strategy="descriptive_cat",
        type_prompt='["a studio product photo of a tshirt", "a studio product photo of a sweatshirt"]',
        color_prompt='["a studio product photo of a black garment", "a studio product photo of a brown garment", "a studio product photo of a nude garment", "a studio product photo of a white garment", "a studio product photo of a pink garment"]',
        detection_prompt="",
    ),
}

CLIP_DEFAULT = "descriptive"

PALIGEMMA_PROMPTS: dict[str, PromptConfig] = {
    "label_only": PromptConfig(
        strategy="label_only",
        type_prompt="tshirt or sweatshirt?",
        color_prompt="black, brown, nude, white or pink?",
        detection_prompt="detect clothing",
    ),
    "question": PromptConfig(
        strategy="question",
        type_prompt="What type of garment is this?",
        color_prompt="What color is the garment?",
        detection_prompt="detect the garment",
    ),
    "constrained": PromptConfig(
        strategy="constrained",
        type_prompt="Is this a tshirt or sweatshirt? Answer with one word.",
        color_prompt="What is the color of this garment? Choose one: black, brown, nude, white, pink.",
        detection_prompt="detect tshirt sweatshirt",
    ),
    "descriptive": PromptConfig(
        strategy="descriptive",
        type_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What type of garment is being ironed, a tshirt or sweatshirt? Answer with one word.",
        color_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What is the color of the garment being ironed? Choose one: black, brown, nude, white, pink.",
        detection_prompt="detect the garment being ironed on the table",
    ),
    "descriptive_cat": PromptConfig(
        strategy="descriptive_cat",
        type_prompt="A studio product photo of one upper-body garment, worn by a model or shown on a plain background. The model's hand may partially cover the garment. What type of garment is shown, a tshirt or sweatshirt? Answer with one word.",
        color_prompt="A studio product photo of one upper-body garment, worn by a model or shown on a plain background. The model's hand may partially cover the garment. What is the color of the garment shown? Choose one: black, brown, nude, white, pink.",
        detection_prompt="detect the garment in the product photo",
    ),
}

PALIGEMMA_DEFAULT = "descriptive"

# LLaVA-1.6 wraps these in the Mistral chat template inside src/models/llava.py,
# so only the payload is written here.

LLAVA_PROMPTS: dict[str, PromptConfig] = {
    "label_only": PromptConfig(
        strategy="label_only",
        type_prompt="tshirt or sweatshirt?",
        color_prompt="black, brown, nude, white or pink?",
        detection_prompt="Provide bounding box of the garment as [x_min, y_min, x_max, y_max].",
    ),
    "question": PromptConfig(
        strategy="question",
        type_prompt="What type of garment is this?",
        color_prompt="What color is the garment?",
        detection_prompt="Where is the garment? Give bounding box coordinates.",
    ),
    "constrained": PromptConfig(
        strategy="constrained",
        type_prompt="Is this a tshirt or sweatshirt? Reply with one word only.",
        color_prompt="What color is this piece of clothing? Choose from: black, brown, nude, white, pink. Reply with one word only.",
        detection_prompt="Detect the garment. Return bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates.",
    ),
    "descriptive": PromptConfig(
        strategy="descriptive",
        type_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What type of garment is being ironed, a tshirt or sweatshirt? Answer with exactly one word.",
        color_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What is the color of the garment being ironed? Choose exactly one from: black, brown, nude, white, pink.",
        detection_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. Detect the garment being processed and return its bounding box as [x_min, y_min, x_max, y_max] in absolute pixel coordinates. Image size is 1280x720.",
    ),
    "descriptive_cat": PromptConfig(
        strategy="descriptive_cat",
        type_prompt="This is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. What type of garment is shown, a tshirt or sweatshirt? Answer with exactly one word.",
        color_prompt="This is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. What is the color of the garment shown? Choose exactly one from: black, brown, nude, white, pink.",
        detection_prompt="This is a studio product photograph of a single upper-body garment. Detect the garment and return its bounding box as [x_min, y_min, x_max, y_max] in absolute pixel coordinates. Image size is 1080x1440.",
    ),
}

LLAVA_DEFAULT = "descriptive"

INTERNVL_PROMPTS: dict[str, PromptConfig] = {
    "label_only": PromptConfig(
        strategy="label_only",
        type_prompt="<image>\ntshirt or sweatshirt?",
        color_prompt="<image>\nblack, brown, nude, white or pink?",
        detection_prompt="<image>\nProvide bounding box of the garment as [x_min, y_min, x_max, y_max].",
    ),
    "question": PromptConfig(
        strategy="question",
        type_prompt="<image>\nWhat type of garment is this?",
        color_prompt="<image>\nWhat color is the garment?",
        detection_prompt="<image>\nWhere is the garment? Give bounding box coordinates.",
    ),
    "constrained": PromptConfig(
        strategy="constrained",
        type_prompt="<image>\nIs this a tshirt or sweatshirt? Reply with one word only.",
        color_prompt="<image>\nWhat color is this piece of clothing? Choose from: black, brown, nude, white, pink. Reply with one word only.",
        detection_prompt="<image>\nDetect the garment. Return bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates.",
    ),
    "descriptive": PromptConfig(
        strategy="descriptive",
        type_prompt="<image>\nA worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What type of garment is being ironed, a tshirt or sweatshirt? Answer with exactly one word.",
        color_prompt="<image>\nA worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What is the color of the garment being ironed? Choose exactly one from: black, brown, nude, white, pink.",
        detection_prompt="<image>\nA worker is performing wet-heat treatment on a garment on an industrial ironing table. Detect the garment being processed and return its bounding box as [x_min, y_min, x_max, y_max] in absolute pixel coordinates. Image size is 1280x720.",
        multitask_prompt="<image>\nA worker is performing wet-heat treatment (ironing/pressing) on a garment on an industrial table. The garment may be partially covered by the iron or the worker's hands. Perform the following tasks:\n1. Identify the garment type (tshirt or sweatshirt)\n2. Identify the garment color (black, brown, nude, white, or pink)\n3. Provide the bounding box of the garment as [x_min, y_min, x_max, y_max] in pixel coordinates\nRespond in this exact format:\nType: <type>\nColor: <color>\nBox: [x_min, y_min, x_max, y_max]",
    ),
    "descriptive_cat": PromptConfig(
        strategy="descriptive_cat",
        type_prompt="<image>\nThis is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. What type of garment is shown, a tshirt or sweatshirt? Answer with exactly one word.",
        color_prompt="<image>\nThis is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. What is the color of the garment shown? Choose exactly one from: black, brown, nude, white, pink.",
        detection_prompt="<image>\nThis is a studio product photograph of a single upper-body garment. Detect the garment and return its bounding box as [x_min, y_min, x_max, y_max] in absolute pixel coordinates. Image size is 1080x1440.",
        multitask_prompt="<image>\nThis is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. Perform the following tasks:\n1. Identify the garment type (tshirt or sweatshirt)\n2. Identify the garment color (black, brown, nude, white, or pink)\n3. Provide the bounding box of the garment as [x_min, y_min, x_max, y_max] in pixel coordinates\nRespond in this exact format:\nType: <type>\nColor: <color>\nBox: [x_min, y_min, x_max, y_max]",
    ),
}

INTERNVL_DEFAULT = "descriptive"

MINICPM_PROMPTS: dict[str, PromptConfig] = {
    "label_only": PromptConfig(
        strategy="label_only",
        type_prompt="tshirt or sweatshirt?",
        color_prompt="black, brown, nude, white or pink?",
        detection_prompt="Provide bounding box of the garment as [x_min, y_min, x_max, y_max].",
    ),
    "question": PromptConfig(
        strategy="question",
        type_prompt="What type of garment is this?",
        color_prompt="What color is the garment?",
        detection_prompt="Where is the garment? Give bounding box coordinates.",
    ),
    "constrained": PromptConfig(
        strategy="constrained",
        type_prompt="Is this a tshirt or sweatshirt? Reply with one word only.",
        color_prompt="What color is this piece of clothing? Choose from: black, brown, nude, white, pink. Reply with one word only.",
        detection_prompt="Detect the garment. Return bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates.",
    ),
    "descriptive": PromptConfig(
        strategy="descriptive",
        type_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What type of garment is being ironed, a tshirt or sweatshirt? Answer with exactly one word.",
        color_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What is the color of the garment being ironed? Choose exactly one from: black, brown, nude, white, pink.",
        detection_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. Detect the garment being processed and return its bounding box as [x_min, y_min, x_max, y_max] in absolute pixel coordinates. Image size is 1280x720.",
        multitask_prompt="A worker is performing wet-heat treatment (ironing/pressing) on a garment on an industrial table. The garment may be partially covered by the iron or the worker's hands. Perform the following tasks:\n1. Identify the garment type (tshirt or sweatshirt)\n2. Identify the garment color (black, brown, nude, white, or pink)\n3. Provide the bounding box of the garment as [x_min, y_min, x_max, y_max] in pixel coordinates\nRespond in this exact format:\nType: <type>\nColor: <color>\nBox: [x_min, y_min, x_max, y_max]",
    ),
    "descriptive_cat": PromptConfig(
        strategy="descriptive_cat",
        type_prompt="This is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. What type of garment is shown, a tshirt or sweatshirt? Answer with exactly one word.",
        color_prompt="This is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. What is the color of the garment shown? Choose exactly one from: black, brown, nude, white, pink.",
        detection_prompt="This is a studio product photograph of a single upper-body garment. Detect the garment and return its bounding box as [x_min, y_min, x_max, y_max] in absolute pixel coordinates. Image size is 1080x1440.",
        multitask_prompt="This is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. Perform the following tasks:\n1. Identify the garment type (tshirt or sweatshirt)\n2. Identify the garment color (black, brown, nude, white, or pink)\n3. Provide the bounding box of the garment as [x_min, y_min, x_max, y_max] in pixel coordinates\nRespond in this exact format:\nType: <type>\nColor: <color>\nBox: [x_min, y_min, x_max, y_max]",
    ),
}

MINICPM_DEFAULT = "descriptive"

QWEN_PROMPTS: dict[str, PromptConfig] = {
    "label_only": PromptConfig(
        strategy="label_only",
        type_prompt="tshirt or sweatshirt?",
        color_prompt="black, brown, nude, white or pink?",
        detection_prompt='Locate the garment. Return coordinates as {"bbox_2d": [x1, y1, x2, y2]}.',
    ),
    "question": PromptConfig(
        strategy="question",
        type_prompt="What type of garment is this?",
        color_prompt="What color is the garment?",
        detection_prompt='Where is the garment? Return as {"bbox_2d": [x1, y1, x2, y2]}.',
    ),
    "constrained": PromptConfig(
        strategy="constrained",
        type_prompt="Is this a tshirt or sweatshirt? Reply with one word only.",
        color_prompt="What color is this garment? Choose one: black, brown, nude, white, pink. Reply with one word only.",
        detection_prompt='Detect the garment on the table. Return bounding box as {"bbox_2d": [x1, y1, x2, y2]}.',
    ),
    "descriptive": PromptConfig(
        strategy="descriptive",
        type_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What type of garment is being ironed, a tshirt or sweatshirt? Answer with exactly one word.",
        color_prompt="A worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker's hands. What is the color of the garment? Choose exactly one: black, brown, nude, white, pink.",
        detection_prompt='A worker is performing wet-heat treatment on a garment on an industrial ironing table. Locate the garment and return its bounding box as {"bbox_2d": [x1, y1, x2, y2]}.',
        multitask_prompt='A worker is performing wet-heat treatment on a garment on an industrial ironing table. The garment may be partially covered by the iron or the worker\'s hands. Perform the following tasks:\n1. Identify the garment type (tshirt or sweatshirt)\n2. Identify the garment color (black, brown, nude, white, or pink)\n3. Locate the garment bounding box\nRespond in this exact format:\nType: <type>\nColor: <color>\nBox: {"bbox_2d": [x1, y1, x2, y2]}',
    ),
    "descriptive_cat": PromptConfig(
        strategy="descriptive_cat",
        type_prompt="This is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. What type of garment is shown, a tshirt or sweatshirt? Answer with exactly one word.",
        color_prompt="This is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model's arm or hand may partially cover part of the garment. What is the color of the garment? Choose exactly one: black, brown, nude, white, pink.",
        detection_prompt='This is a studio product photograph of a single upper-body garment. Locate the garment and return its bounding box as {"bbox_2d": [x1, y1, x2, y2]}.',
        multitask_prompt='This is a studio product photograph of a single upper-body garment. The garment is either worn by a model or shown on its own against a plain background. The model\'s arm or hand may partially cover part of the garment. Perform the following tasks:\n1. Identify the garment type (tshirt or sweatshirt)\n2. Identify the garment color (black, brown, nude, white, or pink)\n3. Locate the garment bounding box\nRespond in this exact format:\nType: <type>\nColor: <color>\nBox: {"bbox_2d": [x1, y1, x2, y2]}',
    ),
}

QWEN_DEFAULT = "descriptive"

PROMPT_REGISTRY: dict[str, dict[str, PromptConfig]] = {
    "VILT-B32-FINETUNED-VQA":      VILT_PROMPTS,
    "BLIP-VQA-CAPFILT-LARGE":      BLIP_PROMPTS,
    "FLORENCE-2-LARGE":            FLORENCE_PROMPTS,
    "CLIP-VIT-L14":                CLIP_PROMPTS,
    "PALIGEMMA":                   PALIGEMMA_PROMPTS,
    "LLAVA-1.6-MISTRAL-7B":        LLAVA_PROMPTS,
    "INTERNVL2-8B":                INTERNVL_PROMPTS,
    "MINICPM-LLAMA3-V-2.5":        MINICPM_PROMPTS,
    "QWEN2.5-VL-7B": QWEN_PROMPTS,
    "QWEN3-VL-8B": QWEN_PROMPTS
}

DEFAULT_STRATEGIES: dict[str, str] = {
    "VILT-B32-FINETUNED-VQA":      VILT_DEFAULT,
    "BLIP-VQA-CAPFILT-LARGE":      BLIP_DEFAULT,
    "FLORENCE-2-LARGE":            FLORENCE_DEFAULT,
    "CLIP-VIT-L14":                CLIP_DEFAULT,
    "PALIGEMMA":                   PALIGEMMA_DEFAULT,
    "LLAVA-1.6-MISTRAL-7B":        LLAVA_DEFAULT,
    "INTERNVL2-8B":                INTERNVL_DEFAULT,
    "MINICPM-LLAMA3-V-2.5":        MINICPM_DEFAULT,
    "QWEN2.5-VL-7B": QWEN_DEFAULT,
    "QWEN3-VL-8B": QWEN_DEFAULT
}


def get_prompt(model_name: str, strategy: str | None = None) -> PromptConfig:
    """
    Retrieve a PromptConfig for a given model and strategy.

    If strategy is None, returns the default strategy for the model.
    Raises KeyError if model_name or strategy is not registered.
    """
    if model_name not in PROMPT_REGISTRY:
        raise KeyError(
            f"Model '{model_name}' not in PROMPT_REGISTRY. "
            f"Available: {list(PROMPT_REGISTRY.keys())}"
        )

    strategy = strategy or DEFAULT_STRATEGIES[model_name]
    strategies = PROMPT_REGISTRY[model_name]

    if strategy not in strategies:
        raise KeyError(
            f"Strategy '{strategy}' not available for '{model_name}'. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy]


def get_all_strategies(model_name: str) -> list[PromptConfig]:
    """Return every PromptConfig registered for a model."""
    if model_name not in PROMPT_REGISTRY:
        raise KeyError(f"Model '{model_name}' not in PROMPT_REGISTRY.")
    return list(PROMPT_REGISTRY[model_name].values())