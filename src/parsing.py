import re

TYPE_LABELS = ("tshirt", "sweatshirt")
COLOR_LABELS = ("black", "brown", "nude", "white", "pink")

# Table 10 of the paper. Longer patterns first.
TYPE_RULES = (
    (re.compile(r"\bt[\s\-_]shirt\b", re.I), "tshirt"),
    (re.compile(r"\bsweat[\s\-_]shirt\b", re.I), "sweatshirt"),
)
COLOR_RULES = (
    (re.compile(r"\blight\s+beige\b", re.I), "nude"),
    (re.compile(r"\bbeige\b", re.I), "nude"),
    (re.compile(r"\btan\b", re.I), "nude"),
    (re.compile(r"\bivory\b", re.I), "white"),
    (re.compile(r"\bcream\b", re.I), "white"),
)

CONTRASTIVE_MODELS = ("CLIP-VIT-L14",)


def parse_label(raw, labels, rules=()):
    if not raw:
        return None
    text = raw
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    text = text.lower().strip().rstrip(".,!?")
    if text in labels:
        return text
    found = [l for l in labels if re.search(r"\b" + re.escape(l) + r"\b", text)]
    return found[0] if len(found) == 1 else None


def parse_type(raw, normalize=True):
    return parse_label(raw, TYPE_LABELS, TYPE_RULES if normalize else ())


def parse_color(raw, normalize=True):
    return parse_label(raw, COLOR_LABELS, COLOR_RULES if normalize else ())


def compute_iou(a, b):
    if not a or not b:
        return 0.0
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def is_contrastive(model_name):
    return any(m in (model_name or "").upper() for m in CONTRASTIVE_MODELS)
