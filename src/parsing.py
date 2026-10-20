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

# Models that write coordinates as free text. Their boxes are parsed from the
# raw response when metrics are computed, like the labels.
FREE_TEXT_BBOX_MODELS = ("LLAVA-1.6-MISTRAL-7B", "INTERNVL2-8B", "MINICPM-LLAMA3-V-2.5")

# Coordinate unit per model and prompt. Values in [0, 1] are always fractions
# of the frame. Larger values are absolute pixels unless the configuration is
# listed here, in which case they lie on a 0-1000 grid. The table is produced
# by src/calibrate_bbox.py, which uses no annotation: on a 1280x720 frame a
# vertical coordinate above 720 cannot be a pixel and a coordinate above 1000
# cannot lie on the grid, so the reading that leaves more boxes inside the
# frame is taken.
BBOX_GRID_1000 = {
    ("INTERNVL2-8B", "label_only"),
    ("INTERNVL2-8B", "constrained"),
    ("INTERNVL2-8B", "question"),
    ("INTERNVL2-8B", "descriptive_multitask"),
    ("MINICPM-LLAMA3-V-2.5", "label_only"),
    ("MINICPM-LLAMA3-V-2.5", "question"),
    ("MINICPM-LLAMA3-V-2.5", "constrained"),
    ("MINICPM-LLAMA3-V-2.5", "descriptive"),
    ("MINICPM-LLAMA3-V-2.5", "descriptive_multitask"),
}

_NUM = r"-?\d+(?:\.\d+)?"
_FOUR = r"(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")"
_BOX_TAG = re.compile(r"<box>\s*(" + _NUM + r")[\s,]+(" + _NUM + r")[\s,]+(" + _NUM + r")[\s,]+(" + _NUM + r")\s*</box>")
_BOX_GROUP = re.compile(r"[\[\(]\s*" + _FOUR + r"\s*[\]\)]")
_BOX_OPEN = re.compile(r"[\[\(]\s*" + _FOUR + r"\.?\s*$")
_BOX_PAIR = re.compile(r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)")


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


def extract_bbox(raw):
    """Four coordinates from a free-text response, or None.

    Tried in order: a <box> tag; the last bracketed group of four numbers; a
    bracketed group of four numbers left unclosed at the end of the response;
    the last two (x, y) corner pairs; a response holding exactly four numbers.
    """
    if not raw:
        return None
    raw = raw.strip()
    found = _BOX_TAG.findall(raw) or _BOX_GROUP.findall(raw)
    if found:
        return [float(v) for v in found[-1]]
    match = _BOX_OPEN.search(raw)
    if match:
        return [float(v) for v in match.groups()]
    pairs = _BOX_PAIR.findall(raw)
    if len(pairs) >= 2:
        return [float(v) for v in pairs[-2] + pairs[-1]]
    numbers = re.findall(_NUM, raw)
    return [float(v) for v in numbers] if len(numbers) == 4 else None


def parse_bbox(raw, grid_1000=False, image_size=(1280, 720)):
    """Box in absolute pixels, or None if the response holds no valid box."""
    values = extract_bbox(raw)
    if values is None:
        return None
    w, h = image_size
    if all(0.0 <= v <= 1.0 for v in values):
        sx, sy = w, h
    elif grid_1000:
        sx, sy = w / 1000, h / 1000
    else:
        sx, sy = 1, 1
    box = [values[0] * sx, values[1] * sy, values[2] * sx, values[3] * sy]
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    if box[2] > w * 1.1 or box[3] > h * 1.1:
        return None
    return box


# Models that return the box in a bbox_2d JSON field. Qwen2.5-VL writes pixels
# of the frame as the processor resized it, Qwen3-VL a 0-1000 grid.
JSON_BBOX_MODELS = ("QWEN2.5-VL-7B", "QWEN3-VL-8B")
JSON_BBOX_GRID_1000 = ("QWEN3-VL-8B",)
_BBOX_FIELD = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]+)\]')


def extract_json_bbox(raw):
    """Four coordinates from a response that carries a bbox_2d field, or None.

    The field is read when it is complete. Otherwise the response is treated
    as free text, with the field name removed so that its digit is not taken
    for a coordinate.
    """
    if not raw:
        return None
    match = _BBOX_FIELD.search(raw)
    if match:
        found = re.findall(_NUM, match.group(1))
        if len(found) >= 4:
            return [float(v) for v in found[:4]]
    return extract_bbox(raw.replace("bbox_2d", "bbox"))


def resized_frame(image_size=(1280, 720), patch=28, min_pixels=256 * 28 * 28,
                  max_pixels=1280 * 28 * 28):
    """The size to which the Qwen2.5-VL processor resizes a frame."""
    width, height = image_size
    total = width * height
    scale = ((min_pixels / total) ** 0.5 if total < min_pixels
             else (max_pixels / total) ** 0.5 if total > max_pixels else 1.0)
    return (max(patch, round(width * scale / patch) * patch),
            max(patch, round(height * scale / patch) * patch))


def parse_json_bbox(raw, grid_1000, image_size=(1280, 720)):
    """Box in absolute pixels from a bbox_2d response, or None."""
    values = extract_json_bbox(raw)
    if values is None:
        return None
    w, h = image_size
    if grid_1000:
        sx, sy = w / 1000, h / 1000
    else:
        rw, rh = resized_frame(image_size)
        sx, sy = w / rw, h / rh
    box = [int(values[0] * sx), int(values[1] * sy), int(values[2] * sx), int(values[3] * sy)]
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    if box[2] > w * 1.1 or box[3] > h * 1.1:
        return None
    return box


def bbox_text(record):
    """The part of a response that carries the box."""
    if record.get("raw_detection"):
        return record["raw_detection"]
    text = record.get("raw_multitask") or ""
    match = re.search(r"box\s*:?(.*)", text, re.I | re.S)
    return match.group(1) if match else text


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
