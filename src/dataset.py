import json
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
from collections import defaultdict, Counter

VALID_TYPES = {"sweatshirt", "tshirt"}
VALID_COLORS = {"black", "brown", "nude", "white", "pink"}
TYPE_TO_IDX = {"tshirt": 0, "sweatshirt": 1}
COLOR_TO_IDX = {"black": 0, "brown": 1, "nude": 2, "white": 3, "pink": 4}
IDX_TO_TYPE = {v: k for k, v in TYPE_TO_IDX.items()}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

logger = logging.getLogger(__name__)


def extract_segment_id(file_name: str) -> str:
    """
    Extract acquisition segment ID from file name.
    Segment ID is the second underscore-separated token (10-digit unix timestamp).
    Example: '07M40S_1678082860_42_jpg.rf.xxx.jpg' → '1678082860'
    Files without a valid segment pattern get 'unknown'.
    """
    parts = file_name.split('_')
    if len(parts) > 2 and parts[1].isdigit():
        return parts[1]
    return 'unknown'


@dataclass
class GarmentRecord:
    """Single dataset item with all labels and metadata."""
    image_id: int
    file_name: str
    image_path: Path
    # Classification labels
    garment_type: str       # "tshirt" | "sweatshirt"
    color: str              # "black" | "brown" | "nude" | "white" | "pink"
    type_idx: int
    color_idx: int
    # Detection label — stored as [x_min, y_min, x_max, y_max] (xyxy)
    bbox_xyxy: list
    bbox_xywh: list
    # Image dimensions
    width: int = IMAGE_WIDTH
    height: int = IMAGE_HEIGHT
    # Acquisition segment ID for grouped splitting
    segment_id: str = 'unknown'


def _parse_user_tags(user_tags: list) -> Optional[tuple]:
    """
    Parse user_tags list into (garment_type, color).
    Returns None if tags are missing or invalid.
    """
    garment_type = None
    color = None
    for tag in user_tags:
        if "-" not in tag:
            continue
        prefix, value = tag.split("-", 1)
        value = value.lower().strip()
        if prefix == "type" and value in VALID_TYPES:
            garment_type = value
        elif prefix == "color" and value in VALID_COLORS:
            color = value
        elif value in VALID_TYPES:
            garment_type = value
        elif value in VALID_COLORS:
            color = value
    if garment_type is None or color is None:
        return None
    return garment_type, color


def _coco_bbox_to_xyxy(bbox: list) -> list:
    """Convert COCO [x, y, w, h] to [x_min, y_min, x_max, y_max]."""
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


class GarmentDataset(Dataset):
    """
    PyTorch Dataset for the garment manufacturing dataset.

    IMPORTANT: Splitting is performed at the ACQUISITION SEGMENT level.
    All frames from one segment go to the same split (train/val/test).
    This prevents data leakage from near-duplicate frames within a segment.

    Args:
        data_dir:   Path to directory containing images and annotation file.
        split:      "train" | "val" | "test" | "all"
        val_ratio:  Fraction of segments for validation (default 0.15)
        test_ratio: Fraction of segments for test (default 0.15)
        seed:       Random seed for reproducible splits
        transform:  Optional callable applied to PIL Image
    """

    def __init__(
        self,
        data_dir,
        split: str = "all",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        transform=None,
    ):
        assert split in {"train", "val", "test", "all"}, \
            f"split must be one of train/val/test/all, got '{split}'"
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        ann_path = self.data_dir / "_annotations.coco.json"
        assert ann_path.exists(), f"Annotation file not found: {ann_path}"

        self.records = self._load_records(ann_path)
        self.records = self._apply_split(
            self.records, split, val_ratio, test_ratio, seed
        )
        logger.info(
            f"GarmentDataset | split={split} | "
            f"{len(self.records)} records loaded from {data_dir}"
        )

    def _load_records(self, ann_path: Path) -> list:
        with open(ann_path, "r") as f:
            data = json.load(f)

        images = data["images"]
        annotations = data["annotations"]

        ann_by_image_id = {}
        for ann in annotations:
            ann_by_image_id[ann["image_id"]] = ann

        records = []
        skipped_no_extra = 0
        skipped_invalid_tags = 0
        skipped_no_annotation = 0
        skipped_missing_file = 0

        for img in images:
            image_id = img["id"]
            if "extra" not in img:
                skipped_no_extra += 1
                continue

            user_tags = img["extra"].get("user_tags", [])
            parsed = _parse_user_tags(user_tags)
            if parsed is None:
                skipped_invalid_tags += 1
                continue

            garment_type, color = parsed

            ann = ann_by_image_id.get(image_id)
            if ann is None:
                skipped_no_annotation += 1
                continue

            image_path = self.data_dir / img["file_name"]
            if not image_path.exists():
                skipped_missing_file += 1
                continue

            bbox_xywh = [int(v) for v in ann["bbox"]]
            bbox_xyxy = _coco_bbox_to_xyxy(ann["bbox"])
            segment_id = extract_segment_id(img["file_name"])

            records.append(GarmentRecord(
                image_id=image_id,
                file_name=img["file_name"],
                image_path=image_path,
                garment_type=garment_type,
                color=color,
                type_idx=TYPE_TO_IDX[garment_type],
                color_idx=COLOR_TO_IDX[color],
                bbox_xyxy=bbox_xyxy,
                bbox_xywh=bbox_xywh,
                width=img["width"],
                height=img["height"],
                segment_id=segment_id,
            ))

        total_skipped = (
            skipped_no_extra + skipped_invalid_tags +
            skipped_no_annotation + skipped_missing_file
        )
        logger.info(
            f"Loaded {len(records)} records. Skipped {total_skipped} "
            f"(no_extra={skipped_no_extra}, invalid_tags={skipped_invalid_tags}, "
            f"no_annotation={skipped_no_annotation}, missing_file={skipped_missing_file})"
        )
        return records

    def _apply_split(self, records: list, split: str,
                     val_ratio: float, test_ratio: float, seed: int) -> list:
        """
        Group-aware stratified split.
        All frames from one acquisition segment go to the same split.
        Stratification is performed jointly by garment type and color at the
        segment level: frames within a segment are homogeneous in type for 98%
        of segments and in color for 94%, and stratifying by type alone leaves
        the color distribution across folds badly unbalanced.
        """
        if split == "all":
            return records

        rng = np.random.default_rng(seed)

        # Group record indices by segment; label each segment by its first frame
        seg_to_indices = defaultdict(list)
        seg_to_stratum = {}
        for i, rec in enumerate(records):
            seg_to_indices[rec.segment_id].append(i)
            if rec.segment_id not in seg_to_stratum:
                seg_to_stratum[rec.segment_id] = (rec.garment_type, rec.color)

        # Group segments by (type, color) stratum
        stratum_to_segments = defaultdict(list)
        for seg, stratum in seg_to_stratum.items():
            stratum_to_segments[stratum].append(seg)

        # Split segments within each stratum
        train_segs, val_segs, test_segs = [], [], []

        for stratum in sorted(stratum_to_segments.keys()):
            segs = stratum_to_segments[stratum]
            shuffled = rng.permutation(np.array(segs)).tolist()
            n = len(shuffled)

            n_test = max(1, int(round(n * test_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            # Ensure at least one segment remains for training
            if n - n_test - n_val < 1:
                n_test = max(1, n // 3)
                n_val = max(1, n // 3)

            train_segs.extend(shuffled[:n - n_val - n_test])
            val_segs.extend(shuffled[n - n_val - n_test:n - n_test])
            test_segs.extend(shuffled[n - n_test:])

        split_map = {
            "train": train_segs,
            "val": val_segs,
            "test": test_segs,
        }

        selected_segs = split_map[split]
        selected_indices = sorted(
            idx for seg in selected_segs for idx in seg_to_indices[seg]
        )
        selected = [records[i] for i in selected_indices]

        # Log split composition
        seg_count = len(selected_segs)
        frame_count = len(selected)
        type_dist = Counter(r.garment_type for r in selected)
        color_dist = Counter(r.color for r in selected)
        logger.info(
            f"Split '{split}': {seg_count} segments, {frame_count} frames | "
            f"types={dict(type_dist)} | colors={dict(color_dist)}"
        )

        return selected

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "image_id": record.image_id,
            "file_name": record.file_name,
            "garment_type": record.garment_type,
            "color": record.color,
            "type_idx": record.type_idx,
            "color_idx": record.color_idx,
            "bbox_xyxy": record.bbox_xyxy,
            "bbox_xywh": record.bbox_xywh,
            "segment_id": record.segment_id,
        }

    def get_record(self, idx: int) -> GarmentRecord:
        """Return raw GarmentRecord without loading image."""
        return self.records[idx]

    def get_by_image_id(self, image_id: int) -> Optional[GarmentRecord]:
        """Find record by image_id. O(n) — use only for debugging."""
        for r in self.records:
            if r.image_id == image_id:
                return r
        return None

    def class_distribution(self) -> dict:
        """Return count breakdown by type and color."""
        types = Counter(r.garment_type for r in self.records)
        colors = Counter(r.color for r in self.records)
        return {"types": dict(types), "colors": dict(colors)}

    def segment_summary(self) -> dict:
        """Return summary of segments in this split."""
        segs = set(r.segment_id for r in self.records)
        return {
            "n_segments": len(segs),
            "n_frames": len(self.records),
            "avg_frames_per_segment": len(self.records) / max(len(segs), 1),
        }

    def __repr__(self) -> str:
        dist = self.class_distribution()
        seg_info = self.segment_summary()
        return (
            f"GarmentDataset(split='{self.split}', n={len(self.records)}, "
            f"segments={seg_info['n_segments']}, "
            f"types={dist['types']}, colors={dist['colors']})"
        )