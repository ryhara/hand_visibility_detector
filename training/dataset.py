"""Hand visibility training datasets (COCO-WholeBody and HInt).

COCO-WholeBody
--------------
Each sample is a single hand crop produced from one annotation's
``lefthand_box`` or ``righthand_box``. Visibility targets are derived from
the 21-point hand keypoint visibility flags ``v in {0, 1, 2}``::

    v == 0  -> not labeled  (mask=0, target ignored)
    v == 1  -> labeled, occluded -> target=0
    v == 2  -> labeled, visible  -> target=1

HInt
----
HInt (``HInt_annotation_partial``) lives under a flat
``<subset>/<base>.{json,jpg}`` layout. Each JSON carries 21 ``keypoints``,
an xyxy ``bbox``, and per-keypoint ``existence`` / ``occlusion`` flags::

    existence==1 & occlusion==0  -> visible  -> target=1, mask=1
    existence==1 & occlusion==1  -> occluded -> target=0, mask=1
    existence==0 (out-of-frame)  -> occluded -> target=0, mask=1

Unlike COCO every HInt keypoint is labeled, so the mask is always 1.

Left hands are mirrored horizontally in both datasets so the model only
ever sees a canonical right hand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hand_visibility_detector.transforms import (
    crop_square,
    expand_square_bbox,
    to_model_tensor,
)

from .augmentation import augment_train, jitter_bbox_center


NUM_HAND_KEYPOINTS = 21


# ---------------------------------------------------------------------------
# COCO-WholeBody
# ---------------------------------------------------------------------------


@dataclass
class HandSample:
    image_id: int
    file_name: str
    bbox_xywh: tuple[float, float, float, float]
    kpts: np.ndarray
    is_right: bool


def _iter_hand_samples(
    annotations: list[dict[str, Any]], min_side: float
) -> list[HandSample]:
    out: list[HandSample] = []
    for ann in annotations:
        for side, box_key, kpt_key, valid_key in [
            (True, "righthand_box", "righthand_kpts", "righthand_valid"),
            (False, "lefthand_box", "lefthand_kpts", "lefthand_valid"),
        ]:
            if not ann.get(valid_key, False):
                continue
            box = ann.get(box_key)
            kpts = ann.get(kpt_key)
            if box is None or kpts is None:
                continue
            x, y, w, h = box
            if w < min_side or h < min_side:
                continue
            kpts_arr = np.asarray(kpts, dtype=np.float32).reshape(-1, 3)
            if kpts_arr.shape[0] != NUM_HAND_KEYPOINTS:
                continue
            if (kpts_arr[:, 2] > 0).sum() == 0:
                continue
            out.append(
                HandSample(
                    image_id=ann["image_id"],
                    file_name="",
                    bbox_xywh=(float(x), float(y), float(w), float(h)),
                    kpts=kpts_arr,
                    is_right=side,
                )
            )
    return out


class COCOWholeBodyHandDataset(Dataset):
    def __init__(
        self,
        ann_path: str,
        img_dir: str,
        crop_size: int = 256,
        bbox_expand: float = 1.25,
        min_bbox_side: float = 20.0,
        train: bool = True,
        augment_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        with open(ann_path, "r") as f:
            data = json.load(f)
        id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
        samples = _iter_hand_samples(data["annotations"], min_side=min_bbox_side)
        for s in samples:
            fn = id_to_file.get(s.image_id)
            if fn is None:
                continue
            s.file_name = fn
        self.samples = [s for s in samples if s.file_name]
        self.img_dir = img_dir
        self.crop_size = crop_size
        self.bbox_expand = bbox_expand
        self.train = train
        self.augment_cfg = augment_cfg or {}

    def __len__(self) -> int:
        return len(self.samples)

    def _make_visibility(self, kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = kpts[:, 2].astype(np.int32)
        target = (v == 2).astype(np.float32)
        mask = ((v == 1) | (v == 2)).astype(np.float32)
        return target, mask

    def positive_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(pos_count, total_count)`` per keypoint over the dataset
        (used to compute ``pos_weight``)."""
        pos = np.zeros(NUM_HAND_KEYPOINTS, dtype=np.float64)
        tot = np.zeros(NUM_HAND_KEYPOINTS, dtype=np.float64)
        for s in self.samples:
            t, m = self._make_visibility(s.kpts)
            pos += t * m
            tot += m
        return pos, tot

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        path = os.path.join(self.img_dir, s.file_name)
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            image_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, side = expand_square_bbox(np.asarray(s.bbox_xywh), self.bbox_expand)
        if self.train:
            cx, cy, side = jitter_bbox_center(cx, cy, side)
        patch, _ = crop_square(image_rgb, cx, cy, side, self.crop_size)

        if not s.is_right:
            patch = patch[:, ::-1, :].copy()

        if self.train:
            patch = augment_train(patch, self.augment_cfg)

        target, mask = self._make_visibility(s.kpts)
        return {
            "image": to_model_tensor(patch),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "is_right": torch.tensor(1 if s.is_right else 0, dtype=torch.long),
            "image_id": torch.tensor(s.image_id, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# HInt
# ---------------------------------------------------------------------------


@dataclass
class HIntSample:
    base: str
    subset: str
    image_path: str
    bbox_xyxy: tuple[float, float, float, float]
    kpts: np.ndarray
    existence: np.ndarray
    occlusion: np.ndarray
    is_right: bool


def _infer_hand_side_from_base(base: str) -> bool | None:
    tail = base.rsplit("_", 1)[-1].lower()
    if tail == "r":
        return True
    if tail == "l":
        return False
    return None


def _load_hint_json(path: str) -> dict | None:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]
    if not isinstance(data, dict):
        return None
    return data


def _parse_hint_bbox(raw_bbox: Any) -> tuple[float, float, float, float] | None:
    if raw_bbox is None:
        return None
    if (
        isinstance(raw_bbox, (list, tuple))
        and len(raw_bbox) > 0
        and isinstance(raw_bbox[0], (list, tuple))
    ):
        b = raw_bbox[0]
    else:
        b = raw_bbox
    if len(b) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except (TypeError, ValueError):
        return None
    if not (x1 > x0 and y1 > y0):
        return None
    return x0, y0, x1, y1


def _scan_hint_subset(
    data_root: str, subset: str, min_side: float
) -> list[HIntSample]:
    out: list[HIntSample] = []
    sub_dir = os.path.join(data_root, subset)
    if not os.path.isdir(sub_dir):
        return out
    for fname in sorted(os.listdir(sub_dir)):
        if not fname.endswith(".json"):
            continue
        base = fname[:-5]
        image_path = os.path.join(sub_dir, base + ".jpg")
        if not os.path.isfile(image_path):
            continue
        is_right = _infer_hand_side_from_base(base)
        if is_right is None:
            continue
        ann = _load_hint_json(os.path.join(sub_dir, fname))
        if ann is None:
            continue
        bbox = _parse_hint_bbox(ann.get("bbox"))
        if bbox is None:
            continue
        if (bbox[2] - bbox[0]) < min_side or (bbox[3] - bbox[1]) < min_side:
            continue
        kpts = np.asarray(ann.get("keypoints", []), dtype=np.float32).reshape(-1, 2)
        existence = np.asarray(ann.get("existence", []), dtype=np.float32).reshape(-1)
        occlusion = np.asarray(ann.get("occlusion", []), dtype=np.float32).reshape(-1)
        if kpts.shape[0] != NUM_HAND_KEYPOINTS:
            continue
        if existence.shape[0] != NUM_HAND_KEYPOINTS:
            existence = np.ones(NUM_HAND_KEYPOINTS, dtype=np.float32)
        if occlusion.shape[0] != NUM_HAND_KEYPOINTS:
            occlusion = np.zeros(NUM_HAND_KEYPOINTS, dtype=np.float32)
        out.append(
            HIntSample(
                base=base,
                subset=subset,
                image_path=image_path,
                bbox_xyxy=bbox,
                kpts=kpts,
                existence=existence,
                occlusion=occlusion,
                is_right=is_right,
            )
        )
    return out


class HIntHandDataset(Dataset):
    """HInt hand visibility dataset.

    ``existence == 0`` (out-of-frame) is folded into the occluded class
    (target=0, mask=1). ``mask`` is always 1 for the 21 keypoints.
    """

    def __init__(
        self,
        data_root: str,
        subsets: list[str],
        crop_size: int = 256,
        bbox_expand: float = 1.25,
        min_bbox_side: float = 20.0,
        train: bool = True,
        augment_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.subsets = list(subsets)
        samples: list[HIntSample] = []
        for subset in self.subsets:
            samples.extend(
                _scan_hint_subset(data_root, subset, min_side=min_bbox_side)
            )
        self.samples = samples
        self.crop_size = crop_size
        self.bbox_expand = bbox_expand
        self.train = train
        self.augment_cfg = augment_cfg or {}

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _make_visibility(
        existence: np.ndarray, occlusion: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        target = ((existence > 0.5) & (occlusion < 0.5)).astype(np.float32)
        mask = np.ones(NUM_HAND_KEYPOINTS, dtype=np.float32)
        return target, mask

    def positive_counts(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.zeros(NUM_HAND_KEYPOINTS, dtype=np.float64)
        tot = np.zeros(NUM_HAND_KEYPOINTS, dtype=np.float64)
        for s in self.samples:
            t, m = self._make_visibility(s.existence, s.occlusion)
            pos += t * m
            tot += m
        return pos, tot

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        image_bgr = cv2.imread(s.image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            image_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        x0, y0, x1, y1 = s.bbox_xyxy
        bbox_xywh = np.asarray([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)
        cx, cy, side = expand_square_bbox(bbox_xywh, self.bbox_expand)
        if self.train:
            cx, cy, side = jitter_bbox_center(cx, cy, side)
        patch, _ = crop_square(image_rgb, cx, cy, side, self.crop_size)

        if not s.is_right:
            patch = patch[:, ::-1, :].copy()

        if self.train:
            patch = augment_train(patch, self.augment_cfg)

        target, mask = self._make_visibility(s.existence, s.occlusion)
        image_id = hash(f"{s.subset}/{s.base}") & 0x7FFFFFFF
        return {
            "image": to_model_tensor(patch),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "is_right": torch.tensor(1 if s.is_right else 0, dtype=torch.long),
            "image_id": torch.tensor(image_id, dtype=torch.long),
        }
