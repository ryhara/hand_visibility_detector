"""Inference-only crop / transform utilities for hand visibility detection."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def xyxy_to_xywh(bbox_xyxy: list[float] | np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = bbox_xyxy[:4]
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def expand_square_bbox(
    bbox_xywh: np.ndarray, expand: float = 1.25
) -> tuple[float, float, float]:
    """Convert [x, y, w, h] to (cx, cy, side) -- a square bbox expanded by
    ``expand``.  The returned side covers ``max(w, h) * expand``."""
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * expand
    return cx, cy, side


def crop_square(
    image: np.ndarray,
    cx: float,
    cy: float,
    side: float,
    out_size: int,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Crop a square region from *image* (H, W, 3) centred at (*cx*, *cy*)
    with the given *side* length, then resize to (*out_size*, *out_size*).
    Out-of-frame pixels are zero-padded.

    Returns ``(patch, (x0, y0, scale))`` where
    ``scale = out_size / side`` maps source pixels to patch pixels.
    """
    side = max(side, 1.0)
    x0 = cx - side / 2.0
    y0 = cy - side / 2.0
    x1 = x0 + side
    y1 = y0 + side

    H, W = image.shape[:2]
    src_x0 = int(math.floor(max(0, x0)))
    src_y0 = int(math.floor(max(0, y0)))
    src_x1 = int(math.ceil(min(W, x1)))
    src_y1 = int(math.ceil(min(H, y1)))

    side_i = max(int(round(side)), 1)
    canvas = np.zeros((side_i, side_i, 3), dtype=image.dtype)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - int(math.floor(x0))
        dst_y0 = src_y0 - int(math.floor(y0))
        src = image[src_y0:src_y1, src_x0:src_x1]
        copy_h = min(src.shape[0], side_i - dst_y0)
        copy_w = min(src.shape[1], side_i - dst_x0)
        if copy_h > 0 and copy_w > 0:
            canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = (
                src[:copy_h, :copy_w]
            )

    patch = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    scale = out_size / float(side_i)
    return patch, (float(x0), float(y0), float(scale))


def to_model_tensor(patch_rgb_uint8: np.ndarray) -> torch.Tensor:
    """Convert a (H, W, 3) uint8 RGB patch to a (3, H, W) float tensor
    normalised with ImageNet statistics."""
    img = patch_rgb_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()
