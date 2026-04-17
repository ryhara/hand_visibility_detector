"""Train-time image and bbox augmentations for hand visibility detection.

Inference-time crop utilities live in :mod:`hand_visibility_detector.transforms`.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def random_affine(
    patch: np.ndarray,
    rot_deg: float = 30.0,
    scale_range: tuple[float, float] = (0.8, 1.2),
    rng: random.Random | None = None,
) -> np.ndarray:
    rng = rng or random
    h, w = patch.shape[:2]
    angle = rng.uniform(-rot_deg, rot_deg)
    scale = rng.uniform(*scale_range)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(
        patch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def color_jitter(
    patch: np.ndarray,
    brightness: tuple[float, float] = (0.8, 1.2),
    contrast: tuple[float, float] = (0.8, 1.2),
    rng: random.Random | None = None,
) -> np.ndarray:
    rng = rng or random
    img = patch.astype(np.float32)
    b = rng.uniform(*brightness)
    c = rng.uniform(*contrast)
    img = img * b
    mean = img.mean()
    img = (img - mean) * c + mean
    return np.clip(img, 0, 255).astype(np.uint8)


def random_hflip(
    patch: np.ndarray, p: float = 0.5, rng: random.Random | None = None
) -> np.ndarray:
    rng = rng or random
    if rng.random() < p:
        return patch[:, ::-1, :].copy()
    return patch


def hsv_jitter(
    patch: np.ndarray,
    p: float = 1.0,
    h: float = 0.015,
    s: float = 0.7,
    v: float = 0.4,
    rng: random.Random | None = None,
) -> np.ndarray:
    """YOLOv5/Ultralytics-style HSV jitter on an RGB uint8 patch."""
    rng = rng or random
    if rng.random() >= p:
        return patch
    r = np.array(
        [
            rng.uniform(-1.0, 1.0) * h,
            rng.uniform(-1.0, 1.0) * s,
            rng.uniform(-1.0, 1.0) * v,
        ],
        dtype=np.float32,
    )
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    dtype = patch.dtype
    x = np.arange(0, 256, dtype=np.float32)
    lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
    lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
    lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
    hsv_jit = cv2.merge(
        [cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)]
    )
    return cv2.cvtColor(hsv_jit, cv2.COLOR_HSV2RGB)


def gaussian_blur(
    patch: np.ndarray,
    p: float = 0.3,
    kernel_range: tuple[int, int] = (3, 7),
    sigma_range: tuple[float, float] = (0.1, 2.0),
    rng: random.Random | None = None,
) -> np.ndarray:
    rng = rng or random
    if rng.random() >= p:
        return patch
    k0 = max(3, int(kernel_range[0]))
    k1 = max(k0, int(kernel_range[1]))
    if k0 % 2 == 0:
        k0 += 1
    if k1 % 2 == 0:
        k1 += 1
    ksize = rng.randrange(k0, k1 + 2, 2)
    sigma = rng.uniform(float(sigma_range[0]), float(sigma_range[1]))
    return cv2.GaussianBlur(patch, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def random_grayscale(
    patch: np.ndarray, p: float = 0.2, rng: random.Random | None = None
) -> np.ndarray:
    """Convert to grayscale but keep 3 channels (replicate)."""
    rng = rng or random
    if rng.random() >= p:
        return patch
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def jitter_bbox_center(
    cx: float,
    cy: float,
    side: float,
    shift: float = 0.1,
    scale: tuple[float, float] = (0.85, 1.15),
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """RTMPose-style bbox center shift + scale jitter applied before cropping."""
    rng = rng or random
    cx = cx + rng.uniform(-shift, shift) * side
    cy = cy + rng.uniform(-shift, shift) * side
    side = side * rng.uniform(*scale)
    return cx, cy, side


def _is_enabled(cfg: dict | None, key: str) -> bool:
    if not cfg:
        return False
    block = cfg.get(key)
    if not block:
        return False
    return bool(block.get("enabled", False))


def augment_train(
    patch: np.ndarray,
    cfg: dict | None = None,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Image-level augmentation applied after cropping.

    ``cfg`` is a nested dict (see ``configs/*.yaml`` ``data.augment``). Each
    augmentation block has an ``enabled`` flag plus its own hyperparameters.
    Blocks that are absent or ``enabled: false`` are skipped.

    NOTE: bbox jitter must be applied BEFORE cropping (see dataset.py); hflip
    is safe here because the 21-point hand layout is index-wise symmetric.
    """
    if cfg is None:
        return patch

    if _is_enabled(cfg, "color_jitter"):
        c = cfg["color_jitter"]
        patch = color_jitter(
            patch,
            brightness=tuple(c.get("brightness", (0.8, 1.2))),
            contrast=tuple(c.get("contrast", (0.8, 1.2))),
            rng=rng,
        )
    if _is_enabled(cfg, "hsv"):
        c = cfg["hsv"]
        patch = hsv_jitter(
            patch,
            p=float(c.get("p", 1.0)),
            h=float(c.get("h", 0.015)),
            s=float(c.get("s", 0.7)),
            v=float(c.get("v", 0.4)),
            rng=rng,
        )
    if _is_enabled(cfg, "grayscale"):
        c = cfg["grayscale"]
        patch = random_grayscale(patch, p=float(c.get("p", 0.2)), rng=rng)
    if _is_enabled(cfg, "gaussian_blur"):
        c = cfg["gaussian_blur"]
        patch = gaussian_blur(
            patch,
            p=float(c.get("p", 0.3)),
            kernel_range=tuple(c.get("kernel_range", (3, 7))),
            sigma_range=tuple(c.get("sigma_range", (0.1, 2.0))),
            rng=rng,
        )
    if _is_enabled(cfg, "affine"):
        c = cfg["affine"]
        patch = random_affine(
            patch,
            rot_deg=float(c.get("rot_deg", 30.0)),
            scale_range=tuple(c.get("scale_range", (0.8, 1.2))),
            rng=rng,
        )
    if _is_enabled(cfg, "hflip"):
        c = cfg["hflip"]
        patch = random_hflip(patch, p=float(c.get("p", 0.5)), rng=rng)
    return patch
