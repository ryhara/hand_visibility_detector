"""Visualization utilities for hand visibility detection results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .pipeline import HandResult


# 21-point MANO-style hand skeleton (wrist=0, thumb 1-4, index 5-8, ...).
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]


def vis_color(v: float) -> tuple[int, int, int]:
    """Map visibility 0..1 to (R, G, B) for OpenCV drawing."""
    r = int((1 - v) * 255)
    g = int(v * 255)
    b = 50
    return (r, g, b)


def draw_detections(
    image_rgb: np.ndarray,
    results: list["HandResult"],
    show_bones: bool = True,
    show_conf: bool = True,
) -> np.ndarray:
    """Draw all detected hands on the original image.

    Draws bounding boxes, 2D keypoint skeletons coloured by visibility,
    and optionally the detection confidence.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8 RGB image.
    results : List of :class:`HandResult` from the pipeline.
    show_bones : Whether to draw skeleton bones.
    show_conf : Whether to show bbox confidence text.

    Returns
    -------
    Annotated RGB uint8 image (copy of input).
    """
    canvas = image_rgb.copy()

    for res in results:
        x1, y1, x2, y2 = res.hand_bbox
        side_str = "R" if res.is_right else "L"
        bbox_color = (0, 200, 255)

        cv2.rectangle(
            canvas,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            bbox_color, 2, cv2.LINE_AA,
        )

        label = side_str
        if show_conf:
            label = f"{side_str} {res.bbox_conf:.2f}"
        cv2.putText(
            canvas, label, (int(x1), int(y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2, cv2.LINE_AA,
        )

        kpts = res.keypoints_2d
        vis = res.visibility

        if show_bones and kpts is not None and vis is not None:
            for a, b in HAND_BONES:
                if a >= len(kpts) or b >= len(kpts):
                    continue
                va, vb = vis[a], vis[b]
                color = vis_color(min(va, vb))
                pt1 = (int(kpts[a, 0]), int(kpts[a, 1]))
                pt2 = (int(kpts[b, 0]), int(kpts[b, 1]))
                cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

        if kpts is not None and vis is not None:
            for k in range(len(kpts)):
                v = float(vis[k])
                color = vis_color(v)
                center = (int(kpts[k, 0]), int(kpts[k, 1]))
                cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, center, 4, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
