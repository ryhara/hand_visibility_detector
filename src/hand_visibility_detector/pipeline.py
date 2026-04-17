"""End-to-end hand detection + 3D pose + per-keypoint visibility pipeline."""

from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from skimage.filters import gaussian
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)
from wilor_mini.utils import utils as wilor_utils

from .hub import download_checkpoint
from .transforms import (
    crop_square,
    expand_square_bbox,
    to_model_tensor,
    xyxy_to_xywh,
)
from .visibility_net import HandVisibilityNet


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HandResult:
    """Detection result for a single hand."""

    hand_bbox: list[float]
    """Bounding box in [x1, y1, x2, y2] format (image pixels)."""

    bbox_conf: float
    """YOLO hand-detection confidence score."""

    is_right: bool
    """``True`` if the hand is a right hand."""

    keypoints_2d: np.ndarray
    """(21, 2) keypoint coordinates in image pixels."""

    keypoints_3d: np.ndarray
    """(21, 3) keypoint coordinates in MANO camera space."""

    visibility: np.ndarray
    """(21,) per-keypoint visibility probability in [0, 1]."""

    wilor_preds: dict = field(default_factory=dict, repr=False)
    """Raw WiLoR output dictionary (for advanced usage)."""


# ---------------------------------------------------------------------------
# WiLoR subclass that preserves bbox confidence
# ---------------------------------------------------------------------------


class _WiLorWithConf(WiLorHandPose3dEstimationPipeline):
    """Thin subclass that adds ``bbox_conf`` to each detection dict."""

    @torch.no_grad()
    def predict(self, image, **kwargs):
        self.logger.info("start hand detection >>> ")
        detections = self.hand_detector(
            image, conf=kwargs.get("hand_conf", 0.3), verbose=self.verbose
        )[0]
        detect_rets = []
        bboxes = []
        is_rights = []
        for det in detections:
            hand_bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_rights.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(hand_bbox[:4].tolist())
            detect_rets.append({
                "hand_bbox": bboxes[-1],
                "is_right": is_rights[-1],
                "bbox_conf": float(hand_bbox[4]),
            })

        if len(bboxes) == 0:
            self.logger.warning("No hand detected!")
            return detect_rets

        bboxes = np.stack(bboxes)

        rescale_factor = kwargs.get("rescale_factor", 2.5)
        center = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
        scale = rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2])
        self.logger.info(f"detect {bboxes.shape[0]} hands")
        self.logger.info("start hand 3d pose estimation >>> ")
        img_patches = []
        img_size = np.array([image.shape[1], image.shape[0]])
        for i in range(bboxes.shape[0]):
            bbox_size = scale[i].max()
            patch_width = patch_height = self.IMAGE_SIZE
            right = is_rights[i]
            flip = right == 0
            box_center = center[i]

            cvimg = image.copy()
            downsampling_factor = ((bbox_size * 1.0) / patch_width)
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(
                    cvimg, sigma=(downsampling_factor - 1) / 2,
                    channel_axis=2, preserve_range=True,
                )

            img_patch_cv, _trans = wilor_utils.generate_image_patch_cv2(
                cvimg,
                box_center[0], box_center[1],
                bbox_size, bbox_size,
                patch_width, patch_height,
                flip, 1.0, 0,
                border_mode=cv2.BORDER_CONSTANT,
            )
            img_patches.append(img_patch_cv)

        img_patches = np.stack(img_patches)
        img_patches = torch.from_numpy(img_patches).to(
            device=self.device, dtype=self.dtype
        )
        wilor_output = self.wilor_model(img_patches)
        wilor_output = {k: v.cpu().float().numpy() for k, v in wilor_output.items()}

        for i in range(len(detect_rets)):
            wilor_output_i = {key: val[[i]] for key, val in wilor_output.items()}
            pred_cam = wilor_output_i["pred_cam"]
            bbox_size = scale[i].max()
            box_center = center[i]
            right = is_rights[i]
            multiplier = (2 * right - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            if right == 0:
                wilor_output_i["pred_keypoints_3d"][:, :, 0] = (
                    -wilor_output_i["pred_keypoints_3d"][:, :, 0]
                )
                wilor_output_i["pred_vertices"][:, :, 0] = (
                    -wilor_output_i["pred_vertices"][:, :, 0]
                )
                wilor_output_i["global_orient"] = np.concatenate(
                    (
                        wilor_output_i["global_orient"][:, :, 0:1],
                        -wilor_output_i["global_orient"][:, :, 1:3],
                    ),
                    axis=-1,
                )
                wilor_output_i["hand_pose"] = np.concatenate(
                    (
                        wilor_output_i["hand_pose"][:, :, 0:1],
                        -wilor_output_i["hand_pose"][:, :, 1:3],
                    ),
                    axis=-1,
                )
            scaled_focal_length = (
                self.FOCAL_LENGTH / self.IMAGE_SIZE * img_size.max()
            )
            pred_cam_t_full = wilor_utils.cam_crop_to_full(
                pred_cam, box_center[None], bbox_size, img_size[None],
                scaled_focal_length,
            )
            wilor_output_i["pred_cam_t_full"] = pred_cam_t_full
            wilor_output_i["scaled_focal_length"] = scaled_focal_length
            pred_keypoints_2d = wilor_utils.perspective_projection(
                wilor_output_i["pred_keypoints_3d"],
                translation=pred_cam_t_full,
                focal_length=np.array([scaled_focal_length] * 2)[None],
                camera_center=img_size[None] / 2,
            )
            wilor_output_i["pred_keypoints_2d"] = pred_keypoints_2d
            detect_rets[i]["wilor_preds"] = wilor_output_i

        self.logger.info("finish detection!")
        return detect_rets


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class HandVisibilityPipeline:
    """End-to-end hand detection, 3D pose estimation, and per-keypoint
    visibility prediction.

    Example
    -------
    >>> pipe = HandVisibilityPipeline(device="cuda")
    >>> results = pipe.predict(image_rgb)
    >>> for r in results:
    ...     print(r.is_right, r.bbox_conf, r.visibility)
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        vis_checkpoint: str | None = None,
        hand_conf: float = 0.3,
        bbox_expand: float = 1.25,
        crop_size: int = 256,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.hand_conf = hand_conf
        self.bbox_expand = bbox_expand
        self.crop_size = crop_size

        # 1. WiLoR pipeline (with confidence). Swallow the MANO `print` banner
        #    emitted by smplx during initialization.
        with contextlib.redirect_stdout(io.StringIO()):
            self._wilor_pipe = _WiLorWithConf(
                device=device, dtype=dtype, verbose=False,
            )

        # 2. Visibility network (share WiLoR backbone)
        if vis_checkpoint is None:
            vis_checkpoint = download_checkpoint()
        ckpt = torch.load(vis_checkpoint, map_location="cpu", weights_only=False)
        head_sd = ckpt["model"]
        cfg = ckpt.get("config", {}).get("model", {})

        self._vis_model = HandVisibilityNet.from_wilor_backbone(
            raw_backbone=self._wilor_pipe.wilor_model.backbone,
            head_state_dict=head_sd,
            hidden_dim=cfg.get("hidden_dim", 256),
        )
        self._vis_model.to(device).eval()

    def predict(self, image: np.ndarray) -> list[HandResult]:
        """Run detection + visibility on a single RGB image.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB image.

        Returns
        -------
        List of :class:`HandResult`, one per detected hand.
        """
        detections = self._wilor_pipe.predict(image, hand_conf=self.hand_conf)
        if not detections:
            return []

        # Filter out detections without wilor_preds (detection-only, no pose)
        valid_dets = [d for d in detections if "wilor_preds" in d]
        if not valid_dets:
            return []

        # Prepare visibility crops
        tensors = []
        for det in valid_dets:
            bbox_xyxy = det["hand_bbox"]
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)
            is_right = bool(det["is_right"])

            cx, cy, side = expand_square_bbox(bbox_xywh, self.bbox_expand)
            patch, _ = crop_square(image, cx, cy, side, self.crop_size)
            if not is_right:
                patch = patch[:, ::-1, :].copy()
            tensors.append(to_model_tensor(patch))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            vis_probs = self._vis_model.predict_proba(batch).cpu().numpy()

        # Assemble results
        results: list[HandResult] = []
        for i, det in enumerate(valid_dets):
            wp = det["wilor_preds"]
            results.append(
                HandResult(
                    hand_bbox=det["hand_bbox"],
                    bbox_conf=det.get("bbox_conf", 0.0),
                    is_right=bool(det["is_right"]),
                    keypoints_2d=wp["pred_keypoints_2d"][0],
                    keypoints_3d=wp["pred_keypoints_3d"][0],
                    visibility=vis_probs[i],
                    wilor_preds=wp,
                )
            )

        return results
