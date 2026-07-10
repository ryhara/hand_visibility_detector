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

from .hub import default_checkpoint_for_backbone
from .rotations import axis_angle_to_euler
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

    global_orient: np.ndarray | None = None
    """(3,) MANO global (wrist) orientation as axis-angle in camera space.
    Only populated when ``return_rotations`` is enabled."""

    global_orient_euler: np.ndarray | None = None
    """(3,) global orientation as (roll, pitch, yaw) in degrees.
    Only populated when ``return_rotations`` is enabled."""

    hand_pose: np.ndarray | None = None
    """(15, 3) MANO per-joint pose as axis-angle, relative to each parent joint.
    Only populated when ``return_rotations`` is enabled."""

    hand_pose_euler: np.ndarray | None = None
    """(15, 3) per-joint pose as (roll, pitch, yaw) in degrees.
    Only populated when ``return_rotations`` is enabled."""


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


def _fresh_wilor_backbone() -> torch.nn.Module:
    """Build a random-initialised WiLoR ViT backbone (for checkpoints that
    contain fine-tuned backbone weights). Only ``mano_mean_params.npz`` is
    fetched from the Hub -- not the full WiLoR checkpoint."""
    import os

    import wilor_mini
    from wilor_mini.models.vit import vit

    pretrained_dir = os.path.join(
        os.path.dirname(wilor_mini.__file__), "pretrained_models"
    )
    path = os.path.join(pretrained_dir, "mano_mean_params.npz")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="warmshao/WiLoR-mini",
            subfolder="pretrained_models",
            filename="mano_mean_params.npz",
            local_dir=os.path.dirname(wilor_mini.__file__),
        )
    return vit(mano_mean_path=path)


class HandVisibilityPipeline:
    """End-to-end hand detection, 3D pose estimation, and per-keypoint
    visibility prediction.

    The visibility backbone (``wilor`` / ``hamer`` / ``resnet*`` / ``vit_*`` /
    ``cspnext_*``) is read from the checkpoint's saved config. Published
    checkpoints exist for ``wilor`` (``best.pt``) and ``hamer``
    (``best_hamer.pt``) and are auto-downloaded; any other backbone needs an
    explicit ``vis_checkpoint`` path to your own trained checkpoint.

    Example
    -------
    >>> pipe = HandVisibilityPipeline(device="cuda")            # WiLoR backbone
    >>> pipe = HandVisibilityPipeline(device="cuda", backbone="hamer")
    >>> pipe = HandVisibilityPipeline(device="cuda", vis_checkpoint="runs/.../best.pt")
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
        return_rotations: bool = False,
        backbone: str | None = None,
        backbone_weights: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        vis_checkpoint : Path to a trained visibility checkpoint. When ``None``
            the published checkpoint for ``backbone`` is downloaded from
            HuggingFace Hub (only available for ``wilor`` and ``hamer``).
        backbone : Which published checkpoint to fetch when ``vis_checkpoint``
            is ``None`` (``"wilor"``, the default, or ``"hamer"``). The actual
            model architecture is always rebuilt from the checkpoint's own
            saved config.
        backbone_weights : Optional local path to the backbone's pre-trained
            weights, used by head-only checkpoints of the ``hamer`` backbone
            (the published ``hamer.ckpt``; resolved via ``HAMER_WEIGHTS`` or
            auto-downloaded from the official HaMeR Space when omitted).
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.hand_conf = hand_conf
        self.bbox_expand = bbox_expand
        self.crop_size = crop_size
        self.return_rotations = return_rotations

        # 1. WiLoR pipeline (with confidence). Swallow the MANO `print` banner
        #    emitted by smplx during initialization.
        with contextlib.redirect_stdout(io.StringIO()):
            self._wilor_pipe = _WiLorWithConf(
                device=device, dtype=dtype, verbose=False,
            )

        # 2. Visibility network. The architecture (backbone / hidden_dim) is
        #    rebuilt from the checkpoint's saved config.
        if vis_checkpoint is None:
            vis_checkpoint = default_checkpoint_for_backbone(backbone or "wilor")
        ckpt = torch.load(vis_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"]
        cfg = ckpt.get("config", {}).get("model", {})
        backbone_name = str(cfg.get("backbone", "wilor")).lower()
        if backbone is not None and backbone.lower() != backbone_name:
            logger.warning(
                "backbone=%r was requested but the checkpoint was trained with "
                "backbone=%r; using the checkpoint's backbone.",
                backbone, backbone_name,
            )
        hidden_dim = int(cfg.get("hidden_dim", 256))
        # head_only=True: the checkpoint holds only head weights, so the
        # backbone must be built with its own pre-trained weights. Otherwise the
        # checkpoint holds the full (fine-tuned) model state.
        head_only = bool(ckpt.get("head_only", True))

        if backbone_name == "wilor":
            if head_only:
                # Share the (frozen, pre-trained) WiLoR backbone with the pose
                # pipeline.
                raw_backbone = self._wilor_pipe.wilor_model.backbone
            else:
                # The checkpoint carries fine-tuned backbone weights: use a
                # separate backbone so the pose pipeline's weights stay intact.
                raw_backbone = _fresh_wilor_backbone()
            self._vis_model = HandVisibilityNet(
                raw_backbone=raw_backbone,
                hidden_dim=hidden_dim,
                freeze_backbone=True,
            )
        else:
            from .backbones import build_backbone

            bb_kwargs = {}
            if backbone_weights is not None:
                bb_kwargs["weights"] = backbone_weights
            bb = build_backbone(backbone_name, pretrained=head_only, **bb_kwargs)
            self._vis_model = HandVisibilityNet(
                backbone=bb,
                feat_dim=bb.feat_dim,
                hidden_dim=hidden_dim,
                freeze_backbone=True,
            )

        if head_only:
            self._vis_model.load_head_state_dict(state_dict)
        else:
            self._vis_model.load_state_dict(state_dict)
        self._vis_model.to(device).eval()

    def predict(
        self,
        image: np.ndarray,
        return_rotations: bool | None = None,
    ) -> list[HandResult]:
        """Run detection + visibility on a single RGB image.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB image.
        return_rotations : If ``True``, populate ``global_orient`` /
            ``hand_pose`` (axis-angle) and their (roll, pitch, yaw) Euler
            counterparts on each result. Defaults to the value passed at
            construction time.

        Returns
        -------
        List of :class:`HandResult`, one per detected hand.
        """
        if return_rotations is None:
            return_rotations = self.return_rotations
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
            global_orient = global_orient_euler = None
            hand_pose = hand_pose_euler = None
            if return_rotations:
                global_orient = wp["global_orient"].reshape(3).copy()
                hand_pose = wp["hand_pose"].reshape(-1, 3).copy()
                global_orient_euler = axis_angle_to_euler(global_orient)
                hand_pose_euler = axis_angle_to_euler(hand_pose)
            results.append(
                HandResult(
                    hand_bbox=det["hand_bbox"],
                    bbox_conf=det.get("bbox_conf", 0.0),
                    is_right=bool(det["is_right"]),
                    keypoints_2d=wp["pred_keypoints_2d"][0],
                    keypoints_3d=wp["pred_keypoints_3d"][0],
                    visibility=vis_probs[i],
                    wilor_preds=wp,
                    global_orient=global_orient,
                    global_orient_euler=global_orient_euler,
                    hand_pose=hand_pose,
                    hand_pose_euler=hand_pose_euler,
                )
            )

        return results
