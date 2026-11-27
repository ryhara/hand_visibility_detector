"""End-to-end hand detection + 3D pose + per-keypoint visibility pipeline."""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from .hub import default_checkpoint_for_backbone, wilor_mini_asset
from .pose import (
    POSE_MODELS,
    HandPoseEstimator,
    default_pose_checkpoint,
    load_checkpoint_state_dict,
)
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
    """YOLO hand-detection confidence score (for
    :meth:`HandVisibilityPipeline.predict_boxes`, the score given with the box;
    1.0 when none is given)."""

    is_right: bool
    """``True`` if the hand is a right hand."""

    keypoints_2d: np.ndarray
    """(21, 2) keypoint coordinates in image pixels."""

    keypoints_3d: np.ndarray
    """(21, 3) keypoint coordinates in MANO camera space."""

    visibility: np.ndarray
    """(21,) per-keypoint visibility probability in [0, 1]."""

    wilor_preds: dict = field(default_factory=dict, repr=False)
    """Raw pose-model output dictionary (for advanced usage: ``betas``,
    ``pred_vertices``, ``pred_cam_t_full``, ...). Same schema for every
    ``pose_model``; the name is kept for backward compatibility."""

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
# Main pipeline
# ---------------------------------------------------------------------------


def _fresh_wilor_backbone() -> torch.nn.Module:
    """Build a random-initialised WiLoR ViT backbone (for checkpoints that
    contain fine-tuned backbone weights). Only ``mano_mean_params.npz`` is
    fetched from the Hub -- not the full WiLoR checkpoint."""
    from wilor_mini.models.vit import vit

    return vit(mano_mean_path=wilor_mini_asset("mano_mean_params.npz"))


def _pretrained_wilor_backbone(pose_model: str = "wilor") -> torch.nn.Module:
    """Build the ViT backbone of a WiLoR-family ``pose_model`` (``wilor`` /
    ``anyhand_wilor``) with its pre-trained weights (for head-only checkpoints
    when that is not the pose model the pipeline runs)."""
    backbone = _fresh_wilor_backbone()
    state = load_checkpoint_state_dict(default_pose_checkpoint(pose_model))
    backbone.load_state_dict(
        {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")},
        strict=True,
    )
    return backbone


class HandVisibilityPipeline:
    """End-to-end hand detection, 3D pose estimation, and per-keypoint
    visibility prediction.

    Hands are detected with WiLoR's YOLO detector; their 3D pose (MANO) comes
    from ``pose_model`` -- ``wilor`` (default), ``hamer``, or the AnyHand
    fine-tuned ``anyhand_wilor`` / ``anyhand_hamer`` (see
    :mod:`hand_visibility_detector.pose`).

    The visibility backbone (``wilor`` / ``hamer`` / ``anyhand_wilor`` /
    ``anyhand_hamer`` / ``resnet*`` / ``vit_*`` / ``cspnext_*``) is read from
    the checkpoint's saved config. Published
    checkpoints exist for ``wilor`` (``best.pt``) and ``hamer``
    (``best_hamer.pt``) and are auto-downloaded; any other backbone needs an
    explicit ``vis_checkpoint`` path to your own trained checkpoint.

    Example
    -------
    >>> pipe = HandVisibilityPipeline(device="cuda")            # WiLoR backbone
    >>> pipe = HandVisibilityPipeline(device="cuda", backbone="hamer")
    >>> pipe = HandVisibilityPipeline(device="cuda", pose_model="anyhand_hamer")
    >>> pipe = HandVisibilityPipeline(device="cuda", vis_checkpoint="runs/.../best.pt")
    >>> results = pipe.predict(image_rgb)
    >>> for r in results:
    ...     print(r.is_right, r.bbox_conf, r.visibility)

    Hands found otherwise (another detector, a tracker, several cameras) skip
    the detector; many images (video frames, camera views) can share batches:

    >>> results = pipe.predict_boxes(image_rgb, [[x1, y1, x2, y2]], [True])
    >>> per_image = pipe.predict_boxes_batch(images, boxes, is_right, batch_size=64)
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
        pose_model: str = "wilor",
        pose_checkpoint: str | None = None,
        focal_length: float | None = None,
        share_backbone: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        vis_checkpoint : Path to a trained visibility checkpoint. When ``None``
            the published checkpoint for ``backbone`` is downloaded from
            HuggingFace Hub (only available for ``wilor`` and ``hamer``).
        backbone : Which published checkpoint to fetch when ``vis_checkpoint``
            is ``None`` (``"wilor"`` or ``"hamer"``; defaults to the family of
            ``pose_model``). The actual model architecture is always rebuilt
            from the checkpoint's own saved config.
        backbone_weights : Optional local path to the backbone's pre-trained
            weights, used by head-only checkpoints of the ``hamer`` /
            ``anyhand_hamer`` backbones (``hamer.ckpt`` / ``anyhand_hamer.ckpt``;
            auto-downloaded when omitted, ``hamer`` also honours
            ``HAMER_WEIGHTS``).
        pose_model : 3D hand pose model: ``"wilor"``, ``"hamer"``,
            ``"anyhand_wilor"`` or ``"anyhand_hamer"``.
        pose_checkpoint : Optional local path to ``pose_model``'s weights
            (downloaded and cached from HuggingFace Hub when omitted).
        focal_length : Override ``pose_model``'s default focal length.
        share_backbone : Run a head-only visibility checkpoint on the backbone
            of a different pose model of the same family (e.g. a ``hamer`` head
            on ``anyhand_hamer``; saves one ViT-H, ~2.5 GB) instead of loading
            the backbone the head was trained on. Heads change substantially
            across these backbones (the AnyHand ones are fine-tuned), so this
            is off by default. A head whose backbone *is* ``pose_model``
            (``wilor`` / ``hamer`` / ``anyhand_wilor`` / ``anyhand_hamer``)
            always shares it.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.hand_conf = hand_conf
        self.bbox_expand = bbox_expand
        self.crop_size = crop_size
        self.return_rotations = return_rotations

        # 1. Hand detector + 3D pose model. Swallow the MANO `print` banner
        #    emitted by smplx during initialization.
        with contextlib.redirect_stdout(io.StringIO()):
            self._pose = HandPoseEstimator(
                pose_model,
                device=device,
                dtype=dtype,
                checkpoint=pose_checkpoint,
                focal_length=focal_length,
            )

        # 2. Visibility network. The architecture (backbone / hidden_dim) is
        #    rebuilt from the checkpoint's saved config.
        if vis_checkpoint is None:
            vis_checkpoint = default_checkpoint_for_backbone(
                backbone or self._pose.family
            )
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

        # A head-only checkpoint shares the (frozen) pose-model backbone when it
        # carries the weights the head was trained on, i.e. the head's backbone
        # is the pose model itself. Other backbones of the same family (original
        # vs. AnyHand fine-tuned) are shared only on request.
        backbone_family = (
            POSE_MODELS[backbone_name].family
            if backbone_name in POSE_MODELS
            else backbone_name
        )
        shared_backbone = None
        if head_only and backbone_family == self._pose.family:
            if self._pose.pose_model == backbone_name:
                shared_backbone = self._pose.backbone
            elif share_backbone:
                logger.warning(
                    "share_backbone=True: the visibility head (trained on the "
                    "%r backbone) runs on the %r backbone.",
                    backbone_name, self._pose.pose_model,
                )
                shared_backbone = self._pose.backbone
        if share_backbone and shared_backbone is None:
            logger.warning(
                "share_backbone=True ignored: the %s visibility checkpoint "
                "cannot use the %r pose model's backbone.",
                "head-only " + backbone_name if head_only else "fine-tuned",
                self._pose.pose_model,
            )

        if backbone_family == "wilor":
            if shared_backbone is not None:
                raw_backbone = shared_backbone
            elif head_only:
                raw_backbone = _pretrained_wilor_backbone(backbone_name)
            else:
                # The checkpoint carries fine-tuned backbone weights: use a
                # separate backbone so the pose model's weights stay intact.
                raw_backbone = _fresh_wilor_backbone()
            self._vis_model = HandVisibilityNet(
                raw_backbone=raw_backbone,
                hidden_dim=hidden_dim,
                freeze_backbone=True,
            )
        else:
            from .backbones import HamerViTBackbone, build_backbone

            if shared_backbone is not None:
                bb = HamerViTBackbone(vit=shared_backbone)
            else:
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
        detections = self._pose.detect(image, hand_conf=self.hand_conf)
        return self._predict_detections(
            [image], [detections], return_rotations,
            batch_size=max(len(detections), 1),
        )[0]

    def predict_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray | Sequence[Sequence[float]],
        is_right: np.ndarray | Sequence[bool],
        scores: np.ndarray | Sequence[float] | None = None,
        return_rotations: bool | None = None,
    ) -> list[HandResult]:
        """Run pose + visibility on hands whose boxes are given (found by
        another detector, a tracker, several cameras, ...), without the hand
        detector.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB image.
        boxes : (K, 4) hand boxes as [x1, y1, x2, y2] (image pixels), with
            x2 > x1 and y2 > y1. They may reach outside the image.
        is_right : (K,) whether each hand is a right hand (left hands are
            mirrored for the models, as the detector's are).
        scores : (K,) optional score of each box, reported as
            :attr:`HandResult.bbox_conf` (1.0 when omitted).
        return_rotations : As for :meth:`predict`.

        Returns
        -------
        List of :class:`HandResult`, one per box, in the order of ``boxes``.
        """
        return self.predict_boxes_batch(
            [image], [boxes], [is_right],
            None if scores is None else [scores],
            return_rotations=return_rotations,
            batch_size=max(len(boxes), 1),
        )[0]

    def predict_boxes_batch(
        self,
        images: Sequence[np.ndarray],
        boxes: Sequence[np.ndarray | Sequence[Sequence[float]]],
        is_right: Sequence[np.ndarray | Sequence[bool]],
        scores: Sequence[np.ndarray | Sequence[float]] | None = None,
        return_rotations: bool | None = None,
        batch_size: int = 32,
    ) -> list[list[HandResult]]:
        """:meth:`predict_boxes` over several images (e.g. the frames of a
        video or the views of a camera rig; their sizes may differ). The hands
        of all the images go through the pose model and the visibility network
        ``batch_size`` at a time, which keeps a GPU busy when each image holds
        only a hand or two.

        ``boxes[i]``, ``is_right[i]`` (and ``scores[i]``) are the hands of
        ``images[i]``, as for :meth:`predict_boxes`; an image may have none.
        It also runs under the caller's ``torch.autocast`` (half precision);
        check the keypoints against float32 ones before relying on it.

        Returns
        -------
        One list of :class:`HandResult` per image, one result per box.
        """
        n_images = len(images)
        if len(boxes) != n_images or len(is_right) != n_images or (
            scores is not None and len(scores) != n_images
        ):
            raise ValueError(
                "boxes, is_right (and scores) need one entry per image "
                f"({n_images} images)"
            )
        detections = [
            _box_detections(boxes[i], is_right[i], None if scores is None else scores[i])
            for i in range(n_images)
        ]
        return self._predict_detections(images, detections, return_rotations, batch_size)

    def _predict_detections(
        self,
        images: Sequence[np.ndarray],
        detections: list[list[dict]],
        return_rotations: bool | None,
        batch_size: int,
    ) -> list[list[HandResult]]:
        """Pose + visibility of the hands ``detections[i]`` of ``images[i]``."""
        if return_rotations is None:
            return_rotations = self.return_rotations
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, not {batch_size}")
        self._pose.estimate_batch(images, detections, batch_size=batch_size)

        hands = [
            (image, det) for image, dets in zip(images, detections) for det in dets
        ]
        vis_probs = [
            self._visibility(hands[start:start + batch_size])
            for start in range(0, len(hands), batch_size)
        ]

        results: list[list[HandResult]] = []
        k = 0
        for dets in detections:
            image_results = []
            for det in dets:
                batch_index, row = divmod(k, batch_size)
                image_results.append(
                    _hand_result(det, vis_probs[batch_index][row], return_rotations)
                )
                k += 1
            results.append(image_results)
        return results

    def _visibility(self, hands: list[tuple[np.ndarray, dict]]) -> np.ndarray:
        """(B, 21) visibility probabilities of one batch of ``(image,
        detection)`` pairs."""
        tensors = []
        for image, det in hands:
            bbox_xywh = xyxy_to_xywh(det["hand_bbox"])
            cx, cy, side = expand_square_bbox(bbox_xywh, self.bbox_expand)
            patch, _ = crop_square(image, cx, cy, side, self.crop_size)
            if not bool(det["is_right"]):
                patch = patch[:, ::-1, :].copy()
            tensors.append(to_model_tensor(patch))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            return self._vis_model.predict_proba(batch).float().cpu().numpy()


def _box_detections(
    boxes: np.ndarray | Sequence[Sequence[float]],
    is_right: np.ndarray | Sequence[bool],
    scores: np.ndarray | Sequence[float] | None,
) -> list[dict]:
    """Given hand boxes as the detection dicts of
    :meth:`HandPoseEstimator.detect`."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be (K, 4) [x1, y1, x2, y2], not {boxes.shape}")
    if not np.isfinite(boxes).all() or (
        (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
    ).any():
        raise ValueError("boxes must be finite, with x2 > x1 and y2 > y1")
    is_right = np.asarray(is_right).reshape(-1)
    scores = (
        np.ones(len(boxes)) if scores is None
        else np.asarray(scores, dtype=np.float64).reshape(-1)
    )
    if len(is_right) != len(boxes) or len(scores) != len(boxes):
        raise ValueError(
            f"{len(boxes)} boxes but {len(is_right)} is_right and {len(scores)} scores"
        )
    return [
        {
            "hand_bbox": box.tolist(),
            "is_right": float(bool(right)),
            "bbox_conf": float(score),
        }
        for box, right, score in zip(boxes, is_right, scores)
    ]


def _hand_result(det: dict, visibility: np.ndarray, return_rotations: bool) -> HandResult:
    """A detection dict (with ``pose_preds``) and its visibility as a
    :class:`HandResult`."""
    wp = det["pose_preds"]
    global_orient = global_orient_euler = None
    hand_pose = hand_pose_euler = None
    if return_rotations:
        global_orient = wp["global_orient"].reshape(3).copy()
        hand_pose = wp["hand_pose"].reshape(-1, 3).copy()
        global_orient_euler = axis_angle_to_euler(global_orient)
        hand_pose_euler = axis_angle_to_euler(hand_pose)
    return HandResult(
        hand_bbox=det["hand_bbox"],
        bbox_conf=det.get("bbox_conf", 0.0),
        is_right=bool(det["is_right"]),
        keypoints_2d=wp["pred_keypoints_2d"][0],
        keypoints_3d=wp["pred_keypoints_3d"][0],
        visibility=visibility,
        wilor_preds=wp,
        global_orient=global_orient,
        global_orient_euler=global_orient_euler,
        hand_pose=hand_pose,
        hand_pose_euler=hand_pose_euler,
    )
