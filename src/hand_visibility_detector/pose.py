"""Hand detection + 3D hand pose (MANO) estimation with a selectable pose model.

Hands are always detected with WiLoR's YOLO hand detector; the 3D pose model
that regresses MANO parameters for each detected hand is chosen by name:

    * ``wilor``         -- WiLoR (``wilor_final.ckpt``, via ``wilor_mini``).
    * ``hamer``         -- HaMeR (``hamer.ckpt``): ViTPose-H backbone +
                           transformer-decoder MANO head, ported in
                           :mod:`~hand_visibility_detector.hamer_head`.
    * ``anyhand_wilor`` -- WiLoR fine-tuned with the AnyHand synthetic dataset.
    * ``anyhand_hamer`` -- HaMeR fine-tuned with the AnyHand synthetic dataset.

The AnyHand checkpoints (``chen-si-02/AnyHand-Models``) share the architecture
and weight layout of the originals, so they load into the same models.

Every pose model returns, per hand, a dict with the schema of ``wilor_mini``'s
pipeline output: axis-angle ``global_orient`` ``(1, 1, 3)`` / ``hand_pose``
``(1, 15, 3)``, ``betas``, ``pred_cam``, ``pred_keypoints_3d`` ``(1, 21, 3)``,
``pred_vertices`` ``(1, 778, 3)``, ``pred_cam_t_full``,
``scaled_focal_length`` and ``pred_keypoints_2d`` ``(1, 21, 2)`` (image pixels).
Left hands are mirrored back after being regressed as right hands.
"""

from __future__ import annotations

import inspect
import pickle
import types
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.filters import gaussian
from wilor_mini.utils import utils as wilor_utils

from .hub import download_anyhand_checkpoint, wilor_mini_asset


# ---------------------------------------------------------------------------
# Pose model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseModelSpec:
    family: str
    """Architecture: ``"wilor"`` or ``"hamer"``."""

    focal_length: float
    """Focal length (pixels, for a 256 crop) of the model's camera; used by
    WiLoR's refinement stage and for the full-image camera / 2D projection."""

    rescale_factor: float
    """Crop side = ``rescale_factor * max(bbox w, h)``."""


POSE_MODELS: dict[str, PoseModelSpec] = {
    # wilor_mini's defaults (focal 5000, rescale 2.5).
    "wilor": PoseModelSpec("wilor", 5000.0, 2.5),
    # HaMeR's model_config.yaml / demo defaults (focal 5000, rescale 2.0).
    "hamer": PoseModelSpec("hamer", 5000.0, 2.0),
    # AnyHand's published configs (model_config_wilor.yaml sets FOCAL_LENGTH
    # 1000, as used by AnyHand's own predictor; the config embedded in
    # anyhand_wilor.ckpt says 500 -- override with ``focal_length`` if needed).
    "anyhand_wilor": PoseModelSpec("wilor", 1000.0, 2.5),
    "anyhand_hamer": PoseModelSpec("hamer", 5000.0, 2.0),
}


def default_pose_checkpoint(pose_model: str) -> str:
    """Local path of ``pose_model``'s weights, downloading them if needed."""
    if pose_model == "wilor":
        return wilor_mini_asset("wilor_final.ckpt")
    if pose_model == "hamer":
        from .backbones.hamer import _resolve_hamer_weights

        return _resolve_hamer_weights(None)
    return download_anyhand_checkpoint(pose_model)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


class _Stub:
    """Inert stand-in for any non-tensor object pickled in a checkpoint (e.g.
    the OmegaConf / yacs config in Lightning's ``hyper_parameters``). Accepts
    every pickle opcode that populates an object, and ignores the data."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs) -> "_Stub":
        return _Stub()

    def __setstate__(self, state) -> None:
        pass

    def __setitem__(self, key, value) -> None:
        pass

    def append(self, item) -> None:
        pass

    def extend(self, items) -> None:
        pass

    def add(self, item) -> None:
        pass


class _StateDictUnpickler(pickle.Unpickler):
    """Resolves only what a tensor ``state_dict`` needs; everything else
    becomes a :class:`_Stub`, so neither ``omegaconf`` nor ``yacs`` has to be
    installed and no foreign code runs while unpickling."""

    def find_class(self, module: str, name: str):
        if (
            (module == "collections" and name == "OrderedDict")
            or (module.startswith("torch") and name.startswith("_rebuild_"))
            or (module == "torch.nn.parameter" and name == "Parameter")
            or (module == "torch" and (
                name in ("Size", "device")
                or isinstance(getattr(torch, name, None), torch.dtype)
            ))
        ):
            return super().find_class(module, name)
        return _Stub


_STATE_DICT_PICKLE = types.SimpleNamespace(
    __name__="_state_dict_pickle",
    Unpickler=_StateDictUnpickler,
    load=pickle.load,
)


def load_checkpoint_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Return the ``state_dict`` of a (Lightning) checkpoint.

    On torch >= 2.1 the file is memory-mapped, so only the tensors actually
    copied into a model are read from disk (torch 2.0 reads it all).
    """
    kwargs = {}
    if "mmap" in inspect.signature(torch.load).parameters:  # torch >= 2.1
        kwargs["mmap"] = True
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_STATE_DICT_PICKLE,
        **kwargs,
    )
    return ckpt.get("state_dict", ckpt)


# ---------------------------------------------------------------------------
# Detector + pose estimator
# ---------------------------------------------------------------------------


class HandPoseEstimator:
    """WiLoR's YOLO hand detector + a selectable 3D hand pose model.

    Example
    -------
    >>> est = HandPoseEstimator("anyhand_hamer", device="cuda")
    >>> dets = est.predict(image_rgb, hand_conf=0.3)
    >>> dets[0]["pose_preds"]["hand_pose"].shape
    (1, 15, 3)
    """

    IMAGE_SIZE = 256

    def __init__(
        self,
        pose_model: str = "wilor",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        checkpoint: str | None = None,
        focal_length: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        pose_model : One of :data:`POSE_MODELS`.
        checkpoint : Local path to ``pose_model``'s weights. When ``None`` they
            are downloaded (and cached) from HuggingFace Hub.
        focal_length : Override the model's default focal length (see
            :class:`PoseModelSpec`).
        """
        from ultralytics import YOLO

        key = pose_model.lower()
        if key not in POSE_MODELS:
            raise ValueError(
                f"unknown pose_model {pose_model!r} "
                f"(supported: {', '.join(POSE_MODELS)})"
            )
        spec = POSE_MODELS[key]
        self.pose_model = key
        self.family = spec.family
        self.focal_length = (
            float(focal_length) if focal_length is not None else spec.focal_length
        )
        self.rescale_factor = spec.rescale_factor
        self.device = torch.device(device)
        self.dtype = dtype

        self.hand_detector = YOLO(wilor_mini_asset("detector.pt"))
        self.hand_detector.to(self.device)

        if checkpoint is None:
            checkpoint = default_pose_checkpoint(key)
        state_dict = load_checkpoint_state_dict(checkpoint)
        mano_model_path = wilor_mini_asset("MANO_RIGHT.pkl")
        if self.family == "wilor":
            from wilor_mini.models.wilor import WiLor

            model = WiLor(
                mano_model_path=mano_model_path,
                mano_mean_path=wilor_mini_asset("mano_mean_params.npz"),
                focal_length=self.focal_length,
                image_size=self.IMAGE_SIZE,
            )
            # Like wilor_mini: the checkpoint also holds the discriminator and
            # may lack some MANO buffers (those come from MANO_RIGHT.pkl).
            missing = model.load_state_dict(state_dict, strict=False).missing_keys
            missing = [k for k in missing if not k.startswith("mano.")]
            if missing:
                raise RuntimeError(
                    f"{checkpoint!r} is not a WiLoR checkpoint "
                    f"(missing {len(missing)} weights, e.g. {missing[0]!r})"
                )
        else:
            from .hamer_head import HamerPoseNet

            model = HamerPoseNet(mano_model_path=mano_model_path)
            model.load_hamer_state_dict(state_dict)
        del state_dict
        self.model = model.eval().to(self.device, dtype=dtype)

    @property
    def backbone(self) -> nn.Module:
        """The pose model's ViT backbone (WiLoR: the ViT with fused MANO
        tokens; HaMeR: plain ViTPose-H)."""
        return self.model.backbone

    @torch.no_grad()
    def detect(self, image: np.ndarray, hand_conf: float = 0.3) -> list[dict]:
        """Detect hands. Returns dicts with ``hand_bbox`` ([x1, y1, x2, y2]),
        ``is_right`` (1.0 / 0.0) and ``bbox_conf``."""
        detections = self.hand_detector(image, conf=hand_conf, verbose=False)[0]
        rets = []
        for det in detections:
            hand_bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            rets.append({
                "hand_bbox": hand_bbox[:4].tolist(),
                "is_right": det.boxes.cls.cpu().detach().squeeze().item(),
                "bbox_conf": float(hand_bbox[4]),
            })
        return rets

    @torch.no_grad()
    def estimate(self, image: np.ndarray, detections: list[dict]) -> list[dict]:
        """Run the pose model on each detection, storing its output under the
        ``pose_preds`` key (in place). Returns ``detections``."""
        if not detections:
            return detections
        bboxes = np.array([d["hand_bbox"] for d in detections], dtype=np.float32)
        center = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
        bbox_sizes = self.rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2]).max(axis=1)
        img_size = np.array([image.shape[1], image.shape[0]])

        img_patches = []
        for i, det in enumerate(detections):
            bbox_size = bbox_sizes[i]
            cvimg = image.copy()
            # Blur image to avoid aliasing artifacts
            downsampling_factor = bbox_size / self.IMAGE_SIZE / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(
                    cvimg, sigma=(downsampling_factor - 1) / 2,
                    channel_axis=2, preserve_range=True,
                )
            img_patch, _trans = wilor_utils.generate_image_patch_cv2(
                cvimg,
                center[i][0], center[i][1],
                bbox_size, bbox_size,
                self.IMAGE_SIZE, self.IMAGE_SIZE,
                det["is_right"] == 0, 1.0, 0,
                border_mode=cv2.BORDER_CONSTANT,
            )
            img_patches.append(img_patch)

        img_patches = torch.from_numpy(np.stack(img_patches)).to(
            device=self.device, dtype=self.dtype
        )
        output = self.model(img_patches)
        output = {k: v.cpu().float().numpy() for k, v in output.items()}

        scaled_focal_length = self.focal_length / self.IMAGE_SIZE * img_size.max()
        for i, det in enumerate(detections):
            preds = {key: val[[i]] for key, val in output.items()}
            right = det["is_right"]
            pred_cam = preds["pred_cam"]
            pred_cam[:, 1] = (2 * right - 1) * pred_cam[:, 1]
            if right == 0:
                preds["pred_keypoints_3d"][:, :, 0] = -preds["pred_keypoints_3d"][:, :, 0]
                preds["pred_vertices"][:, :, 0] = -preds["pred_vertices"][:, :, 0]
                for key in ("global_orient", "hand_pose"):
                    preds[key] = np.concatenate(
                        (preds[key][:, :, 0:1], -preds[key][:, :, 1:3]), axis=-1
                    )
            pred_cam_t_full = wilor_utils.cam_crop_to_full(
                pred_cam, center[i][None], bbox_sizes[i], img_size[None],
                scaled_focal_length,
            )
            preds["pred_cam_t_full"] = pred_cam_t_full
            preds["scaled_focal_length"] = scaled_focal_length
            preds["pred_keypoints_2d"] = wilor_utils.perspective_projection(
                preds["pred_keypoints_3d"],
                translation=pred_cam_t_full,
                focal_length=np.array([scaled_focal_length] * 2)[None],
                camera_center=img_size[None] / 2,
            )
            det["pose_preds"] = preds
        return detections

    def predict(self, image: np.ndarray, hand_conf: float = 0.3) -> list[dict]:
        """Detect hands in an ``(H, W, 3)`` uint8 RGB image and estimate their
        3D pose. Returns the detection dicts with ``pose_preds`` attached."""
        return self.estimate(image, self.detect(image, hand_conf=hand_conf))
