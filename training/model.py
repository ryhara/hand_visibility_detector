"""Training-side construction for
:class:`hand_visibility_detector.visibility_net.HandVisibilityNet`.

The backbone is selectable via config (``model.backbone``):
    * ``wilor`` (default) -- WiLoR ViT backbone loaded via ``wilor_mini`` (the
      WiLoR pipeline is instantiated on CPU just to extract its backbone, then
      discarded). feat_dim 1280, WiLoR's 256->192 center-crop preprocessing.
    * ResNet (``resnet18`` | ``resnet34`` | ``resnet50`` | ``resnet101`` |
      ``resnet152``), ViT (``vit_b_16`` | ``vit_b_32`` | ``vit_l_16`` |
      ``vit_l_32`` | ``vit_h_14``), HaMeR (``hamer``), CSPNeXt
      (``cspnext_tiny`` | ``cspnext_s`` | ``cspnext_m`` | ``cspnext_l`` |
      ``cspnext_x``) -- built via
      :func:`hand_visibility_detector.backbones.build_backbone`;
      ``model.pretrained`` toggles their pre-trained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hand_visibility_detector.backbones import build_backbone
from hand_visibility_detector.visibility_net import (
    NUM_HAND_KEYPOINTS,
    HandVisibilityNet,
)


def _load_wilor_backbone() -> nn.Module:
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    pipeline = WiLorHandPose3dEstimationPipeline(device="cpu", dtype=torch.float32)
    raw_backbone = pipeline.wilor_model.backbone
    del pipeline
    return raw_backbone


def build_model(
    backbone: str = "wilor",
    pretrained: bool = True,
    dropout: float = 0.0,
    hidden_dim: int = 256,
    num_keypoints: int = NUM_HAND_KEYPOINTS,
    freeze_backbone: bool = False,
) -> HandVisibilityNet:
    if backbone.lower() == "wilor":
        return HandVisibilityNet(
            raw_backbone=_load_wilor_backbone(),
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_keypoints=num_keypoints,
            freeze_backbone=freeze_backbone,
        )

    bb = build_backbone(backbone, pretrained=pretrained)
    return HandVisibilityNet(
        backbone=bb,
        feat_dim=bb.feat_dim,
        dropout=dropout,
        hidden_dim=hidden_dim,
        num_keypoints=num_keypoints,
        freeze_backbone=freeze_backbone,
    )
