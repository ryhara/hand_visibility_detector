"""Training-side construction for
:class:`hand_visibility_detector.visibility_net.HandVisibilityNet`.

Loads the WiLoR ViT backbone via ``wilor_mini`` (the WiLoR pipeline is
instantiated on CPU just to extract its backbone, then discarded).
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
    dropout: float = 0.0,
    hidden_dim: int = 256,
    num_keypoints: int = NUM_HAND_KEYPOINTS,
    freeze_backbone: bool = False,
) -> HandVisibilityNet:
    return HandVisibilityNet(
        raw_backbone=_load_wilor_backbone(),
        dropout=dropout,
        hidden_dim=hidden_dim,
        num_keypoints=num_keypoints,
        freeze_backbone=freeze_backbone,
    )
