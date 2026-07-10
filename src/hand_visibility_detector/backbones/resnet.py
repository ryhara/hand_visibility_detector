"""ResNet feature-map backbone (any depth) for :class:`HandVisibilityNet`.

See :mod:`hand_visibility_detector.backbones` for the shared backbone contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_RESNET_NAMES = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}


class ResNetBackbone(nn.Module):
    """A torchvision ResNet truncated after ``layer4``; returns the
    ``(B, feat_dim, H/32, W/32)`` feature map.

    Fully convolutional, so the input is consumed at whatever spatial size it
    arrives (e.g. WiLoR's 256 crop -> ``(B, feat_dim, 8, 8)``); no weights
    change. ``feat_dim`` is read from the model (512 for resnet18/34, 2048 for
    resnet50/101/152).
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        import torchvision

        m = torchvision.models.get_model(
            name, weights="DEFAULT" if pretrained else None
        )
        self.feat_dim = int(m.fc.in_features)
        self.body = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
