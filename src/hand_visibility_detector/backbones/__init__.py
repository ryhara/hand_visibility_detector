"""Selectable feature-map backbones for :class:`HandVisibilityNet`.

Every backbone here is an ``nn.Module`` whose ``forward`` returns a spatial
feature map ``(B, C, H, W)`` and which exposes an integer ``feat_dim``
attribute (``C``). :class:`~hand_visibility_detector.visibility_net.VisibilityHead`
is agnostic to the spatial size and only needs ``feat_dim`` to size its first
1x1 conv, so any of these backbones can be plugged in interchangeably.

Each model family lives in its own submodule (``resnet``, ``vit``, ``hamer``,
``cspnext``); this package wires them together via :func:`build_backbone`.

Supported names (case-insensitive):
    * ResNet  : ``resnet18`` | ``resnet34`` | ``resnet50`` | ``resnet101`` |
                ``resnet152``  (feat_dim 512 for 18/34, 2048 for 50/101/152)
                -- from ``torchvision``.
    * ViT     : ``vit_b_16`` | ``vit_b_32`` | ``vit_l_16`` | ``vit_l_32`` |
                ``vit_h_14``   (feat_dim 768 / 1024 / 1280) -- from
                ``torchvision``. aliases: ``vit_b`` -> ``vit_b_16``,
                ``vit_l`` -> ``vit_l_16``, ``vit_h`` -> ``vit_h_14``
    * HaMeR   : ``hamer`` -- HaMeR's ViTPose-H backbone (feat_dim 1280), ported
                self-contained in :mod:`~.hamer_vit`. The
                weights are read from the published ``hamer.ckpt`` (``backbone.``
                prefix): a local path (``HAMER_WEIGHTS`` env var or
                ``external/hamer_weights/hamer.ckpt``) when present, otherwise
                auto-downloaded from the official HaMeR Space. Like WiLoR it
                consumes a 256x192 center-crop and outputs ``(B, 1280, 16, 12)``,
                but it is a plain backbone with no fused MANO regression tokens.
    * CSPNeXt : ``cspnext_tiny`` | ``cspnext_s`` | ``cspnext_m`` |
                ``cspnext_l`` | ``cspnext_x`` (feat_dim 384 / 512 / 768 / 1024 /
                1280; ``cspnext`` aliases the ``m`` variant). RTMDet's CSPNeXt
                backbone, implemented in :mod:`~.cspnext` in pure PyTorch with
                module names matching MMDetection's checkpoint so the official
                OpenMMLab ``cspnext-*_imagenet`` ImageNet weights load directly
                (fetched and cached via ``torch.hub``). No ``mmdet`` / ``mmcv``
                dependency is needed.

The WiLoR ViT backbone is handled separately in
:func:`hand_visibility_detector.visibility_net.HandVisibilityNet` (it must be
loaded from the ``wilor_mini`` pipeline), so it is not built here.

Input-size note
---------------
ResNets and CSPNeXt are fully convolutional, so they accept WiLoR's crop size
(256) as-is without changing any pre-trained weight (the OpenMMLab CSPNeXt
weights were trained at 224, but the backbone has no positional embeddings).
ViTs tie their positional embeddings to a fixed input size (read from the model:
224 for ImageNet-1k weights, larger for SWAG weights), so the ViT backbone
resizes the input to that size internally and keeps the pre-trained weights
untouched. HaMeR's ViTPose-H, like WiLoR, consumes a 256x192 center-crop of the
square 256 input.
"""

from __future__ import annotations

import torch.nn as nn

from .cspnext import _CSPNEXT_ALIASES, _CSPNEXT_ARCH, CSPNeXtBackbone
from .hamer import _HAMER_NAMES, HamerViTBackbone
from .resnet import _RESNET_NAMES, ResNetBackbone
from .vit import _VIT_ALIASES, _VIT_NAMES, ViTBackbone

__all__ = [
    "build_backbone",
    "ResNetBackbone",
    "ViTBackbone",
    "HamerViTBackbone",
    "CSPNeXtBackbone",
]


def build_backbone(name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    """Build a feature-map backbone by name.

    Returns an ``nn.Module`` with a ``feat_dim`` attribute. ``"wilor"`` is not
    handled here -- see :class:`HandVisibilityNet` / the training ``model.py``.
    Extra ``kwargs`` are forwarded to the backbone constructor (e.g.
    ``weights=...`` for ``hamer``).
    """
    key = name.lower()
    for aliases in (_VIT_ALIASES, _CSPNEXT_ALIASES):
        if key in aliases:
            key = aliases[key]
            break
    if key in _RESNET_NAMES:
        return ResNetBackbone(key, pretrained=pretrained, **kwargs)
    if key in _VIT_NAMES:
        return ViTBackbone(key, pretrained=pretrained, **kwargs)
    if key in _HAMER_NAMES:
        return HamerViTBackbone(key, pretrained=pretrained, **kwargs)
    if key in _CSPNEXT_ARCH:
        return CSPNeXtBackbone(key, pretrained=pretrained, **kwargs)
    raise ValueError(
        f"unknown backbone name: {name!r} (supported: 'wilor', "
        f"{', '.join(sorted(_RESNET_NAMES))}, {', '.join(sorted(_VIT_NAMES))}, "
        f"{', '.join(sorted(_HAMER_NAMES))}, {', '.join(sorted(_CSPNEXT_ARCH))})"
    )
