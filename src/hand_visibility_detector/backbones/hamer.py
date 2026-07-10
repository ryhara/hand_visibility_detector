"""HaMeR ViTPose-H feature-map backbone.

See :mod:`hand_visibility_detector.backbones` for the shared backbone contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# HaMeR ViTPose-H backbone. The published ``hamer.ckpt`` is a Lightning
# checkpoint whose backbone weights live under the ``backbone.`` prefix. It is
# resolved from a local ``.pth``/``.ckpt`` (the ``weights`` arg, the
# ``HAMER_WEIGHTS`` env var, or ``<project_root>/external/hamer_weights/hamer.ckpt``)
# and otherwise auto-downloaded from the official HaMeR demo Space on
# HuggingFace. feat_dim 1280, output map ``(B, 1280, 16, 12)`` for a 256x192
# crop -- the same shape WiLoR yields.
_HAMER_NAMES = {"hamer"}
_HAMER_FEAT_DIM = 1280


def _resolve_hamer_weights(weights: str | None) -> str:
    """Resolve the HaMeR checkpoint path.

    ``weights`` overrides everything; otherwise the ``HAMER_WEIGHTS`` env var,
    then ``<project_root>/external/hamer_weights/hamer.ckpt`` (``project_root`` =
    three parents up from this file, i.e. a repo checkout). If none of those
    exist, the checkpoint is auto-downloaded from the official HaMeR demo Space
    on HuggingFace (``geopavlakos/HaMeR``) and cached.
    """
    import os
    from pathlib import Path

    if weights is not None:
        return weights
    env = os.environ.get("HAMER_WEIGHTS")
    if env:
        return env
    project_root = Path(__file__).resolve().parents[3]
    local = project_root / "external" / "hamer_weights" / "hamer.ckpt"
    if local.exists():
        return str(local)
    from ..hub import download_hamer_backbone

    return download_hamer_backbone()


class HamerViTBackbone(nn.Module):
    """HaMeR's ViTPose-H backbone returning a ``(B, 1280, 16, 12)`` feature map.

    The model code is the self-contained port in
    :mod:`hand_visibility_detector.backbones.hamer_vit` (no ``hamer`` package
    needed). The
    pre-trained weights are read from the published ``hamer.ckpt`` (a Lightning
    checkpoint): its ``state_dict`` keys under the ``backbone.`` prefix (389
    tensors) load into the ported ViT with ``strict=True``.

    HaMeR was trained on ``(H=256, W=192)`` crops. When a square ``256x256``
    tensor is fed in, we centre-crop to ``256x192`` via ``x[:, :, :, 32:-32]``
    -- exactly as the WiLoR backbone wrapper does -- so the pre-trained
    positional embeddings stay valid. ``feat_dim`` is 1280.
    """

    def __init__(
        self,
        name: str = "hamer",
        pretrained: bool = True,
        weights: str | None = None,
    ) -> None:
        super().__init__()
        from .hamer_vit import vit as build_hamer_vit

        self.vit = build_hamer_vit()
        self.feat_dim = _HAMER_FEAT_DIM
        if pretrained:
            self._load_pretrained(_resolve_hamer_weights(weights))

    def _load_pretrained(self, weights: str) -> None:
        import os

        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"HaMeR checkpoint not found: {weights!r}. Copy the published "
                "hamer.ckpt there (or set HAMER_WEIGHTS / pass weights=...)."
            )
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        bb_state = {
            k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        if not bb_state:
            raise RuntimeError(
                f"no 'backbone.'-prefixed weights found in {weights!r}"
            )
        self.vit.load_state_dict(bb_state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 256:
            x = x[:, :, :, 32:-32]
        return self.vit(x)
