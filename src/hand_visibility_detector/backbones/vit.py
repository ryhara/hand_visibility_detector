"""torchvision Vision Transformer feature-map backbone.

See :mod:`hand_visibility_detector.backbones` for the shared backbone contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_VIT_NAMES = {"vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"}
_VIT_ALIASES = {"vit_b": "vit_b_16", "vit_l": "vit_l_16", "vit_h": "vit_h_14"}


class ViTBackbone(nn.Module):
    """A torchvision Vision Transformer returning the patch tokens as a
    ``(B, feat_dim, grid, grid)`` map.

    The class token is dropped and the remaining patch tokens are reshaped back
    to the patch grid. Input is resized to the model's native ``image_size`` so
    the pre-trained positional embeddings stay valid (and unchanged).
    ``feat_dim`` / ``input_size`` / ``patch_size`` are all read from the model.
    """

    def __init__(self, name: str = "vit_b_16", pretrained: bool = True) -> None:
        super().__init__()
        import torchvision

        # ``vit_h_14``'s DEFAULT weights are SWAG_E2E (native 518x518), which
        # blows the patch grid up to 37x37 = 1369 tokens and makes the frozen
        # forward pass ~28x heavier than necessary. Use the 224-native linear
        # weights instead so attention stays at a 16x16 grid.
        if pretrained and name == "vit_h_14":
            weights = "IMAGENET1K_SWAG_LINEAR_V1"
        else:
            weights = "DEFAULT" if pretrained else None
        m = torchvision.models.get_model(name, weights=weights)
        self.vit = m
        self.feat_dim = int(m.hidden_dim)
        self.input_size = int(m.image_size)
        self.patch_size = int(m.patch_size)
        grid = self.input_size // self.patch_size
        self._grid = (grid, grid)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize(x)
        gh, gw = self._grid
        m = self.vit
        tokens = m._process_input(x)  # (B, gh*gw, feat_dim)
        cls = m.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = m.encoder(tokens)
        tokens = tokens[:, 1:, :]  # drop class token -> (B, gh*gw, feat_dim)
        B, N, D = tokens.shape
        return tokens.transpose(1, 2).reshape(B, D, gh, gw)
