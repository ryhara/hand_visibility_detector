"""HandVisibilityNet: Contact4D-style per-keypoint visibility classifier.

Architecture (Contact4D):
    WiLoR ViT backbone -> conv -> FC -> GAU -> conv -> per-keypoint logits

Input:  (B, 3, H, W) ImageNet-normalized tensor
Output: (B, 21)      per-keypoint visibility logits
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_HAND_KEYPOINTS = 21
WILOR_FEAT_DIM = 1280


# ---------------------------------------------------------------------------
# WiLoR ViT backbone wrapper
# ---------------------------------------------------------------------------


class _WiLoRBackboneWrapper(nn.Module):
    """Wrap the WiLoR ViT backbone so it returns only the spatial feature
    map ``(B, 1280, 16, 12)`` expected by :class:`VisibilityHead`.

    The WiLoR ViT was trained on ``(H=256, W=192)`` crops. When a square
    ``256x256`` tensor is fed in, we centre-crop to ``256x192`` via
    ``x[:, :, :, 32:-32]``.
    """

    def __init__(self, raw_backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = raw_backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 256:
            x = x[:, :, :, 32:-32]
        _, _, _, vit_out = self.backbone(x)
        return vit_out


# ---------------------------------------------------------------------------
# Gated Attention Unit (GAU)
# ---------------------------------------------------------------------------


class GAU(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, s: int = 128) -> None:
        super().__init__()
        self.s = s
        self.e = expansion * dim
        self.uv_proj = nn.Linear(dim, 2 * self.e + s)
        self.gamma = nn.Parameter(torch.randn(2, s) * 0.02)
        self.beta = nn.Parameter(torch.zeros(2, s))
        self.out_proj = nn.Linear(self.e, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        uv = self.uv_proj(x)
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)
        u = self.act(u)
        v = self.act(v)
        q = base * self.gamma[0] + self.beta[0]
        k = base * self.gamma[1] + self.beta[1]
        qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.s)
        kernel = F.relu(qk) ** 2
        out = torch.matmul(kernel, v)
        out = self.out_proj(u * out)
        return shortcut + out


# ---------------------------------------------------------------------------
# Visibility Head
# ---------------------------------------------------------------------------


class VisibilityHead(nn.Module):
    def __init__(
        self,
        in_channels: int = WILOR_FEAT_DIM,
        hidden_dim: int = 256,
        num_keypoints: int = NUM_HAND_KEYPOINTS,
        gau_s: int = 128,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.gau = GAU(hidden_dim, expansion=2, s=gau_s)
        self.conv_out = nn.Conv2d(hidden_dim, num_keypoints, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(feat)
        B, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.fc(tokens)
        tokens = self.gau(tokens)
        x = tokens.transpose(1, 2).reshape(B, D, H, W)
        x = self.conv_out(x)
        logits = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return logits


# ---------------------------------------------------------------------------
# HandVisibilityNet
# ---------------------------------------------------------------------------


class HandVisibilityNet(nn.Module):
    """Contact4D-style hand-keypoint visibility classifier (WiLoR backbone).

    Pass a raw WiLoR ViT backbone (from ``wilor_mini``) and this module wraps
    it with the RTMPose-style visibility head. Use
    :meth:`from_wilor_backbone` for inference with a pre-trained head.
    """

    def __init__(
        self,
        raw_backbone: nn.Module,
        dropout: float = 0.0,
        hidden_dim: int = 256,
        num_keypoints: int = NUM_HAND_KEYPOINTS,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = _WiLoRBackboneWrapper(raw_backbone)
        self.head = VisibilityHead(
            in_channels=WILOR_FEAT_DIM,
            hidden_dim=hidden_dim,
            num_keypoints=num_keypoints,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.num_keypoints = num_keypoints
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_wilor_backbone(
        cls,
        raw_backbone: nn.Module,
        head_state_dict: dict,
        hidden_dim: int = 256,
        num_keypoints: int = NUM_HAND_KEYPOINTS,
    ) -> "HandVisibilityNet":
        """Build a frozen-backbone model and load the pre-trained head."""
        model = cls(
            raw_backbone=raw_backbone,
            hidden_dim=hidden_dim,
            num_keypoints=num_keypoints,
            freeze_backbone=True,
        )
        model.head.load_state_dict(head_state_dict, strict=True)
        return model

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.head(feat)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def head_state_dict(self) -> dict:
        return self.head.state_dict()

    def load_head_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        self.head.load_state_dict(state_dict, strict=strict)
