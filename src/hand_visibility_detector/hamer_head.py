"""Self-contained port of HaMeR's transformer-decoder MANO head.

This is the MANO regression head used by HaMeR (Pavlakos et al. 2024), ported
from ``hamer/models/heads/mano_head.py`` and
``hamer/models/components/pose_transformer.py`` so that no ``hamer`` package
needs to be importable at runtime. Only the configuration of the published
HaMeR checkpoints is kept (``JOINT_REP='6d'``, ``TRANSFORMER_INPUT='zero'``,
``IEF_ITERS=1``, ``norm='layer'``); unused branches (adaptive norm, token
dropout, the per-iteration prediction lists) are dropped. Module names match
the original, so the checkpoint's ``mano_head.``-prefixed weights (114 tensors,
including the ``init_*`` mean-parameter buffers) load with ``strict=True``.

Combined with the ViTPose-H port in
:mod:`hand_visibility_detector.backbones.hamer_vit` it forms
:class:`HamerPoseNet`, the full HaMeR model at inference time.
"""

from __future__ import annotations

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.hamer_vit import vit as build_hamer_vit


NUM_HAND_JOINTS = 15


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.
    (B, 6) -> (B, 3, 3). Identical to HaMeR's ``utils.geometry``."""
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _split_heads(t: torch.Tensor, heads: int) -> torch.Tensor:
    """``b n (h d) -> b h n d``"""
    b, n, _ = t.shape
    return t.reshape(b, n, heads, -1).transpose(1, 2)


def _merge_heads(t: torch.Tensor) -> torch.Tensor:
    """``b h n d -> b n (h d)``"""
    b, h, n, d = t.shape
    return t.transpose(1, 2).reshape(b, n, h * d)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (_split_heads(t, self.heads) for t in self.to_qkv(x).chunk(3, dim=-1))
        attn = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        return self.to_out(_merge_heads(torch.matmul(attn, v)))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int | None = None, heads: int = 8,
                 dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        context_dim = context_dim if context_dim is not None else dim
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = x if context is None else context
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = (_split_heads(t, self.heads) for t in (self.to_q(x), k, v))
        attn = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        return self.to_out(_merge_heads(torch.matmul(attn, v)))


class TransformerCrossAttn(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, dropout: float = 0.0,
                 context_dim: int | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, CrossAttention(dim, context_dim=context_dim, heads=heads,
                                            dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ])
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context=context) + x
            x = ff(x) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int, dim: int, depth: int,
                 heads: int, mlp_dim: int, dim_head: int = 64,
                 dropout: float = 0.0, context_dim: int | None = None) -> None:
        super().__init__()
        self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.transformer = TransformerCrossAttn(
            dim, depth, heads, dim_head, mlp_dim, dropout, context_dim=context_dim,
        )

    def forward(self, inp: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.to_token_embedding(inp)
        x = x + self.pos_embedding[:, : x.shape[1]]
        return self.transformer(x, context=context)


class MANOTransformerDecoderHead(nn.Module):
    """HaMeR's cross-attention MANO head (published-checkpoint configuration:
    depth 6, heads 8, dim_head 64, mlp_dim 1024, context_dim 1280).

    ``forward`` takes the ``(B, 1280, 16, 12)`` ViT feature map and returns
    ``(pred_mano_params, pred_cam)`` with rotation-matrix ``global_orient``
    ``(B, 1, 3, 3)`` / ``hand_pose`` ``(B, 15, 3, 3)``, ``betas`` ``(B, 10)``
    and the weak-perspective camera ``(B, 3)``.
    """

    def __init__(self) -> None:
        super().__init__()
        npose = 6 * (NUM_HAND_JOINTS + 1)
        dim = 1024
        self.transformer = TransformerDecoder(
            num_tokens=1, token_dim=1, dim=dim, depth=6, heads=8,
            mlp_dim=1024, dim_head=64, dropout=0.0, context_dim=1280,
        )
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)
        # Mean MANO parameters; overwritten by the checkpoint's buffers.
        self.register_buffer("init_hand_pose", torch.zeros(1, npose))
        self.register_buffer("init_betas", torch.zeros(1, 10))
        self.register_buffer("init_cam", torch.zeros(1, 3))

    def forward(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch_size = x.shape[0]
        # ViT feature map is channel-first; change to token-first.
        x = x.flatten(2).transpose(1, 2)
        token = torch.zeros(batch_size, 1, 1, device=x.device, dtype=x.dtype)
        token_out = self.transformer(token, context=x).squeeze(1)

        pred_hand_pose = self.decpose(token_out) + self.init_hand_pose
        pred_betas = self.decshape(token_out) + self.init_betas
        pred_cam = self.deccam(token_out) + self.init_cam

        pred_hand_pose = rot6d_to_rotmat(pred_hand_pose).view(
            batch_size, NUM_HAND_JOINTS + 1, 3, 3
        )
        pred_mano_params = {
            "global_orient": pred_hand_pose[:, [0]],
            "hand_pose": pred_hand_pose[:, 1:],
            "betas": pred_betas,
        }
        return pred_mano_params, pred_cam


class HamerPoseNet(nn.Module):
    """HaMeR at inference time: ViTPose-H backbone + MANO head + MANO layer.

    Mirrors ``HAMER.forward_step`` but takes the same input as ``wilor_mini``'s
    ``WiLor`` -- ``(B, 256, 256, 3)`` crops with values in ``[0, 255]`` -- and
    returns a dict with the same schema: axis-angle ``global_orient``
    ``(B, 1, 3)`` / ``hand_pose`` ``(B, 15, 3)``, ``betas``, ``pred_cam``,
    ``pred_keypoints_3d`` ``(B, 21, 3)`` (OpenPose order) and
    ``pred_vertices`` ``(B, 778, 3)``.

    Unlike ``WiLor`` (which flips the channel order of its input), the crops
    are expected in RGB, as HaMeR's own ``ViTDetDataset`` feeds them.
    """

    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(self, mano_model_path: str) -> None:
        super().__init__()
        from wilor_mini.models.mano_wrapper import MANO

        self.backbone = build_hamer_vit()
        self.mano_head = MANOTransformerDecoderHead()
        # HaMeR's MANO wrapper is identical to WiLoR's (same OpenPose joint map).
        self.mano = MANO(model_path=mano_model_path, create_body_pose=False)
        self.register_buffer(
            "image_mean", torch.tensor(self.IMAGE_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(self.IMAGE_STD).view(1, 3, 1, 1), persistent=False
        )

    def load_hamer_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load the ``backbone.`` / ``mano_head.`` weights of a HaMeR Lightning
        checkpoint ``state_dict`` (strict). The MANO layer keeps the buffers
        built from ``MANO_RIGHT.pkl``."""
        for prefix in ("backbone", "mano_head"):
            sub = {
                k[len(prefix) + 1:]: v
                for k, v in state_dict.items()
                if k.startswith(prefix + ".")
            }
            if not sub:
                raise RuntimeError(f"no '{prefix}.'-prefixed weights in the HaMeR checkpoint")
            getattr(self, prefix).load_state_dict(sub, strict=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.permute(0, 3, 1, 2) / 255.0
        x = (x - self.image_mean.to(x.dtype)) / self.image_std.to(x.dtype)
        batch_size = x.shape[0]
        # HaMeR's ViT was trained on (H=256, W=192) crops.
        feats = self.backbone(x[:, :, :, 32:-32])
        pred_mano_params, pred_cam = self.mano_head(feats)
        mano_output = self.mano(**pred_mano_params, pose2rot=False)
        return {
            "global_orient": roma.rotmat_to_rotvec(pred_mano_params["global_orient"]),
            "hand_pose": roma.rotmat_to_rotvec(pred_mano_params["hand_pose"]),
            "betas": pred_mano_params["betas"],
            "pred_cam": pred_cam,
            "pred_keypoints_3d": mano_output.joints.reshape(batch_size, -1, 3),
            "pred_vertices": mano_output.vertices.reshape(batch_size, -1, 3),
        }
