"""CSPNeXt (RTMDet) feature-map backbone.

Pure-PyTorch re-implementation of MMDetection's ``CSPNeXt`` whose submodule and
parameter names match the published checkpoint exactly, so the OpenMMLab
ImageNet pre-trained weights load with ``strict=True`` after stripping the
``backbone.`` prefix and dropping the classification ``head.``. No ``mmdet`` /
``mmcv`` dependency is needed.

See :mod:`hand_visibility_detector.backbones` for the shared backbone contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ``deepen_factor`` (block-count multiplier) and ``widen_factor`` (channel
# multiplier) per variant, plus the OpenMMLab ImageNet-pretrained checkpoint URL.
_CSPNEXT_ARCH = {
    "cspnext_tiny": (0.167, 0.375),
    "cspnext_s": (0.33, 0.5),
    "cspnext_m": (0.67, 0.75),
    "cspnext_l": (1.0, 1.0),
    "cspnext_x": (1.33, 1.25),
}
_CSPNEXT_ALIASES = {"cspnext": "cspnext_m"}
_CSPNEXT_URLS = {
    "cspnext_tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth",
    "cspnext_s": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth",
    "cspnext_m": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth",
    "cspnext_l": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth",
    "cspnext_x": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-x_8xb256-rsb-a1-600e_in1k-b3f78edd.pth",
}

# CSPNeXt P5 base arch: (in_ch, out_ch, num_blocks, add_identity, use_spp) before
# applying the widen/deepen factors. Channels are scaled by ``widen_factor`` and
# block counts by ``deepen_factor`` (matching MMDetection's ``CSPNeXt``).
_CSPNEXT_STAGES = (
    (64, 128, 3, True, False),
    (128, 256, 6, True, False),
    (256, 512, 6, True, False),
    (512, 1024, 3, False, True),
)


class _ConvModule(nn.Module):
    """Conv2d (no bias) + BN + activation -- MMCV's ``ConvModule`` layout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.activate = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class _DepthwiseSeparableConvModule(nn.Module):
    """Depthwise ``k x k`` conv followed by pointwise ``1 x 1`` conv, each with
    BN + activation -- MMCV's ``DepthwiseSeparableConvModule`` layout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise_conv = _ConvModule(
            in_ch, in_ch, kernel_size, stride=1,
            padding=kernel_size // 2, groups=in_ch,
        )
        self.pointwise_conv = _ConvModule(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class _ChannelAttention(nn.Module):
    """Squeeze-and-excite style attention used by RTMDet's CSPLayer."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.global_avgpool(x)))


class _CSPNeXtBlock(nn.Module):
    """``3x3`` conv then ``5x5`` depthwise-separable conv, optional residual."""

    def __init__(self, in_ch: int, out_ch: int, add_identity: bool) -> None:
        super().__init__()
        self.conv1 = _ConvModule(in_ch, out_ch, 3, stride=1, padding=1)
        self.conv2 = _DepthwiseSeparableConvModule(out_ch, out_ch, 5)
        self.add_identity = add_identity and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class _CSPLayer(nn.Module):
    """CSPNeXt cross-stage-partial layer (with channel attention)."""

    def __init__(
        self, in_ch: int, out_ch: int, num_blocks: int, add_identity: bool
    ) -> None:
        super().__init__()
        mid = out_ch // 2
        self.main_conv = _ConvModule(in_ch, mid, 1)
        self.short_conv = _ConvModule(in_ch, mid, 1)
        self.final_conv = _ConvModule(2 * mid, out_ch, 1)
        self.blocks = nn.Sequential(
            *[_CSPNeXtBlock(mid, mid, add_identity) for _ in range(num_blocks)]
        )
        self.attention = _ChannelAttention(2 * mid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(self.attention(x_final))


class _SPPBottleneck(nn.Module):
    """Spatial pyramid pooling with parallel ``5/9/13`` max-pools."""

    def __init__(self, in_ch: int, out_ch: int, kernel_sizes=(5, 9, 13)) -> None:
        super().__init__()
        mid = in_ch // 2
        self.conv1 = _ConvModule(in_ch, mid, 1)
        self.poolings = nn.ModuleList(
            [nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernel_sizes]
        )
        self.conv2 = _ConvModule(mid * (len(kernel_sizes) + 1), out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)


def _round_blocks(num_blocks: int, deepen_factor: float) -> int:
    return max(round(num_blocks * deepen_factor), 1)


class CSPNeXtBackbone(nn.Module):
    """RTMDet CSPNeXt backbone returning the final stage feature map.

    For WiLoR's 256 crop the output is ``(B, feat_dim, 8, 8)`` (the network
    downsamples by 32). ``feat_dim`` is the last stage's channel count
    (768 for the ``m`` variant).
    """

    def __init__(self, name: str = "cspnext_m", pretrained: bool = True) -> None:
        super().__init__()
        deepen, widen = _CSPNEXT_ARCH[name]

        stem_ch = int(_CSPNEXT_STAGES[0][0] * widen)
        self.stem = nn.Sequential(
            _ConvModule(3, stem_ch // 2, 3, stride=2, padding=1),
            _ConvModule(stem_ch // 2, stem_ch // 2, 3, stride=1, padding=1),
            _ConvModule(stem_ch // 2, stem_ch, 3, stride=1, padding=1),
        )

        for i, (base_in, base_out, n_blk, add_id, use_spp) in enumerate(
            _CSPNEXT_STAGES, start=1
        ):
            in_ch = int(base_in * widen)
            out_ch = int(base_out * widen)
            layers: list[nn.Module] = [
                _ConvModule(in_ch, out_ch, 3, stride=2, padding=1)
            ]
            if use_spp:
                layers.append(_SPPBottleneck(out_ch, out_ch))
            layers.append(
                _CSPLayer(out_ch, out_ch, _round_blocks(n_blk, deepen), add_id)
            )
            setattr(self, f"stage{i}", nn.Sequential(*layers))

        self.feat_dim = int(_CSPNEXT_STAGES[-1][1] * widen)

        if pretrained:
            self._load_pretrained(name)

    def _load_pretrained(self, name: str) -> None:
        ckpt = torch.hub.load_state_dict_from_url(
            _CSPNEXT_URLS[name], map_location="cpu", check_hash=True
        )
        state = ckpt.get("state_dict", ckpt)
        # Keep backbone weights only; strip the ``backbone.`` prefix.
        bb_state = {
            k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = self.load_state_dict(bb_state, strict=False)
        # ``num_batches_tracked`` buffers are the only acceptable mismatch.
        bad_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"CSPNeXt pretrained load mismatch for {name!r}: "
                f"missing={bad_missing}, unexpected={unexpected}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
