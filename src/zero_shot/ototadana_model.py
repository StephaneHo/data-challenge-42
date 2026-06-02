"""Rebuild the ototadana DeepLabV3+ R101 face-occlusion model without mmsegmentation.

Architecture from inspecting the checkpoint:
  - Backbone: ResNet-101 V1c (deep stem: 3 conv3x3, not 7x7) — matches timm 'resnet101d'.
  - Decode head: DepthwiseSeparableASPPHead from mmseg.
  - 2 output classes (background, occlusion).

This module re-implements the head verbatim from the checkpoint shapes, so
the state_dict loads cleanly via a name-mapping.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Common building blocks -----
class ConvBN(nn.Module):
    """Conv + BN + ReLU module matching mmcv's ConvModule layout (conv, bn)."""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DepthwiseSeparableConv(nn.Module):
    """DW conv (3x3, groups=in) + PW conv (1x1)."""

    def __init__(self, in_ch, out_ch, dilation=1, padding=None):
        super().__init__()
        if padding is None:
            padding = dilation
        self.depthwise_conv = ConvBN(in_ch, in_ch, kernel_size=3, padding=padding, dilation=dilation, groups=in_ch)
        self.pointwise_conv = ConvBN(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


# ----- ASPP head (mmseg's DepthwiseSeparableASPPHead) -----
class DepthwiseSeparableASPPHead(nn.Module):
    """ASPP head with depthwise-separable convs in the dilated branches.

    Layout (matches checkpoint):
      image_pool:    AAP + ConvBN(2048→512, 1x1)
      aspp_modules:
          [0]: ConvBN(2048→512, 1x1, dilation=1)              # vanilla conv1x1
          [1]: DepthwiseSeparableConv(2048→512, dilation=12)
          [2]: DepthwiseSeparableConv(2048→512, dilation=24)
          [3]: DepthwiseSeparableConv(2048→512, dilation=36)
      bottleneck:    ConvBN(2560→512, 3x3, padding=1)
      c1_bottleneck: ConvBN(256→48, 1x1)
      sep_bottleneck:
          [0]: DepthwiseSeparableConv(560→512, dilation=1)
          [1]: DepthwiseSeparableConv(512→512, dilation=1)
      conv_seg:      Conv2d(512→num_classes, 1x1, bias=True)
    """

    def __init__(self, in_channels=2048, channels=512, c1_in_channels=256, c1_channels=48,
                 dilations=(1, 12, 24, 36), num_classes=2):
        super().__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBN(in_channels, channels, kernel_size=1),
        )
        self.aspp_modules = nn.ModuleList()
        # [0] vanilla 1x1 (dilation=1)
        self.aspp_modules.append(ConvBN(in_channels, channels, kernel_size=1))
        # [1, 2, 3] depthwise-separable
        for d in dilations[1:]:
            self.aspp_modules.append(DepthwiseSeparableConv(in_channels, channels, dilation=d))

        # bottleneck after concat (5 feats × 512 = 2560)
        self.bottleneck = ConvBN(channels * 5, channels, kernel_size=3, padding=1)

        # low-level (c1) projection
        self.c1_bottleneck = ConvBN(c1_in_channels, c1_channels, kernel_size=1)

        # sep_bottleneck (decode after concat with c1)
        self.sep_bottleneck = nn.ModuleList([
            DepthwiseSeparableConv(channels + c1_channels, channels),
            DepthwiseSeparableConv(channels, channels),
        ])

        self.dropout = nn.Dropout2d(p=0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1, bias=True)

    def forward(self, c1_feat, c4_feat):
        # c4_feat: (B, 2048, H/8, W/8)
        # c1_feat: (B, 256,  H/4, W/4)  (low-level skip)
        size_high = c4_feat.shape[-2:]
        # ASPP image_pool
        img = self.image_pool(c4_feat)
        img = F.interpolate(img, size=size_high, mode="bilinear", align_corners=False)
        feats = [img]
        for m in self.aspp_modules:
            feats.append(m(c4_feat))
        aspp_out = self.bottleneck(torch.cat(feats, dim=1))

        # Upsample aspp_out to c1 size
        c1_size = c1_feat.shape[-2:]
        aspp_up = F.interpolate(aspp_out, size=c1_size, mode="bilinear", align_corners=False)
        c1_proj = self.c1_bottleneck(c1_feat)
        x = torch.cat([aspp_up, c1_proj], dim=1)
        x = self.sep_bottleneck[0](x)
        x = self.sep_bottleneck[1](x)
        x = self.dropout(x)
        return self.conv_seg(x)


# ----- Full model -----
class OtotadanaFaceOcclusion(nn.Module):
    """Full DeepLabV3+ R101 V1c face-occlusion model.

    Returns logits of shape (B, 2, H, W) at INPUT resolution.
    Class 0 = background/face (visible), class 1 = occlusion (mmseg convention; verify on samples).
    """

    def __init__(self):
        super().__init__()
        from timm.models import resnet
        # ResNet-101 V1c = deep stem (3 conv 3x3 with stem_width=32, output_chs=64),
        # but standard downsample (1x1 conv + BN — NOT the avgpool variant).
        # features_only=True returns intermediate feature maps at multiple scales.
        self.backbone = resnet.resnet101(
            stem_type="deep",
            stem_width=32,
            avg_down=False,
            num_classes=0,
            features_only=True,
            out_indices=(1, 4),  # c1 (after layer1: 256 chs), c4 (after layer4: 2048 chs)
            output_stride=8,     # mmseg default for DeepLabV3+ — dilated layer3/4
        )
        self.head = DepthwiseSeparableASPPHead()

    def forward(self, x):
        h, w = x.shape[-2:]
        c1, c4 = self.backbone(x)
        logits = self.head(c1, c4)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


# ----- Weight loading -----
def _backbone_key_map(k):
    """Map a 'backbone.*' checkpoint key to the corresponding timm key."""
    # Strip the 'backbone.' prefix
    k = k[len("backbone."):]
    # stem.[0..7] in checkpoint maps to:
    #   stem.0 → conv1.0    (conv)
    #   stem.1 → conv1.1    (bn)
    #   stem.3 → conv1.3    (conv)
    #   stem.4 → conv1.4    (bn)
    #   stem.6 → conv1.6    (conv)
    #   stem.7 → bn1        (bn — different name!)
    if k.startswith("stem."):
        idx = k[len("stem."):].split(".", 1)[0]
        rest = k[len(f"stem.{idx}."):]
        if idx == "7":
            return f"bn1.{rest}"
        return f"conv1.{idx}.{rest}"
    return k


def load_ototadana_weights(model: OtotadanaFaceOcclusion, ckpt_path: str, verbose: bool = True):
    """Load the ototadana checkpoint into our custom model.

    Strict loading is impossible because the head naming differs slightly (mmcv ConvModule
    vs our ConvBN). We map keys explicitly and verify shape compatibility.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    target_sd = model.state_dict()
    new_sd = {}
    missing_keys = []
    unmapped_keys = []

    for k, v in sd.items():
        if k.startswith("backbone."):
            target_key = "backbone." + _backbone_key_map(k)
        elif k.startswith("decode_head."):
            # decode_head.* → head.*
            target_key = "head." + k[len("decode_head."):]
        elif k.startswith("auxiliary_head."):
            # We skip the auxiliary head — only used during training
            continue
        else:
            target_key = k

        if target_key in target_sd:
            if target_sd[target_key].shape == v.shape:
                new_sd[target_key] = v
            else:
                unmapped_keys.append((k, target_key, v.shape, target_sd[target_key].shape))
        else:
            unmapped_keys.append((k, target_key, v.shape, None))

    for k in target_sd:
        if k not in new_sd:
            missing_keys.append(k)

    if verbose:
        print(f"loaded {len(new_sd)} / {len(target_sd)} parameters")
        if missing_keys:
            print(f"  {len(missing_keys)} missing in checkpoint:")
            for k in missing_keys[:10]:
                print(f"    {k}")
        if unmapped_keys:
            print(f"  {len(unmapped_keys)} unmapped:")
            for orig, target, src_shape, tgt_shape in unmapped_keys[:10]:
                print(f"    {orig} → {target}  ({src_shape} vs {tgt_shape})")

    res = model.load_state_dict(new_sd, strict=False)
    if verbose:
        print(f"  load_state_dict missing: {len(res.missing_keys)}, unexpected: {len(res.unexpected_keys)}")
    return res
