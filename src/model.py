"""Backbone factory wrapping torchvision models for [0, 1] regression."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class OcclusionRegressor(nn.Module):
    """Wraps a backbone whose last layer outputs 1 logit, then applies sigmoid."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(x)).view(-1)


def _replace_last_linear(module: nn.Module, in_features: int) -> nn.Module:
    return nn.Linear(in_features, 1)


def build_model(name: str = "mobilenet_v3_small", pretrained: bool = True) -> OcclusionRegressor:
    name = name.lower()
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=weights)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 1)
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v3_large(weights=weights)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 1)
    elif name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 1)
    else:
        raise ValueError(f"Unknown model: {name}")
    return OcclusionRegressor(m)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
