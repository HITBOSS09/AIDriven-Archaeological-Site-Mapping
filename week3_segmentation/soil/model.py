from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def build_deeplabv3(num_classes: int = 5, pretrained_backbone: bool = False) -> nn.Module:
    """Build DeepLabV3+ style model with `num_classes` output channels.

    Uses torchvision's implementation and sets the classifier for `num_classes`.
    """
    # torchvision supports a num_classes argument directly
    model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    return model


class SimpleUNet(nn.Module):
    """A minimal U-Net-like architecture (small) as an alternative if needed."""

    def __init__(self, num_classes: int = 5, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        f = base_filters
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, f, 3, padding=1), nn.ReLU(), nn.Conv2d(f, f, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(f, f * 2, 3, padding=1), nn.ReLU(), nn.Conv2d(f * 2, f * 2, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv2d(f * 3, f, 3, padding=1), nn.ReLU(), nn.Conv2d(f, f, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        u = self.up(e2)
        cat = torch.cat([u, e1], dim=1)
        d = self.dec1(cat)
        out = self.final(d)
        return out


__all__ = ["build_deeplabv3", "SimpleUNet"]
