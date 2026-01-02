"""
U-Net Model for Binary Vegetation Segmentation
==============================================

Standard encoder-decoder U-Net with skip connections and a single-channel
output (logits for vegetation vs background).

Assumes input spatial dimensions are multiples of 8 (true for the 256x256
preprocessed images/masks), so encoder/decoder feature maps align cleanly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UNet(nn.Module):
    """Simple U-Net for binary segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = self.conv_block(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with skip connections
        up3 = self.upconv3(b)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))
        up2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        out = self.final_conv(d1)
        return out


