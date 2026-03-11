from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 2, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv3d(base, out_ch, 1)
        # Start from identity residual behavior to avoid corrupting zero-filled initialization.
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def _match(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] == ref.shape[-3:]:
            return x
        dx = ref.shape[-3] - x.shape[-3]
        dy = ref.shape[-2] - x.shape[-2]
        dz = ref.shape[-1] - x.shape[-1]
        pad = (0, max(dz, 0), 0, max(dy, 0), 0, max(dx, 0))
        x = F.pad(x, pad)
        return x[:, :, : ref.shape[-3], : ref.shape[-2], : ref.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = F.max_pool3d(e1, 2, ceil_mode=True)
        e2 = self.enc2(p1)
        p2 = F.max_pool3d(e2, 2, ceil_mode=True)
        e3 = self.enc3(p2)
        p3 = F.max_pool3d(e3, 2, ceil_mode=True)
        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = self._match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return x + self.out(d1)
