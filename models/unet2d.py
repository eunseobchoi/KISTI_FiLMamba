from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 2, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock2D(in_ch, base)
        self.enc2 = ConvBlock2D(base, base * 2)
        self.enc3 = ConvBlock2D(base * 2, base * 4)
        self.bottleneck = ConvBlock2D(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock2D(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock2D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock2D(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        dx = ref.shape[-2] - x.shape[-2]
        dy = ref.shape[-1] - x.shape[-1]
        x = F.pad(x, (0, max(dy, 0), 0, max(dx, 0)))
        return x[:, :, : ref.shape[-2], : ref.shape[-1]]

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2, ceil_mode=True)
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2, ceil_mode=True)
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2, ceil_mode=True)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,2,Nx,Ny,Nz] -> process each slice independently.
        bsz, ch, nx, ny, nz = x.shape
        xs = x.permute(0, 4, 1, 2, 3).reshape(bsz * nz, ch, nx, ny)
        ys = self._forward_2d(xs)
        y = ys.reshape(bsz, nz, ch, nx, ny).permute(0, 2, 3, 4, 1).contiguous()
        return y
