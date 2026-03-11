from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(ch),
            nn.GELU(),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class LRS(nn.Module):
    def __init__(self, hidden: int = 16, num_blocks: int = 3, max_scale: float = 0.3):
        super().__init__()
        self.max_scale = max_scale
        self.stem = nn.Sequential(
            nn.Conv3d(2, hidden, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock3D(hidden) for _ in range(num_blocks)])
        self.res_head = nn.Conv3d(hidden, 2, 3, padding=1)
        self.scale_head = nn.Conv3d(hidden, 1, 3, padding=1)
        # Keep LRS identity at init; stage-2 fine-tuning then learns residual scaling safely.
        nn.init.zeros_(self.res_head.weight)
        if self.res_head.bias is not None:
            nn.init.zeros_(self.res_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        if self.scale_head.bias is not None:
            nn.init.constant_(self.scale_head.bias, -8.0)

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        x = torch.stack([x_complex.real, x_complex.imag], dim=1)
        h = self.blocks(self.stem(x))
        res = self.res_head(h)
        scale = torch.sigmoid(self.scale_head(h)) * self.max_scale
        y = x + scale * res
        return torch.complex(y[:, 0], y[:, 1])
