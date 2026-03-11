from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class SFOS(nn.Module):
    """Spherical Frequency-Ordered Scanning metadata and permutation."""

    def __init__(self, grid_size: Tuple[int, int, int], num_bands: int):
        super().__init__()
        self.grid_size = tuple(int(v) for v in grid_size)
        self.num_bands = int(num_bands)

        forward, inverse, band, rho_bar, band_to_idx = self._build()
        self.register_buffer("forward_order", forward, persistent=False)
        self.register_buffer("inverse_order", inverse, persistent=False)
        self.register_buffer("band_indices", band, persistent=False)
        self.register_buffer("rho_bar", rho_bar, persistent=False)
        self.band_to_token_indices: List[torch.Tensor] = band_to_idx

    def _build(self):
        g1, g2, g3 = self.grid_size
        center = torch.tensor([(g1 - 1) / 2.0, (g2 - 1) / 2.0, (g3 - 1) / 2.0], dtype=torch.float32)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(g1, dtype=torch.float32),
                torch.arange(g2, dtype=torch.float32),
                torch.arange(g3, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)

        rho = torch.linalg.norm(coords - center, dim=-1)
        forward = torch.argsort(rho, stable=True)
        inverse = torch.empty_like(forward)
        inverse[forward] = torch.arange(forward.numel(), dtype=torch.long)

        rho_sorted = rho[forward]
        rho_bar = rho_sorted / rho_sorted.max().clamp_min(1e-8)

        # b_j = min(B-1, floor(rho_bar * B))
        band = torch.clamp(torch.floor(rho_bar * self.num_bands).long(), max=self.num_bands - 1)
        band_to_idx = [torch.where(band == b)[0] for b in range(self.num_bands)]
        return forward, inverse, band, rho_bar, band_to_idx

    def forward(self, x: torch.Tensor):
        # x: [B,L,D] in raster order -> [B,L,D] in SFOS order
        return x[:, self.forward_order, :], self.band_indices, self.rho_bar

    def inverse(self, x: torch.Tensor):
        return x[:, self.inverse_order, :]
