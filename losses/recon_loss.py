from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from filmamba.utils.fft import fft3c
from filmamba.utils.metrics import ssim3d


class HFEN3D(nn.Module):
    def __init__(self, sigma: float = 1.5, kernel_size: int = 15):
        super().__init__()
        ax = torch.arange(kernel_size).float() - kernel_size // 2
        xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")
        rr2 = xx ** 2 + yy ** 2 + zz ** 2
        g = torch.exp(-rr2 / (2 * sigma ** 2))
        g = g / g.sum()
        log = (rr2 - 3 * sigma ** 2) / (sigma ** 4) * g
        log = log - log.mean()
        self.register_buffer("kernel", log.view(1, 1, kernel_size, kernel_size, kernel_size))
        self.pad = kernel_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        kernel = self.kernel.to(device=pred.device, dtype=pred.dtype)
        fp = F.conv3d(pred, kernel, padding=self.pad)
        ft = F.conv3d(target, kernel, padding=self.pad)
        return F.l1_loss(fp, ft)


@dataclass
class LossConfig:
    lambda_k: float = 0.3
    lambda_h: float = 0.1
    lambda_s: float = 0.15
    deep_supervision: float = 0.3
    num_bands: int = 8


class FiLMambaLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.hfen = HFEN3D()

    def _band_masks(self, spatial_shape: Tuple[int, int, int], device) -> List[torch.Tensor]:
        nx, ny, nz = spatial_shape
        gx = torch.arange(nx, device=device).float() - (nx - 1) / 2
        gy = torch.arange(ny, device=device).float() - (ny - 1) / 2
        gz = torch.arange(nz, device=device).float() - (nz - 1) / 2
        xx, yy, zz = torch.meshgrid(gx, gy, gz, indexing="ij")
        rr = torch.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
        rr = rr / rr.max().clamp_min(1e-6)
        masks = []
        for b in range(self.cfg.num_bands):
            lo = b / self.cfg.num_bands
            hi = (b + 1) / self.cfg.num_bands
            m = ((rr >= lo) & (rr < hi)).float()
            if b == self.cfg.num_bands - 1:
                m = ((rr >= lo) & (rr <= hi)).float()
            masks.append(m)
        return masks

    def _freq_stratified(self, pred_c: torch.Tensor, target_c: torch.Tensor) -> torch.Tensor:
        # pred_c/target_c: [B,X,Y,Z] complex
        k_pred = fft3c(pred_c)
        k_tgt = fft3c(target_c)
        masks = self._band_masks(pred_c.shape[-3:], pred_c.device)
        loss = 0.0
        for m in masks:
            m = m[None, ...]
            num = (m * (k_pred - k_tgt).abs()).sum(dim=(-3, -2, -1))
            den = (m * k_tgt.abs()).sum(dim=(-3, -2, -1)).clamp_min(1e-6)
            loss = loss + (num / den).mean()
        return loss / len(masks)

    def forward(
        self,
        pred_mag: torch.Tensor,
        target_mag: torch.Tensor,
        pred_complex: torch.Tensor,
        target_complex: torch.Tensor,
        intermediate: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        l_img = F.l1_loss(pred_mag, target_mag)
        l_k = self._freq_stratified(pred_complex, target_complex)
        l_h = self.hfen(pred_mag, target_mag)
        l_s = 1.0 - ssim3d(pred_mag, target_mag, data_range=1.0)

        l_ds = torch.tensor(0.0, device=pred_mag.device)
        if intermediate:
            terms = []
            for x in intermediate:
                x_mag = x.abs() if torch.is_complex(x) else x
                terms.append(F.l1_loss(x_mag.float(), target_mag))
            l_ds = torch.stack(terms).mean()

        total = l_img + self.cfg.lambda_k * l_k + self.cfg.lambda_h * l_h + self.cfg.lambda_s * l_s
        if intermediate:
            total = total + self.cfg.deep_supervision * l_ds

        return {
            "total": total,
            "img": l_img,
            "kspace": l_k,
            "hfen": l_h,
            "ssim": l_s,
            "deep_supervision": l_ds,
        }
