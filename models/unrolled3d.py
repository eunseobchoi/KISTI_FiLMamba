from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from filmamba.models.lrs import LRS
from filmamba.models.ops import adjoint_op, data_grad, forward_op, soft_dc
from filmamba.models.unet3d import UNet3D


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    return float(x + math.log(-math.expm1(-x)))


@dataclass
class StrongUnrolled3DConfig:
    model_family: str = "unrolled3d_strong"
    num_coils: int = 12
    num_cascades: int = 8
    kspace_unet_base: int = 38
    image_unet_base: int = 32
    use_lrs: bool = True
    lrs_hidden: int = 16
    lrs_blocks: int = 3
    lrs_max_scale: float = 0.3
    gradient_checkpointing: bool = True
    eta_init: float = 0.1
    mu_init: float = 0.01


class StrongUnrolled3DCascade(nn.Module):
    """
    3D physics-guided unrolled baseline.
    Replaces the k-space SSM branch with native 3D U-Nets while
    keeping the same gradient step + soft-DC scaffold for fairness.
    """

    def __init__(self, cfg: StrongUnrolled3DConfig):
        super().__init__()
        self.kspace_block = UNet3D(
            in_ch=2 * int(cfg.num_coils),
            out_ch=2 * int(cfg.num_coils),
            base=int(cfg.kspace_unet_base),
        )
        self.image_block = UNet3D(
            in_ch=2,
            out_ch=2,
            base=int(cfg.image_unet_base),
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
        k = forward_op(img, sens)
        bsz, nc, nx, ny, nz = k.shape

        k_ri = torch.stack([k.real, k.imag], dim=2).reshape(bsz, 2 * nc, nx, ny, nz)
        k_ri = self.kspace_block(k_ri)
        k = k_ri.reshape(bsz, nc, 2, nx, ny, nz)
        k = torch.complex(k[:, :, 0], k[:, :, 1])

        img = adjoint_op(k, sens)
        img_ri = torch.stack([img.real, img.imag], dim=1)
        img_ri = self.image_block(img_ri)
        return torch.complex(img_ri[:, 0], img_ri[:, 1])


class StrongUnrolled3D(nn.Module):
    def __init__(self, cfg: StrongUnrolled3DConfig):
        super().__init__()
        self.cfg = cfg
        self.cascades = nn.ModuleList([StrongUnrolled3DCascade(cfg) for _ in range(int(cfg.num_cascades))])
        eta0 = _inv_softplus(cfg.eta_init)
        mu0 = _inv_softplus(cfg.mu_init)
        self.eta = nn.ParameterList(
            [nn.Parameter(torch.tensor(eta0, dtype=torch.float32)) for _ in range(int(cfg.num_cascades))]
        )
        self.mu = nn.ParameterList(
            [nn.Parameter(torch.tensor(mu0, dtype=torch.float32)) for _ in range(int(cfg.num_cascades))]
        )
        self.lrs = LRS(cfg.lrs_hidden, cfg.lrs_blocks, cfg.lrs_max_scale) if cfg.use_lrs else None

    def _cascade_step(
        self,
        img: torch.Tensor,
        k_under: torch.Tensor,
        mask: torch.Tensor,
        sens: torch.Tensor,
        cascade_idx: int,
    ) -> torch.Tensor:
        grad = data_grad(img, k_under, mask, sens)
        img = img - F.softplus(self.eta[cascade_idx]) * grad

        img = self.cascades[cascade_idx](img, mask, sens)

        k_pred = forward_op(img, sens)
        k_dc = soft_dc(k_pred, k_under, mask, self.mu[cascade_idx])
        return adjoint_op(k_dc, sens)

    def forward(
        self,
        k_under: torch.Tensor,
        mask: torch.Tensor,
        sens: torch.Tensor,
        apply_lrs: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        img = adjoint_op(k_under, sens)
        inter = []

        for t in range(int(self.cfg.num_cascades)):
            if self.cfg.gradient_checkpointing and self.training:
                def _step(i, ku, m, s, cascade_idx=t):
                    return self._cascade_step(i, ku, m, s, cascade_idx)

                img = torch.utils.checkpoint.checkpoint(
                    _step,
                    img,
                    k_under,
                    mask,
                    sens,
                    use_reentrant=False,
                )
            else:
                img = self._cascade_step(img, k_under, mask, sens, t)
            inter.append(img.abs())

        if self.lrs is not None and apply_lrs:
            img = self.lrs(img)
        return img, inter

    def freeze_backbone(self):
        for p in self.cascades.parameters():
            p.requires_grad = False
        for p in self.eta:
            p.requires_grad = False
        for p in self.mu:
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.cascades.parameters():
            p.requires_grad = True
        for p in self.eta:
            p.requires_grad = True
        for p in self.mu:
            p.requires_grad = True
