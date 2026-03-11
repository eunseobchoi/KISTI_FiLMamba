from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from filmamba.models.kmb import KSpaceMambaBlock
from filmamba.models.lrs import LRS
from filmamba.models.ops import adjoint_op, data_grad, forward_op, soft_dc
from filmamba.models.unet2d import UNet2D
from filmamba.models.unet3d import UNet3D


@dataclass
class ModelConfig:
    num_coils: int = 12
    num_cascades: int = 8
    d_model: int = 192
    patch_size: int = 8
    num_bands: int = 8
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    unet_base: int = 32
    image_block_2d: bool = False
    slice_wise_2d: bool = False
    scan_mode: str = "sfos"  # sfos | raster
    use_hs_mamba: bool = True
    num_flat_layers: int = 5
    multi_coil_embed: bool = True
    condtok: bool = False
    cond_use_band: bool = True
    cond_use_radius: bool = True
    cond_use_mask: bool = True
    modulate_delta: bool = True
    modulate_B: bool = True
    modulate_C: bool = True
    mod_use_band: bool = True
    mod_use_radius: bool = True
    mod_use_mask: bool = True
    exact_zoh: bool = True
    smax_dt: float = 2.0
    smax_B: float = 2.0
    smax_C: float = 2.0
    use_lrs: bool = True
    lrs_hidden: int = 16
    lrs_blocks: int = 3
    lrs_max_scale: float = 0.3
    gradient_checkpointing: bool = True
    eta_init: float = 0.1
    mu_init: float = 0.01


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    # Inverse of softplus so raw parameter initializes to desired effective value.
    return float(x + math.log(-math.expm1(-x)))


class FiLMambaCascade(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.kmb = KSpaceMambaBlock(
            num_coils=cfg.num_coils,
            d_model=cfg.d_model,
            patch_size=cfg.patch_size,
            num_bands=cfg.num_bands,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            use_hs_mamba=cfg.use_hs_mamba,
            num_flat_layers=cfg.num_flat_layers,
            scan_mode=cfg.scan_mode,
            multi_coil_embed=cfg.multi_coil_embed,
            slice_wise_2d=cfg.slice_wise_2d,
            condtok=cfg.condtok,
            cond_use_band=cfg.cond_use_band,
            cond_use_radius=cfg.cond_use_radius,
            cond_use_mask=cfg.cond_use_mask,
            modulate_delta=cfg.modulate_delta,
            modulate_B=cfg.modulate_B,
            modulate_C=cfg.modulate_C,
            mod_use_band=cfg.mod_use_band,
            mod_use_radius=cfg.mod_use_radius,
            mod_use_mask=cfg.mod_use_mask,
            exact_zoh=cfg.exact_zoh,
            smax_dt=cfg.smax_dt,
            smax_B=cfg.smax_B,
            smax_C=cfg.smax_C,
        )
        self.imb = UNet2D(in_ch=2, out_ch=2, base=cfg.unet_base) if cfg.image_block_2d else UNet3D(in_ch=2, out_ch=2, base=cfg.unet_base)

    def forward(self, img: torch.Tensor, mask: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
        k = forward_op(img, sens)
        bsz, nc, nx, ny, nz = k.shape
        k_ri = torch.stack([k.real, k.imag], dim=2).reshape(bsz, 2 * nc, nx, ny, nz)
        k_ri = self.kmb(k_ri, mask)
        k = k_ri.reshape(bsz, nc, 2, nx, ny, nz)
        k = torch.complex(k[:, :, 0], k[:, :, 1])

        img = adjoint_op(k, sens)
        img_ri = torch.stack([img.real, img.imag], dim=1)
        img_ri = self.imb(img_ri)
        return torch.complex(img_ri[:, 0], img_ri[:, 1])


class FiLMamba(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.cascades = nn.ModuleList([FiLMambaCascade(cfg) for _ in range(cfg.num_cascades)])
        eta0 = _inv_softplus(cfg.eta_init)
        mu0 = _inv_softplus(cfg.mu_init)
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(eta0, dtype=torch.float32)) for _ in range(cfg.num_cascades)])
        self.mu = nn.ParameterList([nn.Parameter(torch.tensor(mu0, dtype=torch.float32)) for _ in range(cfg.num_cascades)])
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
        img = adjoint_op(k_dc, sens)
        return img

    def forward(
        self,
        k_under: torch.Tensor,
        mask: torch.Tensor,
        sens: torch.Tensor,
        apply_lrs: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        img = adjoint_op(k_under, sens)
        inter = []

        for t in range(self.cfg.num_cascades):
            if self.cfg.gradient_checkpointing and self.training:
                # Bind cascade index at definition time; otherwise checkpoint
                # recomputation can capture the loop variable late and backprop
                # through the wrong cascade.
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
            # Deep supervision only uses |x_t|, so keep magnitude to reduce memory.
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
