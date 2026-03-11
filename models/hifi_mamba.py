from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from filmamba.models.fcssm import FCSSM
from filmamba.models.lrs import LRS
from filmamba.models.ops import adjoint_op, data_grad, forward_op, soft_dc


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    return float(x + math.log(-math.expm1(-x)))


class ChannelAttentionBlock2D(nn.Module):
    """Channel attention block aligned with the released HiFi-Mamba module style."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv3x3_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.gelu(self.conv1x1(x))
        a = torch.sigmoid(self.conv3x3_dw(a))
        return a * x


class DWInvertedBottleneck2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion_ratio: int = 4):
        super().__init__()
        hidden_dim = int(in_channels * expansion_ratio)
        self.use_residual = in_channels == out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_residual:
            return x + y
        return y


class HaarLikeWLBlock2D(nn.Module):
    """
    Wavelet-like low/high frequency split without extra dependencies.
    Uses average-pool lowpass and residual highpass at identical resolution.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.pre = DWInvertedBottleneck2D(dim, dim, expansion_ratio=4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pre(x)
        low = F.avg_pool2d(x, kernel_size=2, stride=2)
        low_up = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        high = x - low_up
        return low_up, high


class RasterSSM2D(nn.Module):
    """2D raster SSM using the same selective scan backend as the main codebase."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pre_norm = nn.LayerNorm(self.d_model)
        self.guide_proj = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, bias=False)
        self.ssm = FCSSM(
            d_model=self.d_model,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            num_bands=1,
            modulate_delta=False,
            modulate_B=False,
            modulate_C=False,
            use_band=False,
            use_mask=False,
            use_radius=False,
            exact_zoh=True,
        )
        self._raster_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_raster_meta(self, hw: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if hw not in self._raster_cache:
            h, w = hw
            l = int(h * w)
            band = torch.zeros(l, dtype=torch.long)
            if l <= 1:
                rho = torch.zeros(l, dtype=torch.float32)
            else:
                rho = torch.arange(l, dtype=torch.float32) / float(l - 1)
            self._raster_cache[hw] = (band, rho)
        band, rho = self._raster_cache[hw]
        return band.to(device), rho.to(device)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        bsz, ch, h, w = x.shape
        if ch != self.d_model:
            raise ValueError(f"RasterBiSSM2D expects channels={self.d_model}, got {ch}")

        seq = (x + self.guide_proj(guide)).flatten(2).transpose(1, 2).contiguous()  # [B,L,C]
        seq = self.pre_norm(seq)

        band, rho = self._get_raster_meta((h, w), x.device)
        m_frac = torch.ones((bsz, h * w), device=x.device, dtype=seq.dtype)
        y = self.ssm(seq, band, m_frac, rho)
        return y.transpose(1, 2).reshape(bsz, ch, h, w).contiguous()


class HiFiMambaUnit2D(nn.Module):
    """
    Dual-stream unit inspired by HiFi-Mamba:
    frequency split + guidance branch + 2D SSM on low-frequency stream.
    """

    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"HiFi unit requires even channels, got dim={dim}")
        half = dim // 2

        self.wl = HaarLikeWLBlock2D(half)
        self.crm = DWInvertedBottleneck2D(half, half, expansion_ratio=4)
        self.high_refine = DWInvertedBottleneck2D(half, half, expansion_ratio=2)
        self.low_ssm = RasterSSM2D(d_model=half, d_state=d_state, d_conv=d_conv, expand=expand)

        self.fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.dsfa = ChannelAttentionBlock2D(dim)
        self.out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_low, f_high = torch.chunk(x, 2, dim=1)

        low, high = self.wl(f_low)
        guide = self.crm(f_high)

        low_enh = self.low_ssm(low, guide)
        high_enh = self.high_refine(high + guide)

        fused = torch.cat([low_enh, high_enh], dim=1)
        fused = self.fuse(fused)
        fused = self.dsfa(fused)
        return x + self.out(fused)


class HiFiGroup2D(nn.Module):
    """One unrolled group containing two HiFi units (paper setting)."""

    def __init__(
        self,
        hidden_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        units_per_group: int,
        patch_size: int,
        use_checkpoint: bool,
    ):
        super().__init__()
        self.patch_size = int(max(1, patch_size))
        self.use_checkpoint = bool(use_checkpoint)
        p = self.patch_size
        self.in_proj = nn.Conv2d(2, hidden_dim, kernel_size=p, stride=p, bias=False)
        self.units = nn.ModuleList(
            [
                HiFiMambaUnit2D(
                    dim=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(max(1, int(units_per_group)))
            ]
        )
        self.out_proj = nn.Conv2d(hidden_dim, 2 * p * p, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x_ri_2d: torch.Tensor) -> torch.Tensor:
        _, _, h0, w0 = x_ri_2d.shape
        p = self.patch_size
        pad_h = (p - (h0 % p)) % p
        pad_w = (p - (w0 % p)) % p
        x_pad = F.pad(x_ri_2d, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else x_ri_2d

        h = self.in_proj(x_pad)
        for unit in self.units:
            if self.use_checkpoint and self.training:
                h = checkpoint_fn(unit, h, use_reentrant=False)
            else:
                h = unit(h)

        y_pad = F.pixel_shuffle(self.out_proj(h), p)
        y = y_pad[..., :h0, :w0]
        return x_ri_2d + y


@dataclass
class HiFiModelConfig:
    model_family: str = "hifi_mamba"
    num_coils: int = 12
    num_cascades: int = 8
    d_state: int = 16
    d_conv: int = 3
    expand: int = 2
    use_lrs: bool = False
    lrs_hidden: int = 16
    lrs_blocks: int = 3
    lrs_max_scale: float = 0.3
    eta_init: float = 0.1
    mu_init: float = 0.01
    # 192 channels yields ~7.6M params for 8 groups, close to reported 7.5M.
    hifi_hidden: int = 192
    hifi_units_per_group: int = 2
    hifi_slice_batch: int = 1
    hifi_patch_size: int = 4
    gradient_checkpointing: bool = False


class HiFiMamba(nn.Module):
    """
    Slice-wise 2D HiFi-Mamba baseline under the same unrolled DC framework.
    """

    def __init__(self, cfg: HiFiModelConfig):
        super().__init__()
        self.cfg = cfg
        self.groups = nn.ModuleList(
            [
                HiFiGroup2D(
                    hidden_dim=int(cfg.hifi_hidden),
                    d_state=int(cfg.d_state),
                    d_conv=int(cfg.d_conv),
                    expand=int(cfg.expand),
                    units_per_group=int(cfg.hifi_units_per_group),
                    patch_size=int(cfg.hifi_patch_size),
                    use_checkpoint=bool(cfg.gradient_checkpointing),
                )
                for _ in range(int(cfg.num_cascades))
            ]
        )
        eta0 = _inv_softplus(cfg.eta_init)
        mu0 = _inv_softplus(cfg.mu_init)
        self.eta = nn.ParameterList(
            [nn.Parameter(torch.tensor(eta0, dtype=torch.float32)) for _ in range(int(cfg.num_cascades))]
        )
        self.mu = nn.ParameterList(
            [nn.Parameter(torch.tensor(mu0, dtype=torch.float32)) for _ in range(int(cfg.num_cascades))]
        )
        self.lrs = LRS(cfg.lrs_hidden, cfg.lrs_blocks, cfg.lrs_max_scale) if cfg.use_lrs else None

    def _apply_group_slice_wise(self, img: torch.Tensor, group: HiFiGroup2D) -> torch.Tensor:
        # img: [B, X, Y, Z] complex
        bsz, nx, ny, nz = img.shape
        x_ri = torch.stack([img.real, img.imag], dim=1)  # [B,2,X,Y,Z]
        xs = x_ri.permute(0, 4, 1, 2, 3).reshape(bsz * nz, 2, nx, ny).contiguous()

        chunk = max(1, int(self.cfg.hifi_slice_batch))
        ys: List[torch.Tensor] = []
        for i in range(0, xs.shape[0], chunk):
            ys.append(group(xs[i : i + chunk]))
        y = torch.cat(ys, dim=0)

        y = y.reshape(bsz, nz, 2, nx, ny).permute(0, 2, 3, 4, 1).contiguous()  # [B,2,X,Y,Z]
        return torch.complex(y[:, 0], y[:, 1])

    def _group_step(
        self,
        img: torch.Tensor,
        k_under: torch.Tensor,
        mask: torch.Tensor,
        sens: torch.Tensor,
        group_idx: int,
    ) -> torch.Tensor:
        grad = data_grad(img, k_under, mask, sens)
        img = img - F.softplus(self.eta[group_idx]) * grad

        img = self._apply_group_slice_wise(img, self.groups[group_idx])

        k_pred = forward_op(img, sens)
        k_dc = soft_dc(k_pred, k_under, mask, self.mu[group_idx])
        return adjoint_op(k_dc, sens)

    def forward(
        self,
        k_under: torch.Tensor,
        mask: torch.Tensor,
        sens: torch.Tensor,
        apply_lrs: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        img = adjoint_op(k_under, sens)
        inter: List[torch.Tensor] = []

        for t in range(int(self.cfg.num_cascades)):
            img = self._group_step(img, k_under, mask, sens, t)
            inter.append(img.abs())

        if self.lrs is not None and apply_lrs:
            img = self.lrs(img)
        return img, inter

    def freeze_backbone(self):
        for p in self.groups.parameters():
            p.requires_grad = False
        for p in self.eta:
            p.requires_grad = False
        for p in self.mu:
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.groups.parameters():
            p.requires_grad = True
        for p in self.eta:
            p.requires_grad = True
        for p in self.mu:
            p.requires_grad = True
