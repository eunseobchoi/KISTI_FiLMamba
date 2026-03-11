from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from filmamba.models.fcssm import BiFCSSM
from filmamba.models.hs_mamba import HSMamba
from filmamba.models.sfos import SFOS


class KSpaceMambaBlock(nn.Module):
    def __init__(
        self,
        num_coils: int,
        d_model: int,
        patch_size: int,
        num_bands: int,
        d_state: int,
        d_conv: int,
        expand: int,
        use_hs_mamba: bool = True,
        num_flat_layers: int = 5,
        scan_mode: str = "sfos",
        multi_coil_embed: bool = True,
        slice_wise_2d: bool = False,
        condtok: bool = False,
        cond_use_band: bool = True,
        cond_use_radius: bool = True,
        cond_use_mask: bool = True,
        modulate_delta: bool = True,
        modulate_B: bool = True,
        modulate_C: bool = True,
        mod_use_band: bool = True,
        mod_use_radius: bool = True,
        mod_use_mask: bool = True,
        exact_zoh: bool = True,
        smax_dt: float = 2.0,
        smax_B: float = 2.0,
        smax_C: float = 2.0,
    ):
        super().__init__()
        if scan_mode not in {"sfos", "raster"}:
            raise ValueError(f"Unsupported scan_mode={scan_mode}; use 'sfos' or 'raster'")

        self.num_coils = int(num_coils)
        self.num_bands = int(num_bands)
        self.use_hs_mamba = bool(use_hs_mamba)
        self.scan_mode = scan_mode
        self.multi_coil_embed = bool(multi_coil_embed)
        self.slice_wise_2d = bool(slice_wise_2d)
        self.condtok = bool(condtok)
        self.cond_use_band = bool(cond_use_band)
        self.cond_use_radius = bool(cond_use_radius)
        self.cond_use_mask = bool(cond_use_mask)

        self.patch_size_xy = int(patch_size)
        self.patch_size_z = 1 if self.slice_wise_2d else int(patch_size)
        patch_kernel = (self.patch_size_xy, self.patch_size_xy, self.patch_size_z)
        patch_stride = patch_kernel

        in_ch_proc = 2 * self.num_coils if self.multi_coil_embed else 2
        self.patch_embed = nn.Conv3d(in_ch_proc, d_model, kernel_size=patch_kernel, stride=patch_stride)
        self.patch_unembed = nn.ConvTranspose3d(d_model, in_ch_proc, kernel_size=patch_kernel, stride=patch_stride)
        self.norm = nn.LayerNorm(d_model)

        # Residual branch starts from identity.
        nn.init.zeros_(self.patch_unembed.weight)
        if self.patch_unembed.bias is not None:
            nn.init.zeros_(self.patch_unembed.bias)

        if self.use_hs_mamba:
            self.hsm = HSMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                num_bands=num_bands,
                modulate_delta=modulate_delta,
                modulate_B=modulate_B,
                modulate_C=modulate_C,
                mod_use_band=mod_use_band,
                mod_use_mask=mod_use_mask,
                mod_use_radius=mod_use_radius,
                exact_zoh=exact_zoh,
                smax_dt=smax_dt,
                smax_B=smax_B,
                smax_C=smax_C,
                num_intra_layers=2,
                num_inter_layers=1,
                num_refine_layers=2,
            )
            self.flat_layers = None
            self.flat_norm = None
        else:
            self.hsm = None
            self.flat_layers = nn.ModuleList(
                [
                    BiFCSSM(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        num_bands=num_bands,
                        modulate_delta=modulate_delta,
                        modulate_B=modulate_B,
                        modulate_C=modulate_C,
                        use_band=mod_use_band,
                        use_mask=mod_use_mask,
                        use_radius=mod_use_radius,
                        exact_zoh=exact_zoh,
                        smax_dt=smax_dt,
                        smax_B=smax_B,
                        smax_C=smax_C,
                    )
                    for _ in range(max(1, int(num_flat_layers)))
                ]
            )
            self.flat_norm = nn.LayerNorm(d_model)

        if self.condtok:
            self.cond_band_embed = nn.Embedding(num_bands, d_model) if self.cond_use_band else None
            self.cond_radius_proj = nn.Linear(1, d_model, bias=False) if self.cond_use_radius else None
            self.cond_mask_proj = nn.Linear(1, d_model, bias=False) if self.cond_use_mask else None
            if self.cond_band_embed is not None:
                nn.init.zeros_(self.cond_band_embed.weight)
            if self.cond_radius_proj is not None:
                nn.init.zeros_(self.cond_radius_proj.weight)
            if self.cond_mask_proj is not None:
                nn.init.zeros_(self.cond_mask_proj.weight)
        else:
            self.cond_band_embed = None
            self.cond_radius_proj = None
            self.cond_mask_proj = None

        self._sfos_cache: Dict[Tuple[int, int, int], SFOS] = {}
        self._raster_cache: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_sfos(self, grid: Tuple[int, int, int], device: torch.device) -> SFOS:
        if grid not in self._sfos_cache:
            self._sfos_cache[grid] = SFOS(grid, self.num_bands)
        return self._sfos_cache[grid].to(device)

    def _get_raster(self, grid: Tuple[int, int, int], device: torch.device):
        if grid not in self._raster_cache:
            l = int(grid[0] * grid[1] * grid[2])
            forward = torch.arange(l, dtype=torch.long)
            inverse = torch.arange(l, dtype=torch.long)
            if l <= 1:
                rho_bar = torch.zeros(l, dtype=torch.float32)
            else:
                rho_bar = torch.arange(l, dtype=torch.float32) / float(l - 1)
            band = torch.clamp(torch.floor(rho_bar * self.num_bands).long(), max=self.num_bands - 1)
            self._raster_cache[grid] = (forward, inverse, band, rho_bar)

        forward, inverse, band, rho_bar = self._raster_cache[grid]
        return forward.to(device), inverse.to(device), band.to(device), rho_bar.to(device)

    def _pad_to_patch(self, x: torch.Tensor):
        _, _, nx, ny, nz = x.shape
        px = self.patch_size_xy
        py = self.patch_size_xy
        pz = self.patch_size_z
        tx = ((nx + px - 1) // px) * px
        ty = ((ny + py - 1) // py) * py
        tz = ((nz + pz - 1) // pz) * pz
        padx = tx - nx
        pady = ty - ny
        padz = tz - nz
        pad = (0, padz, 0, pady, 0, padx)
        if any(v > 0 for v in pad):
            x = F.pad(x, pad)
        return x, (nx, ny, nz)

    def _compute_patch_mask_fraction(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute continuous measured-sample fraction per patch in [0,1],
        corrected for boundary patches via valid voxel count.
        """
        k = (self.patch_size_xy, self.patch_size_xy, self.patch_size_z)
        s = k
        patch_vol = float(k[0] * k[1] * k[2])

        m_pad, _ = self._pad_to_patch(mask.float())
        valid_pad, _ = self._pad_to_patch(torch.ones_like(mask, dtype=torch.float32))

        measured = F.avg_pool3d(m_pad, kernel_size=k, stride=s) * patch_vol
        valid = F.avg_pool3d(valid_pad, kernel_size=k, stride=s) * patch_vol

        frac = measured / valid.clamp_min(1.0)
        return frac.clamp(0.0, 1.0)

    def _collapse_coils(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 2Nc, X, Y, Z] -> [B, 2, X, Y, Z]
        bsz, _, nx, ny, nz = x.shape
        xc = x.view(bsz, self.num_coils, 2, nx, ny, nz)
        return xc.mean(dim=1)

    def _expand_single_delta(self, delta_single: torch.Tensor) -> torch.Tensor:
        # [B,2,X,Y,Z] -> [B,2Nc,X,Y,Z]
        return delta_single.unsqueeze(1).repeat(1, self.num_coils, 1, 1, 1, 1).flatten(1, 2)

    def _apply_model(
        self,
        x_ord: torch.Tensor,
        band: torch.Tensor,
        m_ord: torch.Tensor,
        rho_ord: torch.Tensor,
    ) -> torch.Tensor:
        if self.condtok:
            cond = torch.zeros_like(x_ord)
            if self.cond_band_embed is not None:
                cond = cond + self.cond_band_embed(band).unsqueeze(0)
            if self.cond_radius_proj is not None:
                cond = cond + self.cond_radius_proj(rho_ord.unsqueeze(-1)).unsqueeze(0)
            if self.cond_mask_proj is not None:
                cond = cond + self.cond_mask_proj(m_ord.unsqueeze(-1))
            x_in = self.norm(x_ord + cond)
        else:
            x_in = self.norm(x_ord)

        if self.use_hs_mamba:
            return self.hsm(x_in, band, m_ord, rho_ord)

        h = x_in
        for layer in self.flat_layers:
            h = h + layer(self.flat_norm(h), band, m_ord, rho_ord)
        return h

    def _forward_core(self, kspace_ri: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # kspace_ri: [B,2Nc,Nx,Ny,Nz], mask:[B,1,Nx,Ny,Nz]
        x_in = kspace_ri
        if self.multi_coil_embed:
            x_proc = kspace_ri
        else:
            x_proc = self._collapse_coils(kspace_ri)

        x, original = self._pad_to_patch(x_proc)

        bsz = x.shape[0]
        x = self.patch_embed(x)  # [B,D,Gx,Gy,Gz]
        gx, gy, gz = x.shape[-3:]
        l = gx * gy * gz
        x = x.flatten(2).transpose(1, 2)  # [B,L,D]

        m_patch = self._compute_patch_mask_fraction(mask).view(bsz, l)

        if self.scan_mode == "sfos":
            sfos = self._get_sfos((gx, gy, gz), x.device)
            x_ord, band, rho_ord = sfos(x)
            m_ord = m_patch[:, sfos.forward_order]
            h = self._apply_model(x_ord, band, m_ord, rho_ord)
            h = sfos.inverse(h)
        else:
            forward, inverse, band, rho_ord = self._get_raster((gx, gy, gz), x.device)
            x_ord = x[:, forward, :]
            m_ord = m_patch[:, forward]
            h = self._apply_model(x_ord, band, m_ord, rho_ord)
            h = h[:, inverse, :]

        h = h.transpose(1, 2).reshape(bsz, -1, gx, gy, gz)
        out_proc = self.patch_unembed(h)
        nx, ny, nz = original
        out_proc = out_proc[:, :, :nx, :ny, :nz]

        if self.multi_coil_embed:
            return x_in + out_proc

        # Non-MCA path learns a shared residual update and applies it to all coils.
        delta_rep = self._expand_single_delta(out_proc)
        return x_in + delta_rep

    def forward(self, kspace_ri: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.slice_wise_2d:
            return self._forward_core(kspace_ri, mask)

        # Slice-wise 2D mode: no through-plane interaction in the k-space branch.
        bsz, ch, nx, ny, nz = kspace_ri.shape
        xs = kspace_ri.permute(0, 4, 1, 2, 3).reshape(bsz * nz, ch, nx, ny, 1)
        ms = mask.permute(0, 4, 1, 2, 3).reshape(bsz * nz, 1, nx, ny, 1)
        ys = self._forward_core(xs, ms)
        y = ys.reshape(bsz, nz, ch, nx, ny, 1).squeeze(-1).permute(0, 2, 3, 4, 1).contiguous()
        return y
