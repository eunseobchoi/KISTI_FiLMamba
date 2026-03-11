from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from filmamba.models.fcssm import BiFCSSM


class HSMamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        num_bands: int,
        modulate_delta: bool = True,
        modulate_B: bool = True,
        modulate_C: bool = True,
        mod_use_band: bool = True,
        mod_use_mask: bool = True,
        mod_use_radius: bool = True,
        exact_zoh: bool = True,
        smax_dt: float = 2.0,
        smax_B: float = 2.0,
        smax_C: float = 2.0,
        num_intra_layers: int = 2,
        num_inter_layers: int = 1,
        num_refine_layers: int = 2,
    ):
        super().__init__()
        self.num_bands = int(num_bands)

        self.intra_layers = nn.ModuleList(
            [
                BiFCSSM(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    num_bands,
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
                for _ in range(num_intra_layers)
            ]
        )
        self.inter_layers = nn.ModuleList(
            [
                BiFCSSM(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    num_bands,
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
                for _ in range(num_inter_layers)
            ]
        )
        self.refine_layers = nn.ModuleList(
            [
                BiFCSSM(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    num_bands,
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
                for _ in range(num_refine_layers)
            ]
        )

        self.intra_norm = nn.LayerNorm(d_model)
        self.inter_norm = nn.LayerNorm(d_model)
        self.refine_norm = nn.LayerNorm(d_model)

        self.gate = nn.Linear(2 * d_model, d_model)
        self.gate_norm = nn.LayerNorm(d_model)

    def _band_indices(self, band: torch.Tensor) -> List[torch.Tensor]:
        return [torch.where(band == b)[0] for b in range(self.num_bands)]

    def _apply_band_layers(
        self,
        x: torch.Tensor,
        band: torch.Tensor,
        mask_fraction: torch.Tensor,
        rho_bar: torch.Tensor,
        layers: nn.ModuleList,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        h = x
        token_idx = self._band_indices(band)
        for layer in layers:
            h_new = torch.zeros_like(h)
            h_norm = norm(h)
            for idx in token_idx:
                if idx.numel() == 0:
                    continue
                hb = h_norm.index_select(1, idx)
                bb = band.index_select(0, idx)
                mb = mask_fraction.index_select(1, idx)
                rb = rho_bar.index_select(0, idx)
                outb = layer(hb, bb, mb, rb)
                h_new[:, idx, :] = h.index_select(1, idx) + outb
            h = h_new
        return h

    def _summary_cues(
        self,
        band: torch.Tensor,
        rho_bar: torch.Tensor,
        mask_fraction: torch.Tensor,
    ):
        """
        Build level-2 summary-token cues:
          rho_summary[b] = mean rho over tokens in band b
          m_summary[b]   = mean mask-fraction over tokens in band b
        """
        token_idx = self._band_indices(band)
        bsz = mask_fraction.shape[0]

        rho_summary = torch.zeros(self.num_bands, device=mask_fraction.device, dtype=mask_fraction.dtype)
        m_summary = torch.zeros(bsz, self.num_bands, device=mask_fraction.device, dtype=mask_fraction.dtype)

        for b, idx in enumerate(token_idx):
            if idx.numel() == 0:
                # Defensive fallback; proper configs should keep all bands non-empty.
                rho_summary[b] = (b + 0.5) / float(max(1, self.num_bands))
                m_summary[:, b] = 0.0
                continue
            rho_summary[b] = rho_bar.index_select(0, idx).mean()
            m_summary[:, b] = mask_fraction.index_select(1, idx).mean(dim=1)

        return token_idx, rho_summary, m_summary

    def forward(
        self,
        x: torch.Tensor,
        band: torch.Tensor,
        mask_fraction: torch.Tensor,
        rho_bar: torch.Tensor,
    ) -> torch.Tensor:
        # Level 1: intra-band FC-SSM.
        h = self._apply_band_layers(x, band, mask_fraction, rho_bar, self.intra_layers, self.intra_norm)

        # Level 2: inter-band summary FC-SSM.
        bsz, _, dim = h.shape
        token_idx, rho_summary, m_summary = self._summary_cues(band, rho_bar, mask_fraction)

        summaries = torch.zeros(bsz, self.num_bands, dim, device=h.device, dtype=h.dtype)
        for b, idx in enumerate(token_idx):
            if idx.numel() == 0:
                continue
            summaries[:, b, :] = h.index_select(1, idx).mean(dim=1)

        inter_band = torch.arange(self.num_bands, device=h.device)

        g = summaries
        for layer in self.inter_layers:
            g = g + layer(self.inter_norm(g), inter_band, m_summary, rho_summary)

        # Level 3: global-to-local broadcast + refinement FC-SSM.
        g_up = g[:, band, :]
        alpha = torch.sigmoid(self.gate(torch.cat([h, g_up], dim=-1)))
        h = h + alpha * self.gate_norm(g_up)

        # Reuse token-level cues (band/rho/mask-fraction) in refinement FC-SSM.
        h = self._apply_band_layers(h, band, mask_fraction, rho_bar, self.refine_layers, self.refine_norm)
        return h
