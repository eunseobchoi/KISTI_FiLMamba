from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_SELECTIVE_SCAN_FN = None
_SELECTIVE_SCAN_READY = False
_SELECTIVE_SCAN_FALLBACK_WARNED = False
_SELECTIVE_SCAN_SUCCESS_WARNED = False


def _get_selective_scan_fn():
    global _SELECTIVE_SCAN_FN
    global _SELECTIVE_SCAN_READY
    if not _SELECTIVE_SCAN_READY:
        _SELECTIVE_SCAN_READY = True
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            _SELECTIVE_SCAN_FN = selective_scan_fn
        except Exception:
            _SELECTIVE_SCAN_FN = None
    return _SELECTIVE_SCAN_FN


def _selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Real-valued reference selective scan.

    Args:
        u/delta/z: [B,D,L]
        A: [D,N]
        B: [B,N,L] or [B,D,N,L]
        C: [B,N,L] or [B,D,N,L]
        D: [D]
    """
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    if B.dim() not in (3, 4):
        raise ValueError(f"B must be 3D or 4D, got {B.shape}")
    if C.dim() not in (3, 4):
        raise ValueError(f"C must be 3D or 4D, got {C.shape}")

    if B.dim() == 4 and B.shape[1] != dim:
        if dim % B.shape[1] != 0:
            raise ValueError(f"B groups={B.shape[1]} must divide dim={dim}")
        B = B.repeat_interleave(dim // B.shape[1], dim=1).contiguous()
    if C.dim() == 4 and C.shape[1] != dim:
        if dim % C.shape[1] != 0:
            raise ValueError(f"C groups={C.shape[1]} must divide dim={dim}")
        C = C.repeat_interleave(dim // C.shape[1], dim=1).contiguous()

    # Memory-safe recurrence scan (x0 = 0) to avoid 4D time tensors in fallback mode.
    x = torch.zeros((batch, dim, dstate), device=u.device, dtype=u.dtype)
    ys = []
    A_ = A.unsqueeze(0)  # [1,D,N]
    for i in range(seqlen):
        dt_i = delta[:, :, i]  # [B,D]
        u_i = u[:, :, i]  # [B,D]
        a_i = torch.exp(dt_i.unsqueeze(-1) * A_)  # [B,D,N]
        if B.dim() == 3:
            b_i = dt_i.unsqueeze(-1) * B[:, :, i].unsqueeze(1) * u_i.unsqueeze(-1)
        else:
            b_i = dt_i.unsqueeze(-1) * B[:, :, :, i] * u_i.unsqueeze(-1)
        x = a_i * x + b_i
        if C.dim() == 3:
            y_i = (x * C[:, :, i].unsqueeze(1)).sum(dim=-1)
        else:
            y_i = (x * C[:, :, :, i]).sum(dim=-1)
        ys.append(y_i)
    y = torch.stack(ys, dim=2)

    out = y + u * D.view(1, -1, 1)
    if z is not None:
        out = out * F.silu(z)
    return out


def _selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
) -> torch.Tensor:
    global _SELECTIVE_SCAN_FALLBACK_WARNED
    global _SELECTIVE_SCAN_SUCCESS_WARNED

    if u.is_cuda:
        selective_scan_fn = _get_selective_scan_fn()
        if selective_scan_fn is not None:
            try:
                # Kernel expects B/C group dims to be aligned: [B, n_groups, d_state, L].
                B_k = B.unsqueeze(1) if B.dim() == 3 else B
                C_k = C.unsqueeze(1) if C.dim() == 3 else C
                if B_k.dim() == 4 and C_k.dim() == 4 and B_k.shape[1] != C_k.shape[1]:
                    if B_k.shape[1] == 1:
                        B_k = B_k.expand(-1, C_k.shape[1], -1, -1)
                    elif C_k.shape[1] == 1:
                        C_k = C_k.expand(-1, B_k.shape[1], -1, -1)
                    else:
                        raise ValueError(
                            f"Incompatible B/C groups for selective_scan: B={tuple(B_k.shape)} C={tuple(C_k.shape)}"
                        )
                out = selective_scan_fn(
                    u,
                    delta,
                    A,
                    B_k,
                    C_k,
                    D,
                    z=z,
                    delta_bias=None,
                    delta_softplus=False,
                )
                if not _SELECTIVE_SCAN_SUCCESS_WARNED and os.environ.get("FILMAMBA_SCAN_DEBUG", "0") == "1":
                    _SELECTIVE_SCAN_SUCCESS_WARNED = True
                    print("[FCSSM] selective_scan_fn active", flush=True)
                return out
            except Exception as e:
                if not _SELECTIVE_SCAN_FALLBACK_WARNED and os.environ.get("FILMAMBA_SCAN_DEBUG", "0") == "1":
                    _SELECTIVE_SCAN_FALLBACK_WARNED = True
                    print(f"[FCSSM] selective_scan_fn failed; fallback to reference scan: {repr(e)}", flush=True)
        elif not _SELECTIVE_SCAN_FALLBACK_WARNED and os.environ.get("FILMAMBA_SCAN_DEBUG", "0") == "1":
            _SELECTIVE_SCAN_FALLBACK_WARNED = True
            print("[FCSSM] selective_scan_fn unavailable; fallback to reference scan", flush=True)

    return _selective_scan_ref(u, delta, A, B, C, D, z=z)


class FCSSM(nn.Module):
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
        use_band: bool = True,
        use_mask: bool = True,
        use_radius: bool = True,
        exact_zoh: bool = True,
        smax_dt: float = 2.0,
        smax_B: float = 2.0,
        smax_C: float = 2.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(d_model * expand)
        self.num_bands = int(num_bands)
        self.dt_rank = math.ceil(d_model / 16)

        self.modulate_delta = bool(modulate_delta)
        self.modulate_B = bool(modulate_B)
        self.modulate_C = bool(modulate_C)
        self.use_band = bool(use_band)
        self.use_mask = bool(use_mask)
        self.use_radius = bool(use_radius)
        self.exact_zoh = bool(exact_zoh)

        self.smax_dt = float(smax_dt)
        self.smax_B = float(smax_B)
        self.smax_C = float(smax_C)

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Stable diagonal continuous transition: A = -diag(lambda), lambda > 0.
        A0 = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.lambda_log = nn.Parameter(torch.log(A0))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Bounded log-gain conditioners s(·) for Delta / B / C.
        self.s_band_dt = nn.Embedding(self.num_bands, self.d_inner) if self.use_band else None
        self.s_rho_dt = nn.Linear(1, self.d_inner, bias=False) if self.use_radius else None
        self.s_mask_dt = nn.Linear(1, self.d_inner, bias=False) if self.use_mask else None

        self.s_band_B = nn.Embedding(self.num_bands, self.d_state) if self.use_band else None
        self.s_rho_B = nn.Linear(1, self.d_state, bias=False) if self.use_radius else None
        self.s_mask_B = nn.Linear(1, self.d_state, bias=False) if self.use_mask else None

        self.s_band_C = nn.Embedding(self.num_bands, self.d_state) if self.use_band else None
        self.s_rho_C = nn.Linear(1, self.d_state, bias=False) if self.use_radius else None
        self.s_mask_C = nn.Linear(1, self.d_state, bias=False) if self.use_mask else None

        self._init_params(dt_min, dt_max)

    def _init_params(self, dt_min: float, dt_max: float):
        # Base Delta init: log-uniform in [1e-3, 1e-1] via inverse softplus bias.
        dt_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_std, dt_std)
        dt0 = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_sp = dt0 + torch.log(-torch.expm1(-dt0))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_sp)

        # Initialize all log-gain branches to zero => exact fallback to standard selective SSM.
        gain_modules = [
            self.s_band_dt,
            self.s_rho_dt,
            self.s_mask_dt,
            self.s_band_B,
            self.s_rho_B,
            self.s_mask_B,
            self.s_band_C,
            self.s_rho_C,
            self.s_mask_C,
        ]
        for mod in gain_modules:
            if mod is None:
                continue
            if isinstance(mod, nn.Embedding):
                nn.init.zeros_(mod.weight)
            elif isinstance(mod, nn.Linear):
                nn.init.zeros_(mod.weight)

        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)

    @staticmethod
    def _sum_log_gain(
        band_indices: torch.Tensor,
        rho_bar: torch.Tensor,
        mask_fraction: torch.Tensor,
        s_band: Optional[nn.Embedding],
        s_rho: Optional[nn.Linear],
        s_mask: Optional[nn.Linear],
        out_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Output: [B,L,out_dim]
        bsz, seqlen = mask_fraction.shape
        g = torch.zeros((bsz, seqlen, out_dim), device=mask_fraction.device, dtype=dtype)

        if s_band is not None:
            g = g + s_band(band_indices).unsqueeze(0).to(dtype=dtype)
        if s_mask is not None:
            g = g + s_mask(mask_fraction.unsqueeze(-1).to(dtype=dtype))
        if s_rho is not None:
            g = g + s_rho(rho_bar.unsqueeze(-1).to(dtype=dtype)).unsqueeze(0)
        return g

    def _apply_bounded_log_gain(self, base: torch.Tensor, log_gain: torch.Tensor, smax: float) -> torch.Tensor:
        # base * exp(clip(log_gain, -smax, smax))
        return base * torch.exp(torch.clamp(log_gain, min=-smax, max=smax))

    def _prepare_B_for_scan(
        self,
        delta: torch.Tensor,
        B_token: torch.Tensor,
        lambda_pos: torch.Tensor,
    ) -> torch.Tensor:
        # delta: [B,D,L], B_token: [B,N,L], lambda_pos: [D,N]
        if not self.exact_zoh:
            return B_token

        # exact ZOH: Bbar = ((1 - exp(-lambda*dt))/lambda) * B
        # selective_scan uses delta * B_scan * u, so
        # B_scan = ((1 - exp(-lambda*dt)) / (lambda * dt)) * B
        lam = lambda_pos.to(dtype=delta.dtype)
        delta_lambda = torch.einsum("bdl,dn->bdnl", delta, lam)
        # Stable ratio: (1 - exp(-x)) / x, with first-order series near x=0.
        eps = torch.tensor(1e-4, device=delta.device, dtype=delta.dtype)
        one = torch.tensor(1.0, device=delta.device, dtype=delta.dtype)
        half = torch.tensor(0.5, device=delta.device, dtype=delta.dtype)
        tiny = torch.tensor(1e-8, device=delta.device, dtype=delta.dtype)
        ratio = torch.where(
            delta_lambda.abs() < eps,
            one - half * delta_lambda,
            -torch.expm1(-delta_lambda) / delta_lambda.clamp_min(tiny),
        )
        return (B_token.unsqueeze(1) * ratio).to(dtype=delta.dtype)

    def forward(
        self,
        x: torch.Tensor,
        band_indices: torch.Tensor,
        mask_fraction: torch.Tensor,
        rho_bar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B,L,D]
            band_indices: [L]
            mask_fraction: [B,L] in [0,1]
            rho_bar: [L] in [0,1]
        """
        bsz, seqlen, _ = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_proj.transpose(1, 2))[..., :seqlen]
        x_conv = F.silu(x_conv)

        x_flat = x_conv.transpose(1, 2).reshape(bsz * seqlen, self.d_inner)
        x_dbl = self.x_proj(x_flat)
        dt_token, B_token, C_token = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt_pre = self.dt_proj(dt_token).reshape(bsz, seqlen, self.d_inner)
        B_pre = B_token.reshape(bsz, seqlen, self.d_state)
        C_pre = C_token.reshape(bsz, seqlen, self.d_state)

        # Base Delta from selective SSM, then bounded multiplicative log-gain modulation.
        dt_base = F.softplus(dt_pre)
        if self.modulate_delta:
            g_dt = self._sum_log_gain(
                band_indices,
                rho_bar,
                mask_fraction,
                s_band=self.s_band_dt,
                s_rho=self.s_rho_dt,
                s_mask=self.s_mask_dt,
                out_dim=self.d_inner,
                dtype=dt_base.dtype,
            )
            dt = self._apply_bounded_log_gain(dt_base, g_dt, self.smax_dt)
        else:
            dt = dt_base

        if self.modulate_B:
            g_B = self._sum_log_gain(
                band_indices,
                rho_bar,
                mask_fraction,
                s_band=self.s_band_B,
                s_rho=self.s_rho_B,
                s_mask=self.s_mask_B,
                out_dim=self.d_state,
                dtype=B_pre.dtype,
            )
            B_pre = self._apply_bounded_log_gain(B_pre, g_B, self.smax_B)

        if self.modulate_C:
            g_C = self._sum_log_gain(
                band_indices,
                rho_bar,
                mask_fraction,
                s_band=self.s_band_C,
                s_rho=self.s_rho_C,
                s_mask=self.s_mask_C,
                out_dim=self.d_state,
                dtype=C_pre.dtype,
            )
            C_pre = self._apply_bounded_log_gain(C_pre, g_C, self.smax_C)

        u = x_conv
        work_dtype = u.dtype

        delta = dt.transpose(1, 2).to(dtype=work_dtype).contiguous()  # [B,D,L]
        B_tok = B_pre.transpose(1, 2).to(dtype=work_dtype).contiguous()  # [B,N,L]
        C_tok = C_pre.transpose(1, 2).to(dtype=work_dtype).contiguous()  # [B,N,L]
        z = z.transpose(1, 2).to(dtype=work_dtype).contiguous()

        lambda_pos = torch.exp(self.lambda_log.float()).clamp_min(1e-4)  # [D,N]
        A = -lambda_pos
        B_scan = self._prepare_B_for_scan(delta, B_tok, lambda_pos)

        y = _selective_scan(u, delta, A, B_scan.contiguous(), C_tok.contiguous(), self.D.float(), z=z)
        y = y.transpose(1, 2)
        return self.out_proj(y)


class BiFCSSM(nn.Module):
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
        use_band: bool = True,
        use_mask: bool = True,
        use_radius: bool = True,
        exact_zoh: bool = True,
        smax_dt: float = 2.0,
        smax_B: float = 2.0,
        smax_C: float = 2.0,
    ):
        super().__init__()
        self.fwd = FCSSM(
            d_model,
            d_state,
            d_conv,
            expand,
            num_bands,
            modulate_delta=modulate_delta,
            modulate_B=modulate_B,
            modulate_C=modulate_C,
            use_band=use_band,
            use_mask=use_mask,
            use_radius=use_radius,
            exact_zoh=exact_zoh,
            smax_dt=smax_dt,
            smax_B=smax_B,
            smax_C=smax_C,
        )
        self.bwd = FCSSM(
            d_model,
            d_state,
            d_conv,
            expand,
            num_bands,
            modulate_delta=modulate_delta,
            modulate_B=modulate_B,
            modulate_C=modulate_C,
            use_band=use_band,
            use_mask=use_mask,
            use_radius=use_radius,
            exact_zoh=exact_zoh,
            smax_dt=smax_dt,
            smax_B=smax_B,
            smax_C=smax_C,
        )
        self.fuse = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        band_indices: torch.Tensor,
        mask_fraction: torch.Tensor,
        rho_bar: torch.Tensor,
    ) -> torch.Tensor:
        y_f = self.fwd(x, band_indices, mask_fraction, rho_bar)

        x_r = x.flip(1)
        b_r = band_indices.flip(0)
        m_r = mask_fraction.flip(1)
        r_r = rho_bar.flip(0)
        y_b = self.bwd(x_r, b_r, m_r, r_r).flip(1)

        return self.fuse(torch.cat([y_f, y_b], dim=-1))
