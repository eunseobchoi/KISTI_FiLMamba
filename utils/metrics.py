from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    # Official Calgary challenge implementation uses sewar.full_ref.vifp.
    from sewar.full_ref import vifp as _sewar_vifp
except Exception:
    _sewar_vifp = None


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean() if x.numel() > 0 else torch.tensor(0.0, device=x.device)


def nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = torch.sum((pred - target) ** 2, dim=(-3, -2, -1))
    den = torch.sum(target ** 2, dim=(-3, -2, -1)).clamp_min(1e-12)
    return _safe_mean(num / den)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(-3, -2, -1)).clamp_min(1e-12)
    return _safe_mean(20.0 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10.0 * torch.log10(mse))


def _gaussian_kernel_1d(size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_3d_single(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    # pred/target: [B,1,X,Y,Z]
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    k = _gaussian_kernel_1d(device=pred.device, dtype=pred.dtype)
    kx = k.view(1, 1, -1, 1, 1)
    ky = k.view(1, 1, 1, -1, 1)
    kz = k.view(1, 1, 1, 1, -1)

    def filt(x):
        x = F.conv3d(x, kx, padding=(k.numel() // 2, 0, 0), groups=1)
        x = F.conv3d(x, ky, padding=(0, k.numel() // 2, 0), groups=1)
        x = F.conv3d(x, kz, padding=(0, 0, k.numel() // 2), groups=1)
        return x

    mu_x = filt(pred)
    mu_y = filt(target)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = filt(pred * pred) - mu_x2
    sigma_y2 = filt(target * target) - mu_y2
    sigma_xy = filt(pred * target) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (num / den.clamp_min(1e-12)).mean(dim=(1, 2, 3, 4))


def ssim3d(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)
    vals = _ssim_3d_single(pred, target, data_range=data_range)
    return vals.mean()


def metric_dict(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> Dict[str, float]:
    pred = pred.float()
    target = target.float()
    return {
        "psnr": float(psnr(pred, target, data_range=data_range).item()),
        "ssim": float(ssim3d(pred, target, data_range=data_range).item()),
        "nmse": float(nmse(pred, target).item()),
    }


def sense_metric_dict(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> Dict[str, float]:
    """Secondary SENSE-domain metrics (3D PSNR/SSIM/NMSE)."""
    d = metric_dict(pred, target, data_range=data_range)
    return {
        "sense_psnr": d["psnr"],
        "sense_ssim3d": d["ssim"],
        "sense_nmse": d["nmse"],
    }


def _crop_challenge_x_slices(vol: np.ndarray, crop_slices: int) -> np.ndarray:
    if crop_slices <= 0:
        return vol
    if vol.shape[0] <= 2 * crop_slices:
        # Keep full volume for tiny debug crops.
        return vol
    return vol[crop_slices:-crop_slices]


def _gaussian_kernel_2d(ws: int, sigma: float) -> np.ndarray:
    x, y = np.mgrid[-ws // 2 + 1 : ws // 2 + 1, -ws // 2 + 1 : ws // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    den = g.sum()
    if den != 0:
        g /= den
    return g


def _filter2(img: np.ndarray, fltr: np.ndarray, mode: str = "same") -> np.ndarray:
    return signal.convolve2d(img, np.rot90(fltr, 2), mode=mode)


def _vifp_single_2d_fallback(ref: np.ndarray, rec: np.ndarray, sigma_n_sq: float) -> float:
    # sewar-compatible VIF-P fallback (used when sewar is unavailable).
    eps = 1e-10
    ref = ref.astype(np.float64, copy=False)
    rec = rec.astype(np.float64, copy=False)
    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        n = int(2.0 ** (4 - scale + 1) + 1)
        win = _gaussian_kernel_2d(ws=n, sigma=n / 5.0)

        if scale > 1:
            ref = _filter2(ref, win, mode="valid")[::2, ::2]
            rec = _filter2(rec, win, mode="valid")[::2, ::2]

        mu_ref = _filter2(ref, win, mode="valid")
        mu_rec = _filter2(rec, win, mode="valid")

        ref_sum_sq = mu_ref * mu_ref
        rec_sum_sq = mu_rec * mu_rec
        ref_rec_sum_mul = mu_ref * mu_rec

        sigma_ref_sq = _filter2(ref * ref, win, mode="valid") - ref_sum_sq
        sigma_rec_sq = _filter2(rec * rec, win, mode="valid") - rec_sum_sq
        sigma_ref_rec = _filter2(ref * rec, win, mode="valid") - ref_rec_sum_mul

        sigma_ref_sq[sigma_ref_sq < 0] = 0
        sigma_rec_sq[sigma_rec_sq < 0] = 0

        g = sigma_ref_rec / (sigma_ref_sq + eps)
        sv_sq = sigma_rec_sq - g * sigma_ref_rec

        g[sigma_ref_sq < eps] = 0
        sv_sq[sigma_ref_sq < eps] = sigma_rec_sq[sigma_ref_sq < eps]
        sigma_ref_sq[sigma_ref_sq < eps] = 0

        g[sigma_rec_sq < eps] = 0
        sv_sq[sigma_rec_sq < eps] = 0

        sv_sq[g < 0] = sigma_rec_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += float(np.sum(np.log10(1.0 + (g ** 2.0) * sigma_ref_sq / (sv_sq + sigma_n_sq))))
        den += float(np.sum(np.log10(1.0 + sigma_ref_sq / sigma_n_sq)))

    if den <= eps:
        return 1.0
    return float(num / den)


def _vifp_2d(ref: np.ndarray, rec: np.ndarray, sigma_n_sq: float) -> float:
    if _sewar_vifp is not None:
        return float(_sewar_vifp(ref, rec, sigma_nsq=sigma_n_sq))
    return _vifp_single_2d_fallback(ref, rec, sigma_n_sq=sigma_n_sq)


def _challenge_metrics_single_volume(
    pred_vol: np.ndarray,
    target_vol: np.ndarray,
    compute_vif: bool,
    vif_sigma_n_sq: float,
    crop_slices: int,
) -> Dict[str, float]:
    """
    Match Calgary challenge-style RSS metrics:
      - Per-slice pSNR/SSIM/VIF on axis-0 slices
      - Report per-volume mean across slices
    """
    if pred_vol.ndim != 3 or target_vol.ndim != 3:
        raise ValueError(f"Expected 3D volumes [X,Y,Z], got pred={pred_vol.shape}, target={target_vol.shape}")
    if pred_vol.shape != target_vol.shape:
        raise ValueError(f"Shape mismatch pred={pred_vol.shape}, target={target_vol.shape}")

    pred_eval = _crop_challenge_x_slices(pred_vol, crop_slices=crop_slices)
    target_eval = _crop_challenge_x_slices(target_vol, crop_slices=crop_slices)
    if pred_eval.shape != target_eval.shape:
        raise ValueError(f"Cropped shape mismatch pred={pred_eval.shape}, target={target_eval.shape}")

    nslices = pred_eval.shape[0]
    psnr_vals = np.zeros(nslices, dtype=np.float64)
    ssim_vals = np.zeros(nslices, dtype=np.float64)

    for i in range(nslices):
        rec = pred_eval[i]
        ref = target_eval[i]
        data_range = float(max(ref.max(), rec.max()) - min(ref.min(), rec.min()))
        data_range = max(data_range, 1e-8)
        psnr_vals[i] = float(peak_signal_noise_ratio(ref, rec, data_range=data_range))
        ssim_vals[i] = float(structural_similarity(ref, rec, data_range=data_range))

    out = {
        "psnr": float(psnr_vals.mean()),
        "ssim": float(ssim_vals.mean()),
    }

    if compute_vif:
        vif_vals = np.zeros(nslices, dtype=np.float64)
        for i in range(nslices):
            vif_vals[i] = _vifp_2d(target_eval[i], pred_eval[i], sigma_n_sq=float(vif_sigma_n_sq))
        out["vif"] = float(np.mean(vif_vals))
    else:
        out["vif"] = float("nan")

    return out


def challenge_rss_metric_dict(
    pred_rss: torch.Tensor,
    target_rss: torch.Tensor,
    compute_vif: bool = True,
    vif_sigma_n_sq: float = 0.4,
    crop_slices: int = 50,
) -> Dict[str, float]:
    """
    Primary challenge-domain metrics in RSS domain.

    Args:
        pred_rss: [B,X,Y,Z] or [X,Y,Z]
        target_rss: same shape as pred_rss
    Returns:
        Dict with keys: psnr, ssim, vif
    """
    pred = pred_rss.detach().float().cpu().numpy()
    target = target_rss.detach().float().cpu().numpy()
    if pred.ndim == 3:
        pred = pred[None]
        target = target[None]
    if pred.ndim != 4:
        raise ValueError(f"Expected [B,X,Y,Z] or [X,Y,Z], got {pred.shape}")
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch pred={pred.shape}, target={target.shape}")

    scores = [
        _challenge_metrics_single_volume(
            pred[i],
            target[i],
            compute_vif=compute_vif,
            vif_sigma_n_sq=vif_sigma_n_sq,
            crop_slices=int(crop_slices),
        )
        for i in range(pred.shape[0])
    ]
    keys = ["psnr", "ssim", "vif"]
    out = {}
    for k in keys:
        vals = np.array([s[k] for s in scores], dtype=np.float64)
        if np.all(np.isnan(vals)):
            out[k] = float("nan")
        else:
            out[k] = float(np.nanmean(vals))
    return out
