from __future__ import annotations

import hashlib
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
try:
    import sigpy.mri as spmri
except Exception:
    spmri = None

from filmamba.utils.fft import fft1c, ifft3c


@dataclass
class CalgaryConfig:
    root: str
    acceleration: int
    split: str
    acs_size: int = 24
    mask_dir: Optional[str] = None
    num_coils: int = 12
    training: bool = True
    fast_sens: bool = True
    sens_method: str = "espirit"  # espirit | ratio
    espirit_calib_width: int = 24
    espirit_thresh: float = 0.02
    espirit_kernel_width: int = 6
    espirit_crop: float = 0.95
    espirit_max_iter: int = 100
    espirit_show_pbar: bool = False
    espirit_timeout_sec: float = 45.0
    sens_cache_dir: Optional[str] = None
    crop_size: Optional[Tuple[int, int, int]] = None
    max_samples: Optional[int] = None
    public_val_count: int = 10
    public_test_count: int = 10


class CalgaryDataset(Dataset):
    def __init__(self, cfg: CalgaryConfig):
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.files = self._resolve_files()
        if not self.files:
            raise FileNotFoundError(f"No .h5 files resolved for split '{cfg.split}' at {self.root}")
        if cfg.max_samples is not None:
            self.files = self.files[: cfg.max_samples]
        self.mask_cache = self._load_masks()
        self.sens_cache_dir = self._resolve_sens_cache_dir()

    def _resolve_files(self) -> List[Path]:
        split = self.cfg.split
        if split == "public_val":
            files = sorted((self.root / "val").glob("*.h5"))
            n_val = int(self.cfg.public_val_count)
            n_test = int(self.cfg.public_test_count)
            if len(files) < n_val + n_test:
                raise ValueError(
                    f"Need at least {n_val+n_test} files in {self.root/'val'} for public 47/10/10 split, found {len(files)}"
                )
            return files[:n_val]
        if split == "public_test":
            files = sorted((self.root / "val").glob("*.h5"))
            n_val = int(self.cfg.public_val_count)
            n_test = int(self.cfg.public_test_count)
            if len(files) < n_val + n_test:
                raise ValueError(
                    f"Need at least {n_val+n_test} files in {self.root/'val'} for public 47/10/10 split, found {len(files)}"
                )
            return files[n_val : n_val + n_test]
        return sorted((self.root / split).glob("*.h5"))

    def _load_masks(self) -> Dict[Tuple[int, int], np.ndarray]:
        mask_dir = Path(self.cfg.mask_dir or (self.root / "poisson_masks"))
        out = {}
        for nz in (170, 174, 180):
            path = mask_dir / f"R{self.cfg.acceleration}_218x{nz}.npy"
            if path.exists():
                out[(218, nz)] = np.load(path)
        if not out:
            raise FileNotFoundError(f"No official masks found in {mask_dir}")
        return out

    def _resolve_sens_cache_dir(self) -> Optional[Path]:
        method = str(getattr(self.cfg, "sens_method", "ratio")).lower()
        if method != "espirit":
            return None
        if self.cfg.sens_cache_dir:
            cache_dir = Path(self.cfg.sens_cache_dir)
        else:
            tag = (
                f"cw{int(self.cfg.espirit_calib_width)}"
                f"_th{float(self.cfg.espirit_thresh):.4f}"
                f"_kw{int(self.cfg.espirit_kernel_width)}"
                f"_cr{float(self.cfg.espirit_crop):.3f}"
            )
            cache_dir = self.root / "sens_cache" / tag
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _sens_cache_path(self, cache_key: str) -> Optional[Path]:
        if self.sens_cache_dir is None:
            return None
        safe = cache_key.replace(os.sep, "_")
        return self.sens_cache_dir / f"{safe}.pt"

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _to_complex(kspace_raw: np.ndarray) -> np.ndarray:
        # Raw Calgary format: [Nx, Ny, Nz, 2*Nc], float32
        if np.iscomplexobj(kspace_raw):
            if kspace_raw.shape[0] <= 32:
                # already [Nc, Nx, Ny, Nz]
                return kspace_raw.astype(np.complex64)
            raise ValueError(f"Unexpected complex shape: {kspace_raw.shape}")
        if kspace_raw.ndim != 4 or kspace_raw.shape[-1] % 2 != 0:
            raise ValueError(f"Unexpected raw shape: {kspace_raw.shape}")
        real = kspace_raw[..., 0::2]
        imag = kspace_raw[..., 1::2]
        kc = (real + 1j * imag).astype(np.complex64)  # [Nx, Ny, Nz, Nc]
        kc = np.transpose(kc, (3, 0, 1, 2))  # [Nc, Nx, Ny, Nz]
        return kc

    def _pick_mask(self, ny: int, nz: int, key: str) -> np.ndarray:
        # Official masks are 2D over (ky, kz), repeated over readout (kx)
        if (ny, nz) not in self.mask_cache:
            raise KeyError(f"Missing mask shape {(ny, nz)} for R={self.cfg.acceleration}")
        pool = self.mask_cache[(ny, nz)]
        if self.cfg.training:
            idx = np.random.randint(0, pool.shape[0])
        else:
            h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
            idx = h % pool.shape[0]
        return pool[idx]

    def _estimate_sens_ratio(self, kspace_full: torch.Tensor) -> torch.Tensor:
        # kspace_full: [Nc, Nx, Ny, Nz] complex
        nc, nx, ny, nz = kspace_full.shape
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        h = self.cfg.acs_size // 2
        if self.cfg.fast_sens:
            # Estimate smooth coil sensitivities from low-res ACS and upsample.
            k_acs = kspace_full[:, cx - h:cx + h, cy - h:cy + h, cz - h:cz + h]
            img_acs = ifft3c(k_acs)
            rss = torch.sqrt((img_acs.abs() ** 2).sum(dim=0, keepdim=True).clamp_min(1e-8))
            sens_low = img_acs / rss
            ri = torch.stack([sens_low.real, sens_low.imag], dim=1).reshape(1, 2 * nc, 2 * h, 2 * h, 2 * h)
            ri = F.interpolate(ri, size=(nx, ny, nz), mode="trilinear", align_corners=False)[0]
            ri = ri.reshape(nc, 2, nx, ny, nz)
            sens = torch.complex(ri[:, 0], ri[:, 1])
            rss_full = torch.sqrt((sens.abs() ** 2).sum(dim=0, keepdim=True).clamp_min(1e-8))
            sens = sens / rss_full
        else:
            k_acs = torch.zeros_like(kspace_full)
            k_acs[:, cx - h:cx + h, cy - h:cy + h, cz - h:cz + h] = kspace_full[:, cx - h:cx + h, cy - h:cy + h, cz - h:cz + h]
            img_acs = ifft3c(k_acs)
            rss = torch.sqrt((img_acs.abs() ** 2).sum(dim=0, keepdim=True).clamp_min(1e-8))
            sens = img_acs / rss
        return sens

    def _estimate_sens_espirit(self, kspace_full: torch.Tensor) -> torch.Tensor:
        if spmri is None:
            raise ImportError("sigpy.mri is required for sens_method='espirit'")

        k_np = np.asarray(kspace_full.detach().cpu().numpy())
        app = spmri.app.EspiritCalib(
            k_np,
            calib_width=int(self.cfg.espirit_calib_width),
            thresh=float(self.cfg.espirit_thresh),
            kernel_width=int(self.cfg.espirit_kernel_width),
            crop=float(self.cfg.espirit_crop),
            max_iter=int(self.cfg.espirit_max_iter),
            show_pbar=bool(self.cfg.espirit_show_pbar),
        )
        sens_np = app.run()
        sens = torch.from_numpy(np.asarray(sens_np)).to(torch.complex64)
        rss = torch.sqrt((sens.abs() ** 2).sum(dim=0, keepdim=True).clamp_min(1e-8))
        return sens / rss

    def _estimate_sens_espirit_with_timeout(self, kspace_full: torch.Tensor) -> torch.Tensor:
        timeout_sec = float(getattr(self.cfg, "espirit_timeout_sec", 0.0))
        if timeout_sec <= 0.0:
            return self._estimate_sens_espirit(kspace_full)
        if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
            return self._estimate_sens_espirit(kspace_full)

        def _timeout_handler(_signum, _frame):
            raise TimeoutError(f"ESPIRiT timeout after {timeout_sec:.1f}s")

        prev_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        try:
            return self._estimate_sens_espirit(kspace_full)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, prev_handler)

    def _estimate_sens(self, kspace_full: torch.Tensor, cache_key: Optional[str] = None) -> torch.Tensor:
        method = str(getattr(self.cfg, "sens_method", "ratio")).lower()
        if method == "espirit":
            cache_path = self._sens_cache_path(cache_key) if cache_key else None
            if cache_path is not None and cache_path.exists():
                try:
                    sens_cached = torch.load(cache_path, map_location="cpu")
                    if torch.is_tensor(sens_cached):
                        return sens_cached.to(torch.complex64)
                except Exception:
                    pass
            from_espirit = True
            try:
                sens = self._estimate_sens_espirit_with_timeout(kspace_full)
            except Exception as e:
                from_espirit = False
                if not hasattr(self, "_espirit_warned"):
                    self._espirit_warned = False
                if not self._espirit_warned:
                    print(f"[CalgaryDataset] ESPIRiT estimation failed, falling back to ratio sensitivities: {repr(e)}", flush=True)
                    self._espirit_warned = True
                sens = self._estimate_sens_ratio(kspace_full)
            # For paper runs, keep ESPIRiT cache pure: do not store fallback ratio maps.
            if cache_path is not None and from_espirit:
                tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp_{os.getpid()}")
                try:
                    torch.save(sens.cpu(), tmp_path)
                    os.replace(tmp_path, cache_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
            return sens
        if method == "ratio":
            return self._estimate_sens_ratio(kspace_full)
        raise ValueError(f"Unsupported sens_method={self.cfg.sens_method}; use 'espirit' or 'ratio'")

    @staticmethod
    def _center_crop_nd(x: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
        # Crop last 3 dims to (sx,sy,sz)
        sx, sy, sz = size
        nx, ny, nz = x.shape[-3:]
        sx = min(sx, nx)
        sy = min(sy, ny)
        sz = min(sz, nz)
        ox = (nx - sx) // 2
        oy = (ny - sy) // 2
        oz = (nz - sz) // 2
        return x[..., ox:ox + sx, oy:oy + sy, oz:oz + sz]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        with h5py.File(path, "r") as f:
            kraw = f["kspace"][:]
        k_np = self._to_complex(kraw)
        kspace = torch.from_numpy(k_np).to(torch.complex64)  # [Nc,Nx,Ny,Nz]

        # Calgary raw is hybrid (x, ky, kz). Convert readout x -> kx.
        kspace = fft1c(kspace, dim=-3)

        _, nx, ny, nz = kspace.shape
        m2d = self._pick_mask(ny, nz, path.stem)
        mask = torch.from_numpy(m2d.astype(np.float32))[None, None, :, :]  # [1,1,Ny,Nz]
        mask = mask.repeat(1, nx, 1, 1)  # [1,Nx,Ny,Nz]

        kspace_under = kspace * mask
        sens = self._estimate_sens(kspace, cache_key=f"{path.parent.name}_{path.stem}")

        if self.cfg.crop_size is not None:
            kspace = self._center_crop_nd(kspace, self.cfg.crop_size)
            kspace_under = self._center_crop_nd(kspace_under, self.cfg.crop_size)
            sens = self._center_crop_nd(sens, self.cfg.crop_size)
            mask = self._center_crop_nd(mask, self.cfg.crop_size)

        img_full_coils = ifft3c(kspace)
        img_under_coils = ifft3c(kspace_under)
        img_gt_c = (img_full_coils * sens.conj()).sum(dim=0)  # [Nx,Ny,Nz]
        img_zf_c = (img_under_coils * sens.conj()).sum(dim=0)
        img_gt_rss = torch.sqrt((img_full_coils.abs() ** 2).sum(dim=0).clamp_min(1e-8))
        img_zf_rss = torch.sqrt((img_under_coils.abs() ** 2).sum(dim=0).clamp_min(1e-8))

        # Use RSS normalization to align with primary challenge-domain evaluation.
        scale = img_gt_rss.amax().clamp_min(1e-6)
        kspace = kspace / scale
        kspace_under = kspace_under / scale
        img_gt_c = img_gt_c / scale
        img_zf_c = img_zf_c / scale
        img_gt_rss = img_gt_rss / scale
        img_zf_rss = img_zf_rss / scale

        return {
            "kspace_full": kspace,
            "kspace_under": kspace_under,
            "mask": mask,
            "sens": sens,
            "img_gt": img_gt_c,
            "img_zf": img_zf_c,
            "img_gt_mag": img_gt_c.abs(),
            "img_gt_rss": img_gt_rss,
            "img_zf_rss": img_zf_rss,
            "filename": path.stem,
            "shape": torch.tensor([nx, ny, nz], dtype=torch.int16),
            "scale": scale.real.to(torch.float32),
        }


def build_loaders(
    data_root: str,
    acceleration: int,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    mask_dir: Optional[str] = None,
    fast_sens: bool = True,
    sens_method: str = "espirit",
    espirit_calib_width: int = 24,
    espirit_thresh: float = 0.02,
    espirit_kernel_width: int = 6,
    espirit_crop: float = 0.95,
    espirit_max_iter: int = 100,
    espirit_show_pbar: bool = False,
    espirit_timeout_sec: float = 45.0,
    sens_cache_dir: Optional[str] = None,
    crop_size: Optional[Tuple[int, int, int]] = None,
    train_crop_size: Optional[Tuple[int, int, int]] = None,
    val_crop_size: Optional[Tuple[int, int, int]] = None,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    eval_split: str = "public_val",
    public_val_count: int = 10,
    public_test_count: int = 10,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    train_crop = train_crop_size if train_crop_size is not None else crop_size
    val_crop = val_crop_size if val_crop_size is not None else crop_size

    train_ds = CalgaryDataset(
        CalgaryConfig(
            root=data_root,
            acceleration=acceleration,
            split="train",
            mask_dir=mask_dir,
            training=True,
            fast_sens=fast_sens,
            sens_method=sens_method,
            espirit_calib_width=espirit_calib_width,
            espirit_thresh=espirit_thresh,
            espirit_kernel_width=espirit_kernel_width,
            espirit_crop=espirit_crop,
            espirit_max_iter=espirit_max_iter,
            espirit_show_pbar=espirit_show_pbar,
            espirit_timeout_sec=espirit_timeout_sec,
            sens_cache_dir=sens_cache_dir,
            crop_size=train_crop,
            max_samples=train_limit,
            public_val_count=public_val_count,
            public_test_count=public_test_count,
        )
    )
    val_ds = CalgaryDataset(
        CalgaryConfig(
            root=data_root,
            acceleration=acceleration,
            split=eval_split,
            mask_dir=mask_dir,
            training=False,
            fast_sens=fast_sens,
            sens_method=sens_method,
            espirit_calib_width=espirit_calib_width,
            espirit_thresh=espirit_thresh,
            espirit_kernel_width=espirit_kernel_width,
            espirit_crop=espirit_crop,
            espirit_max_iter=espirit_max_iter,
            espirit_show_pbar=espirit_show_pbar,
            espirit_timeout_sec=espirit_timeout_sec,
            sens_cache_dir=sens_cache_dir,
            crop_size=val_crop,
            max_samples=val_limit,
            public_val_count=public_val_count,
            public_test_count=public_test_count,
        )
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    def _worker_init(_worker_id: int):
        # Avoid CPU over-subscription from FFT-heavy preprocessing.
        torch.set_num_threads(1)

    common_train_kwargs = {}
    common_val_kwargs = {}
    if num_workers > 0:
        common_train_kwargs["persistent_workers"] = True
        common_train_kwargs["prefetch_factor"] = 2
        common_train_kwargs["worker_init_fn"] = _worker_init
        common_val_kwargs["persistent_workers"] = True
        common_val_kwargs["prefetch_factor"] = 2
        common_val_kwargs["worker_init_fn"] = _worker_init

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **common_train_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(0, num_workers // 2),
        pin_memory=True,
        drop_last=False,
        **common_val_kwargs,
    )
    return train_loader, val_loader, train_sampler, val_sampler
