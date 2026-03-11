from __future__ import annotations

import torch

from filmamba.utils.fft import fft3c, ifft3c


def forward_op(img: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    # img:[B,X,Y,Z], sens:[B,Nc,X,Y,Z]
    return fft3c(img.unsqueeze(1) * sens)


def adjoint_op(kspace: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    # kspace:[B,Nc,X,Y,Z]
    img_coils = ifft3c(kspace)
    return (img_coils * sens.conj()).sum(dim=1)


def data_grad(img: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    pred = forward_op(img, sens)
    resid = (pred - y) * mask
    return adjoint_op(resid, sens)


def soft_dc(k_pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    # mu > 0 scalar
    mu = torch.nn.functional.softplus(mu)
    return torch.where(mask.bool(), (mu * k_pred + y) / (mu + 1.0), k_pred)
