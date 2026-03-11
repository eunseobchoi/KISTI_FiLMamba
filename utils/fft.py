import torch


def _fftshift(x: torch.Tensor, dim):
    if isinstance(dim, int):
        dim = (dim,)
    shifts = [x.size(d) // 2 for d in dim]
    return torch.roll(x, shifts=shifts, dims=dim)


def _ifftshift(x: torch.Tensor, dim):
    if isinstance(dim, int):
        dim = (dim,)
    shifts = [-(x.size(d) // 2) for d in dim]
    return torch.roll(x, shifts=shifts, dims=dim)


def fftnc(x: torch.Tensor, dim, norm: str = "ortho") -> torch.Tensor:
    x = _ifftshift(x, dim)
    x = torch.fft.fftn(x, dim=dim, norm=norm)
    return _fftshift(x, dim)


def ifftnc(x: torch.Tensor, dim, norm: str = "ortho") -> torch.Tensor:
    x = _ifftshift(x, dim)
    x = torch.fft.ifftn(x, dim=dim, norm=norm)
    return _fftshift(x, dim)


def fft3c(x: torch.Tensor) -> torch.Tensor:
    return fftnc(x, dim=(-3, -2, -1), norm="ortho")


def ifft3c(x: torch.Tensor) -> torch.Tensor:
    return ifftnc(x, dim=(-3, -2, -1), norm="ortho")


def fft1c(x: torch.Tensor, dim: int) -> torch.Tensor:
    return fftnc(x, dim=(dim,), norm="ortho")


def ifft1c(x: torch.Tensor, dim: int) -> torch.Tensor:
    return ifftnc(x, dim=(dim,), norm="ortho")
