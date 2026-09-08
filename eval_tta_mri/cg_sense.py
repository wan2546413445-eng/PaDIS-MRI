"""Small, non-differentiable complex CG-SENSE projection used by Scan-TTA."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class CGSenseInfo:
    residual_before: float
    residual_after: float
    iterations: int


def _as_single_channel(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(image):
        raise TypeError("CG-SENSE image tensors must be complex")
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"Expected complex image [B,1,H,W], got {tuple(image.shape)}")
    return image


def sense_forward(
    image: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply A(x) = M F S x using the repository's orthonormal FFT convention."""
    image = _as_single_channel(image)
    coil_images = maps * image
    return mask * torch.fft.fft2(coil_images, dim=(-2, -1), norm="ortho")


def sense_adjoint(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply A^H(y) = sum_c conj(S_c) F^H(M y_c)."""
    coil_images = torch.fft.ifft2(mask * kspace, dim=(-2, -1), norm="ortho")
    return torch.sum(torch.conj(maps) * coil_images, dim=1, keepdim=True)


def measurement_residual_sse(
    image: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    residual = sense_forward(image, maps, mask) - measurement
    return torch.sum(torch.abs(residual).square())


@torch.no_grad()
def cg_sense(
    z0: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    iterations: int = 5,
    *,
    return_info: bool = False,
) -> Tuple[torch.Tensor, Optional[CGSenseInfo]]:
    """Run a fixed number of CG iterations on A^H A z = A^H y.

    The initial image is retained in the null space of the acquired encoding,
    making this a measurement projection of the current prior prediction rather
    than a standalone reconstruction. The complete routine is stop-gradient.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")

    z = _as_single_channel(z0).detach().clone()
    measurement = measurement.detach()
    maps = maps.detach()
    mask = mask.detach()

    def normal(image: torch.Tensor) -> torch.Tensor:
        return sense_adjoint(sense_forward(image, maps, mask), maps, mask)

    rhs = sense_adjoint(measurement, maps, mask)
    r = rhs - normal(z)
    p = r.clone()
    rr = torch.real(torch.sum(torch.conj(r) * r))
    eps = torch.finfo(z.real.dtype).eps
    residual_before = (
        measurement_residual_sse(z, measurement, maps, mask)
        if return_info else None
    )

    for _ in range(iterations):
        ap = normal(p)
        denom = torch.real(torch.sum(torch.conj(p) * ap))
        alpha = rr / (denom + eps)
        z = z + alpha * p
        r_next = r - alpha * ap
        rr_next = torch.real(torch.sum(torch.conj(r_next) * r_next))
        beta = rr_next / (rr + eps)
        p = r_next + beta * p
        r = r_next
        rr = rr_next

    info = None
    if return_info:
        residual_after = measurement_residual_sse(z, measurement, maps, mask)
        info = CGSenseInfo(
            residual_before=float(residual_before.item()),
            residual_after=float(residual_after.item()),
            iterations=int(iterations),
        )
    return z.detach(), info
