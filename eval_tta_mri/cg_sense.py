"""Complex CG-SENSE utilities for PaDIS-MRI Scan-TTA."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class CGSenseInfo:
    residual_before: float
    residual_after: float
    iterations: int


@dataclass(frozen=True)
class CGProxInfo:
    measurement_residual_before: float
    measurement_residual_after: float
    prior_mse_after: float
    iterations: int
    gamma: float


def _as_single_channel(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(image):
        raise TypeError("CG-SENSE image tensors must be complex")
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(
            f"Expected complex image [B,1,H,W], got {tuple(image.shape)}"
        )
    return image


def sense_forward(
    image: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply A(x) = M F S x using the repository's orthonormal FFT convention."""
    image = _as_single_channel(image)
    coil_images = maps * image
    return mask * torch.fft.fft2(
        coil_images, dim=(-2, -1), norm="ortho"
    )


def sense_adjoint(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply A^H(y) = sum_c conj(S_c) F^H(M y_c)."""
    coil_images = torch.fft.ifft2(
        mask * kspace, dim=(-2, -1), norm="ortho"
    )
    return torch.sum(
        torch.conj(maps) * coil_images, dim=1, keepdim=True
    )


def measurement_residual_sse(
    image: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    residual = sense_forward(image, maps, mask) - measurement
    return torch.sum(torch.abs(residual).square())


def measurement_residual_mse(
    image: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean acquired-k-space squared residual.

    This is the self-supervised loss used for differentiable Scan-TTA.
    """
    residual = sense_forward(image, maps, mask) - measurement
    return torch.mean(torch.abs(residual).square())


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
    """Historical stop-gradient CG projection on A^H A z = A^H y.

    Kept for smoke-test/backward compatibility with the first projected-target
    Scan-TTA implementation. The new TTA refinement path uses
    ``cg_sense_prox_differentiable`` below.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")

    z = _as_single_channel(z0).detach().clone()
    measurement = measurement.detach()
    maps = maps.detach()
    mask = mask.detach()

    def normal(image: torch.Tensor) -> torch.Tensor:
        return sense_adjoint(
            sense_forward(image, maps, mask), maps, mask
        )

    rhs = sense_adjoint(measurement, maps, mask)
    r = rhs - normal(z)
    p = r.clone()
    rr = torch.real(torch.sum(torch.conj(r) * r))
    eps = torch.finfo(z.real.dtype).eps

    residual_before = (
        measurement_residual_sse(
            z, measurement, maps, mask
        )
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
        residual_after = measurement_residual_sse(
            z, measurement, maps, mask
        )
        info = CGSenseInfo(
            residual_before=float(residual_before.item()),
            residual_after=float(residual_after.item()),
            iterations=int(iterations),
        )
    return z.detach(), info


def cg_sense_prox_differentiable(
    z0: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    iterations: int = 5,
    *,
    gamma: float = 1.0,
    return_info: bool = False,
) -> Tuple[torch.Tensor, Optional[CGProxInfo]]:
    r"""Finite differentiable CG for the conditional denoiser.

    Solves, for a fixed number of CG iterations,

        argmin_z 0.5 * ||z - z0||_2^2
               + 0.5 * gamma * ||A z - y||_2^2

    whose normal equation is

        (I + gamma A^H A) z = z0 + gamma A^H y.

    Crucially, ``z0`` is NOT detached. Autograd therefore propagates the
    measurement loss through the CG iterations and back into the denoiser.
    Measurement, sensitivity maps and mask are treated as constants.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    z0 = _as_single_channel(z0)
    measurement_const = measurement.detach()
    maps_const = maps.detach()
    mask_const = mask.detach()
    gamma_t = z0.real.new_tensor(float(gamma))

    def aha(image: torch.Tensor) -> torch.Tensor:
        return sense_adjoint(
            sense_forward(image, maps_const, mask_const),
            maps_const,
            mask_const,
        )

    def system(image: torch.Tensor) -> torch.Tensor:
        return image + gamma_t * aha(image)

    rhs = z0 + gamma_t * sense_adjoint(
        measurement_const, maps_const, mask_const
    )

    # Initialize with the unconditional denoiser output, as in the TTA method.
    z = z0
    r = rhs - system(z)
    p = r
    rr = torch.real(torch.sum(torch.conj(r) * r))
    eps = torch.finfo(z0.real.dtype).eps

    residual_before = None
    if return_info:
        residual_before = measurement_residual_sse(
            z0, measurement_const, maps_const, mask_const
        )

    for _ in range(iterations):
        ap = system(p)
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
        residual_after = measurement_residual_sse(
            z, measurement_const, maps_const, mask_const
        )
        prior_mse = torch.mean(torch.abs(z - z0).square())
        info = CGProxInfo(
            measurement_residual_before=float(
                residual_before.detach().item()
            ),
            measurement_residual_after=float(
                residual_after.detach().item()
            ),
            prior_mse_after=float(prior_mse.detach().item()),
            iterations=int(iterations),
            gamma=float(gamma),
        )

    return z, info
