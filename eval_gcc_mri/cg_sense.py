"""Finite-step CG-SENSE conditioning for GCC-PaDIS."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class CGInfo:
    measurement_residual_before: float
    measurement_residual_after: float
    iterations: int


def _single_channel(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(image):
        raise TypeError("SENSE image tensors must be complex")
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
    """Apply A(x)=MFSx with orthonormal FFT."""
    image = _single_channel(image)
    coil_images = maps * image
    coil_kspace = torch.fft.fft2(coil_images, dim=(-2, -1), norm="ortho")
    return mask * coil_kspace


def sense_adjoint(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply A^H(y)=sum_c conj(S_c) F^H(M y_c)."""
    coil_images = torch.fft.ifft2(
        mask * kspace, dim=(-2, -1), norm="ortho"
    )
    return torch.sum(torch.conj(maps) * coil_images, dim=1, keepdim=True)


def measurement_residual_sse(
    image: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    residual = sense_forward(image, maps, mask) - mask * measurement
    return torch.sum(torch.abs(residual).square())


def acquired_measurement_mse(
    image: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Complex MSE over actual acquired coil samples only."""
    residual = sense_forward(image, maps, mask) - mask * measurement
    acquired_count = torch.count_nonzero(mask) * residual.shape[1]
    return torch.sum(torch.abs(residual).square()) / acquired_count


def _cg_data_consistency_impl(
    prior: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    iterations: int,
    return_info: bool,
) -> Tuple[torch.Tensor, Optional[CGInfo]]:
    """Finite CG for A^H A z = A^H y, initialized at the diffusion prior."""
    prior = _single_channel(prior)

    def normal(image: torch.Tensor) -> torch.Tensor:
        return sense_adjoint(sense_forward(image, maps, mask), maps, mask)

    rhs = sense_adjoint(measurement, maps, mask)

    # Starting from D is essential: finite CG corrects measurement-observable
    # directions while carrying the patch prior through unresolved directions.
    solution = prior
    residual = rhs - normal(solution)
    direction = residual
    residual_norm = torch.real(torch.sum(torch.conj(residual) * residual))
    eps = torch.finfo(prior.real.dtype).eps

    before = None
    if return_info:
        before = measurement_residual_sse(prior, measurement, maps, mask)

    for _ in range(iterations):
        normal_direction = normal(direction)
        denominator = torch.real(
            torch.sum(torch.conj(direction) * normal_direction)
        )
        step = residual_norm / (denominator + eps)
        solution = solution + step * direction

        next_residual = residual - step * normal_direction
        next_norm = torch.real(
            torch.sum(torch.conj(next_residual) * next_residual)
        )
        momentum = next_norm / (residual_norm + eps)
        direction = next_residual + momentum * direction
        residual = next_residual
        residual_norm = next_norm

    info = None
    if return_info:
        after = measurement_residual_sse(
            solution, measurement, maps, mask
        )
        info = CGInfo(
            measurement_residual_before=float(before.detach().item()),
            measurement_residual_after=float(after.detach().item()),
            iterations=int(iterations),
        )
    return solution, info


def cg_data_consistency(
    prior: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    iterations: int = 5,
    *,
    differentiable: bool = False,
    return_info: bool = False,
) -> Tuple[torch.Tensor, Optional[CGInfo]]:
    """Strong full-image data conditioning from a patch-diffusion initialization.

    Solves the normal equations A^H A z = A^H y with z0=prior.
    TTA uses the differentiable finite unroll; formal reconstruction runs
    without an autograd graph.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")

    measurement_const = measurement.detach()
    maps_const = maps.detach()
    mask_const = mask.detach()

    if differentiable:
        return _cg_data_consistency_impl(
            prior,
            measurement_const,
            maps_const,
            mask_const,
            iterations,
            return_info,
        )

    with torch.no_grad():
        solution, info = _cg_data_consistency_impl(
            prior.detach(),
            measurement_const,
            maps_const,
            mask_const,
            iterations,
            return_info,
        )
    return solution.detach(), info
