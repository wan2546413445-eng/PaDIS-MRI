"""Shared complex proximal CG-SENSE operator for GCC-PaDIS."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class ProximalCGInfo:
    measurement_residual_before: float
    measurement_residual_after: float
    prior_mse_after: float
    iterations: int
    gamma: float


def _single_channel(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(image):
        raise TypeError("SENSE image tensors must be complex")
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
    """Apply A(x) = M F S x with the repository's orthonormal FFT."""
    image = _single_channel(image)
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
    residual = sense_forward(image, maps, mask) - mask * measurement
    return torch.sum(torch.abs(residual).square())


def acquired_measurement_mse(
    image: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared residual over actual acquired coil samples only."""
    residual = sense_forward(image, maps, mask) - mask * measurement
    acquired = (mask != 0).expand_as(residual)
    acquired_count = torch.count_nonzero(acquired)
    if int(acquired_count.item()) == 0:
        raise ValueError("The loss mask contains no acquired samples")
    return torch.sum(torch.abs(residual).square()) / acquired_count


def _proximal_cg_impl(
    prior: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    iterations: int,
    gamma: float,
    return_info: bool,
) -> Tuple[torch.Tensor, Optional[ProximalCGInfo]]:
    prior = _single_channel(prior)
    gamma_t = prior.real.new_tensor(float(gamma))

    def normal(image: torch.Tensor) -> torch.Tensor:
        return sense_adjoint(
            sense_forward(image, maps, mask), maps, mask
        )

    def system(image: torch.Tensor) -> torch.Tensor:
        return image + gamma_t * normal(image)

    rhs = prior + gamma_t * sense_adjoint(measurement, maps, mask)
    solution = prior
    residual = rhs - system(solution)
    direction = residual
    residual_norm = torch.real(
        torch.sum(torch.conj(residual) * residual)
    )
    eps = torch.finfo(prior.real.dtype).eps

    residual_before = None
    if return_info:
        residual_before = measurement_residual_sse(
            prior, measurement, maps, mask
        )

    for _ in range(iterations):
        system_direction = system(direction)
        denominator = torch.real(
            torch.sum(torch.conj(direction) * system_direction)
        )
        step = residual_norm / (denominator + eps)
        solution = solution + step * direction

        next_residual = residual - step * system_direction
        next_norm = torch.real(
            torch.sum(torch.conj(next_residual) * next_residual)
        )
        momentum = next_norm / (residual_norm + eps)
        direction = next_residual + momentum * direction
        residual = next_residual
        residual_norm = next_norm

    info = None
    if return_info:
        residual_after = measurement_residual_sse(
            solution, measurement, maps, mask
        )
        prior_mse = torch.mean(torch.abs(solution - prior).square())
        info = ProximalCGInfo(
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
    return solution, info


def proximal_cg(
    prior: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    iterations: int = 5,
    *,
    gamma: float = 1.0,
    differentiable: bool = False,
    return_info: bool = False,
) -> Tuple[torch.Tensor, Optional[ProximalCGInfo]]:
    r"""Solve (I + gamma A^H A)z = D + gamma A^H y.

    The same finite CG implementation is used in both branches. The TTA path
    sets ``differentiable=True`` so the held-out loss reaches ``prior`` and the
    network. Formal reconstruction runs it without a graph.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    measurement_const = measurement.detach()
    maps_const = maps.detach()
    mask_const = mask.detach()

    if differentiable:
        return _proximal_cg_impl(
            prior,
            measurement_const,
            maps_const,
            mask_const,
            iterations,
            gamma,
            return_info,
        )

    with torch.no_grad():
        solution, info = _proximal_cg_impl(
            prior.detach(),
            measurement_const,
            maps_const,
            mask_const,
            iterations,
            gamma,
            return_info,
        )
    return solution.detach(), info
