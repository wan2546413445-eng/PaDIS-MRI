"""Core reference aggregation and LGFC frequency update."""

from typing import Optional

import torch

from .config import EPS


def _batch_l2(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.reshape(value.shape[0], -1), dim=1)


def _batch_scale(scale: torch.Tensor, ndim: int) -> torch.Tensor:
    return scale.reshape(scale.shape[0], *([1] * (ndim - 1)))


@torch.no_grad()
def update_reference(
    reference: Optional[torch.Tensor],
    denoised_fov: torch.Tensor,
    reference_mode: str,
    ema_beta: float,
) -> torch.Tensor:
    """Aggregate detached denoiser predictions within the current outer step."""

    current = denoised_fov.detach()
    if reference is None or reference_mode == "last":
        return current
    return ema_beta * reference + (1.0 - ema_beta) * current


@torch.no_grad()
def apply_lgfc(
    x_fov: torch.Tensor,
    reference: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    unsampled_weight: float,
    max_relative_update: float,
    collect_diagnostics: bool = False,
):
    """Correct only unmeasured k-space, then clip the image update by norm."""

    k_x = torch.fft.fft2(maps * x_fov, dim=(-2, -1), norm="ortho")
    k_reference = torch.fft.fft2(
        maps * reference, dim=(-2, -1), norm="ortho"
    )
    delta_k = (1.0 - mask) * (k_reference - k_x)
    delta_coil = torch.fft.ifft2(delta_k, dim=(-2, -1), norm="ortho")

    numerator = torch.sum(torch.conj(maps) * delta_coil, dim=1, keepdim=True)
    denominator = torch.sum(torch.abs(maps) ** 2, dim=1, keepdim=True) + EPS
    raw_update = unsampled_weight * numerator / denominator

    state_norm = _batch_l2(x_fov)
    raw_update_norm = _batch_l2(raw_update)
    raw_relative_update = raw_update_norm / (state_norm + EPS)
    scale = torch.clamp(
        max_relative_update / (raw_relative_update + EPS), max=1.0
    )
    update = raw_update * _batch_scale(scale, raw_update.ndim)
    x_corrected = x_fov + update

    diagnostics = None
    if collect_diagnostics:
        clipped_update_norm = _batch_l2(update)
        diagnostics = {
            "reference_to_state_norm": _batch_l2(reference - x_fov),
            "unsampled_k_difference_norm": _batch_l2(delta_k),
            "raw_update_norm": raw_update_norm,
            "raw_relative_update": raw_relative_update,
            "clipped_update_norm": clipped_update_norm,
            "clipped_relative_update": clipped_update_norm / (state_norm + EPS),
            "update_was_clipped": raw_relative_update > max_relative_update,
            "final_state_update_norm": _batch_l2(x_corrected - x_fov),
        }
    return x_corrected.detach(), diagnostics
