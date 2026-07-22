"""Training-free Late-Stage Global Frequency Correction utilities.

LGFC uses only denoiser outputs already produced in the current outer step and
the acquired measurements.  The temporal reference is not a strict
same-state, multi-partition consensus and never contains true unmeasured
k-space.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class LGFCConfig:
    """Configuration for one late-stage global frequency correction."""

    enabled: bool = False
    reference_mode: str = "ema"
    ema_beta: float = 0.8
    start_outer: int = 40
    correction_every: int = 1
    unsampled_weight: float = 0.25
    max_relative_update: float = 0.05
    dc_growth_limit: float = 1.02
    backtrack_factor: float = 0.5
    max_backtracks: int = 3
    eps: float = 1e-8


def validate_lgfc_config(config: LGFCConfig) -> LGFCConfig:
    """Validate ``config`` and return it unchanged."""

    if not isinstance(config, LGFCConfig):
        raise TypeError(f"config must be LGFCConfig, got {type(config).__name__}")
    if config.reference_mode not in {"last", "ema"}:
        raise ValueError("reference_mode must be 'last' or 'ema'")
    if not 0.0 <= config.ema_beta < 1.0:
        raise ValueError("ema_beta must satisfy 0 <= ema_beta < 1")
    if config.start_outer < 1:
        raise ValueError("start_outer must be at least 1")
    if config.correction_every < 1:
        raise ValueError("correction_every must be at least 1")
    if config.unsampled_weight < 0.0:
        raise ValueError("unsampled_weight must be non-negative")
    if config.max_relative_update <= 0.0:
        raise ValueError("max_relative_update must be positive")
    if config.dc_growth_limit < 1.0:
        raise ValueError("dc_growth_limit must be at least 1")
    if not 0.0 < config.backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must satisfy 0 < value < 1")
    if config.max_backtracks < 0:
        raise ValueError("max_backtracks must be non-negative")
    if config.eps <= 0.0:
        raise ValueError("eps must be positive")
    return config


def canonicalize_mask(
    mask: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    """Return a finite sampling mask with shape ``[B,1,H,W]``."""

    if not torch.is_tensor(mask):
        raise TypeError("mask must be a torch.Tensor")
    if torch.is_complex(mask):
        if not torch.all(mask.imag == 0):
            raise ValueError("mask must not contain a nonzero imaginary component")
        mask = mask.real

    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        if mask.shape[0] not in {1, batch_size}:
            raise ValueError("3D mask must have shape [1,H,W] or [B,H,W]")
        mask = mask[:, None]
    elif mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError("4D mask must have shape [B,1,H,W]")
    else:
        raise ValueError("mask must have 2, 3, or 4 dimensions")

    if tuple(mask.shape[-2:]) != (height, width):
        raise ValueError(
            f"mask spatial shape must be {(height, width)}, got {tuple(mask.shape[-2:])}"
        )
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1)
    elif mask.shape[0] != batch_size:
        raise ValueError(f"mask batch must be 1 or {batch_size}, got {mask.shape[0]}")

    mask = mask.to(device=device, dtype=real_dtype)
    if not torch.isfinite(mask).all():
        raise ValueError("mask contains NaN or Inf")
    if torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError("mask values must lie in [0,1]")
    return mask


def _validate_complex_image(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W], got {tuple(value.shape)}")
    if not torch.is_complex(value):
        raise TypeError(f"{name} must have a complex dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _canonicalize_maps(maps: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(maps):
        raise TypeError("maps must be a torch.Tensor")
    if maps.ndim == 3:
        maps = maps[None]
    if maps.ndim != 4 or not torch.is_complex(maps):
        raise ValueError("maps must be complex with shape [B,C,H,W]")
    batch, _, height, width = image.shape
    if tuple(maps.shape[-2:]) != (height, width):
        raise ValueError("maps and image spatial dimensions do not match")
    if maps.shape[0] == 1 and batch > 1:
        maps = maps.expand(batch, -1, -1, -1)
    elif maps.shape[0] != batch:
        raise ValueError(f"maps batch must be 1 or {batch}, got {maps.shape[0]}")
    if maps.device != image.device:
        raise ValueError("maps and image must be on the same device")
    if maps.dtype != image.dtype:
        maps = maps.to(dtype=image.dtype)
    if not torch.isfinite(maps).all():
        raise ValueError("maps contain NaN or Inf")
    return maps


def _batch_l2(value: torch.Tensor) -> torch.Tensor:
    """Return one Euclidean norm per batch element."""

    return torch.linalg.vector_norm(value.reshape(value.shape[0], -1), dim=1)


def _broadcast_batch_scale(scale: torch.Tensor, ndim: int) -> torch.Tensor:
    return scale.reshape(scale.shape[0], *([1] * (ndim - 1)))


@torch.no_grad()
def update_reference(
    reference: Optional[torch.Tensor],
    denoised_fov: torch.Tensor,
    mode: str,
    ema_beta: float,
) -> torch.Tensor:
    """Update a detached temporal reference for the current outer step."""

    if mode not in {"last", "ema"}:
        raise ValueError("mode must be 'last' or 'ema'")
    if not 0.0 <= ema_beta < 1.0:
        raise ValueError("ema_beta must satisfy 0 <= ema_beta < 1")
    _validate_complex_image("denoised_fov", denoised_fov)

    current = denoised_fov.detach()
    if reference is None or mode == "last":
        return current
    _validate_complex_image("reference", reference)
    if reference.shape != current.shape:
        raise ValueError("reference and denoised_fov must have identical shapes")
    if reference.device != current.device or reference.dtype != current.dtype:
        raise ValueError("reference and denoised_fov must share device and dtype")
    return (ema_beta * reference.detach() + (1.0 - ema_beta) * current).detach()


@torch.no_grad()
def compute_full_multicoil_kspace(
    image: torch.Tensor,
    maps: torch.Tensor,
) -> torch.Tensor:
    """Encode ``image=[B,1,H,W]`` as full multi-coil orthonormal k-space."""

    _validate_complex_image("image", image)
    maps = _canonicalize_maps(maps, image)
    return torch.fft.fft2(maps * image, dim=(-2, -1), norm="ortho")


def _validate_measurement(measurement: torch.Tensor, encoded: torch.Tensor) -> None:
    if not torch.is_tensor(measurement):
        raise TypeError("measurement must be a torch.Tensor")
    if measurement.shape != encoded.shape:
        raise ValueError(
            f"measurement shape {tuple(measurement.shape)} must match "
            f"encoded shape {tuple(encoded.shape)}"
        )
    if not torch.is_complex(measurement):
        raise TypeError("measurement must have a complex dtype")
    if measurement.device != encoded.device:
        raise ValueError("measurement and image must be on the same device")
    if not torch.isfinite(measurement).all():
        raise ValueError("measurement contains NaN or Inf")


@torch.no_grad()
def compute_dc_residual(
    x_fov: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    measurement: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return ``||measurement-M FFT(Sx)||_2`` for each batch sample.

    ``eps`` is validated for API consistency but is not added to the residual.
    It is used by the caller only when forming ratios and acceptance bounds.
    """

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    encoded = compute_full_multicoil_kspace(x_fov, maps)
    _validate_measurement(measurement, encoded)
    batch, _, height, width = x_fov.shape
    mask4 = canonicalize_mask(
        mask,
        batch,
        height,
        width,
        device=x_fov.device,
        real_dtype=x_fov.real.dtype,
    )
    residual = measurement.to(dtype=encoded.dtype) - mask4 * encoded
    return _batch_l2(residual).detach()


@torch.no_grad()
def apply_lgfc_frequency_correction(
    x_fov: torch.Tensor,
    reference: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    measurement: torch.Tensor,
    unsampled_weight: float = 0.25,
    max_relative_update: float = 0.05,
    dc_growth_limit: float = 1.02,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 3,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply a clipped LGFC update with sampled-data residual backtracking.

    All decisions are made independently per batch sample. If no candidate
    satisfies the DC bound, that sample is returned unchanged. The function
    uses neither ground truth nor fully sampled k-space.
    """

    config = validate_lgfc_config(
        LGFCConfig(
            enabled=True,
            unsampled_weight=unsampled_weight,
            max_relative_update=max_relative_update,
            dc_growth_limit=dc_growth_limit,
            backtrack_factor=backtrack_factor,
            max_backtracks=max_backtracks,
            eps=eps,
        )
    )
    _validate_complex_image("x_fov", x_fov)
    _validate_complex_image("reference", reference)
    if reference.shape != x_fov.shape:
        raise ValueError("reference and x_fov must have identical shapes")
    if reference.device != x_fov.device or reference.dtype != x_fov.dtype:
        raise ValueError("reference and x_fov must share device and dtype")

    maps = _canonicalize_maps(maps, x_fov)
    batch, _, height, width = x_fov.shape
    mask4 = canonicalize_mask(
        mask,
        batch,
        height,
        width,
        device=x_fov.device,
        real_dtype=x_fov.real.dtype,
    )
    k_x = compute_full_multicoil_kspace(x_fov, maps)
    k_reference = compute_full_multicoil_kspace(reference, maps)
    _validate_measurement(measurement, k_x)
    measurement = measurement.to(dtype=k_x.dtype)

    delta_k = (1.0 - mask4) * (k_reference - k_x)
    delta_coil = torch.fft.ifft2(delta_k, dim=(-2, -1), norm="ortho")
    numerator = torch.sum(torch.conj(maps) * delta_coil, dim=1, keepdim=True)
    denominator = torch.sum(torch.abs(maps) ** 2, dim=1, keepdim=True) + config.eps
    raw_update = config.unsampled_weight * numerator / denominator

    x_norm = _batch_l2(x_fov)
    raw_update_norm = _batch_l2(raw_update)
    raw_relative_update = raw_update_norm / (x_norm + config.eps)
    clip_scale = torch.clamp(
        config.max_relative_update / (raw_relative_update + config.eps),
        max=1.0,
    )
    clipped_update = raw_update * _broadcast_batch_scale(clip_scale, raw_update.ndim)
    clipped_update_norm = _batch_l2(clipped_update)
    clipped_relative_update = clipped_update_norm / (x_norm + config.eps)
    update_was_clipped = raw_relative_update > config.max_relative_update

    dc_before = _batch_l2(measurement - mask4 * k_x)
    accepted = torch.zeros(batch, dtype=torch.bool, device=x_fov.device)
    effective_scale = torch.zeros(batch, dtype=x_fov.real.dtype, device=x_fov.device)
    backtrack_count = torch.full(
        (batch,), config.max_backtracks, dtype=torch.int64, device=x_fov.device
    )
    dc_after = dc_before.clone()
    x_out = x_fov.clone()

    for attempt in range(config.max_backtracks + 1):
        scale_value = config.backtrack_factor ** attempt
        candidate = x_fov + scale_value * clipped_update
        candidate_k = compute_full_multicoil_kspace(candidate, maps)
        candidate_dc = _batch_l2(measurement - mask4 * candidate_k)
        newly_accepted = (~accepted) & (
            candidate_dc <= config.dc_growth_limit * dc_before + config.eps
        )
        if torch.any(newly_accepted):
            selector = _broadcast_batch_scale(newly_accepted, x_out.ndim)
            x_out = torch.where(selector, candidate, x_out)
            effective_scale = torch.where(
                newly_accepted,
                torch.full_like(effective_scale, scale_value),
                effective_scale,
            )
            backtrack_count = torch.where(
                newly_accepted,
                torch.full_like(backtrack_count, attempt),
                backtrack_count,
            )
            dc_after = torch.where(newly_accepted, candidate_dc, dc_after)
            accepted = accepted | newly_accepted
        if bool(torch.all(accepted)):
            break

    final_update_norm = _batch_l2(x_out - x_fov)
    dc_relative_change = (dc_after - dc_before) / (dc_before + config.eps)
    if not torch.isfinite(x_out).all():
        raise FloatingPointError("LGFC output contains NaN or Inf")

    diagnostics = {
        "reference_to_state_norm": _batch_l2(reference - x_fov),
        "unsampled_k_difference_norm": _batch_l2(delta_k),
        "raw_update_norm": raw_update_norm,
        "raw_relative_update": raw_relative_update,
        "clipped_update_norm": clipped_update_norm,
        "clipped_relative_update": clipped_relative_update,
        "update_was_clipped": update_was_clipped,
        "effective_scale": effective_scale,
        "backtrack_count": backtrack_count,
        "correction_accepted": accepted,
        "dc_before": dc_before,
        "dc_after": dc_after,
        "dc_relative_change": dc_relative_change,
        "final_state_update_norm": final_update_norm,
    }
    return x_out.to(dtype=x_fov.dtype).detach(), {
        key: value.detach() for key, value in diagnostics.items()
    }
