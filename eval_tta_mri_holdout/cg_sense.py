"""Differentiable finite-step proximal CG-SENSE for conditional k-space."""

import torch


def _single_channel(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(image):
        raise TypeError("SENSE image tensors must be complex")
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"Expected [B,1,H,W], got {tuple(image.shape)}")
    return image


def sense_forward_unmasked(image: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    """Apply B=FS with the repository's orthonormal FFT convention."""
    image = _single_channel(image)
    return torch.fft.fft2(maps * image, dim=(-2, -1), norm="ortho")


def sense_forward(
    image: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return mask * sense_forward_unmasked(image, maps)


def sense_adjoint(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    coil_images = torch.fft.ifft2(mask * kspace, dim=(-2, -1), norm="ortho")
    return torch.sum(torch.conj(maps) * coil_images, dim=1, keepdim=True)


def proximal_cg_sense(
    prediction: torch.Tensor,
    measurement_cond: torch.Tensor,
    maps: torch.Tensor,
    mask_cond: torch.Tensor,
    *,
    gamma: float = 1.0,
    iterations: int = 5,
) -> torch.Tensor:
    """Unroll CG for (I + gamma A_c^H A_c)z = D + gamma A_c^H y_c.

    No tensor is detached. Consequently, held-out loss gradients propagate
    through all finite CG iterations to ``prediction`` and the diffusion prior.
    This function has no holdout-mask argument and cannot access held-out data.
    """
    z = _single_channel(prediction)
    gamma_tensor = torch.as_tensor(gamma, dtype=z.real.dtype, device=z.device)

    def system(image: torch.Tensor) -> torch.Tensor:
        encoded = sense_forward(image, maps, mask_cond)
        return image + gamma_tensor * sense_adjoint(encoded, maps, mask_cond)

    rhs = prediction + gamma_tensor * sense_adjoint(
        measurement_cond, maps, mask_cond
    )
    r = rhs - system(z)
    p = r
    rr = torch.real(torch.sum(torch.conj(r) * r))
    eps = torch.finfo(z.real.dtype).eps

    for _ in range(iterations):
        hp = system(p)
        denominator = torch.real(torch.sum(torch.conj(p) * hp))
        alpha = rr / (denominator + eps)
        z = z + alpha * p
        r_next = r - alpha * hp
        rr_next = torch.real(torch.sum(torch.conj(r_next) * r_next))
        beta = rr_next / (rr + eps)
        p = r_next + beta * p
        r = r_next
        rr = rr_next
    return z


def heldout_kspace_loss(
    image: torch.Tensor,
    measurement: torch.Tensor,
    maps: torch.Tensor,
    mask_hold: torch.Tensor,
) -> torch.Tensor:
    """Mean squared complex residual over acquired holdout samples only."""
    predicted_kspace = sense_forward_unmasked(image, maps)
    residual = mask_hold * (predicted_kspace - measurement)
    number_of_coils = predicted_kspace.shape[1]
    denominator = mask_hold.sum() * number_of_coils
    return residual.abs().square().sum() / denominator
