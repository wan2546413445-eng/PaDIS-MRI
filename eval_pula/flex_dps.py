"""Single-step original PaDIS DPS tail: raw measurement and Jacobian."""
from math import sqrt
import torch
from eval_pula.recon_pula import native_complex_noise
from eval_pula.score_opa import real_channels

def dps_step(padded, oracle, raw_operator, raw_measurement, generator,
             sigma, shift, zeta, final_level):
    """Exactly one original PaDIS DPS tail update, including its noise rule."""
    fov = slice(oracle.pad, oracle.pad + oracle.image_size)
    alpha = 0.5 * sigma**2
    with torch.enable_grad():
        current = padded.detach().requires_grad_(True)
        noisy = current + sigma * native_complex_noise(current, generator)
        denoised = oracle.denoise_padded(noisy, sigma, shift)
        denoised_fov = denoised[:, :, fov, fov]
        residual = raw_measurement - raw_operator.forward(real_channels(denoised_fov))
        norm = torch.linalg.vector_norm(residual.reshape(residual.shape[0], -1), dim=-1)
        if not torch.isfinite(norm).all().item() or (norm == 0).any().item():
            raise FloatingPointError('DPS residual norm is zero or nonfinite; original update undefined')
        grad = torch.autograd.grad(norm.square().sum(), current)[0]
        score = (denoised - noisy) / sigma**2
        updated = current - (zeta / norm[:, None, None, None]) * grad
        updated = updated + (alpha / 2) * score
        if not final_level:
            updated = updated + sqrt(alpha) * native_complex_noise(updated, generator)
    return updated.detach(), {'measurement_residual_norm': norm.detach().cpu().tolist()}
