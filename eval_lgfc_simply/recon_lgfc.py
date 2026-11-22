"""PaDIS reconstruction loop with the three LGFC insertion points marked."""

import csv
import os
from typing import Optional

import numpy as np
import torch
import tqdm

from denoise_padding import denoisedFromPatches, getIndices

from .config import LGFCConfig
from .lgfc_core import apply_lgfc, update_reference


DIAGNOSTIC_FIELDS = [
    "outer_step",
    "sigma",
    "reference_to_state_norm",
    "unsampled_k_difference_norm",
    "raw_update_norm",
    "raw_relative_update",
    "clipped_update_norm",
    "clipped_relative_update",
    "update_was_clipped",
    "final_state_update_norm",
]


def _ve_schedule(
    net,
    device: torch.device,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
) -> torch.Tensor:
    indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    schedule = (
        sigma_max ** (1.0 / rho)
        + indices / (num_steps - 1.0)
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    schedule = net.round_sigma(schedule)
    return torch.cat(
        [schedule, torch.zeros(1, dtype=torch.float64, device=device)]
    )


def _diagnostic_row(outer_step: int, sigma: torch.Tensor, stats: dict) -> dict:
    row = {"outer_step": outer_step, "sigma": float(sigma.detach().cpu())}
    for name in DIAGNOSTIC_FIELDS[2:]:
        value = stats[name].detach().cpu().reshape(-1)[0]
        row[name] = bool(value) if name == "update_was_clipped" else float(value)
    return row


def _save_diagnostics(rows: list[dict], save_dir: str, tag: Optional[str]) -> None:
    diagnostic_dir = os.path.join(save_dir, "lgfc_diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    filename = f"{tag or 'sample'}_lgfc.csv"
    with open(os.path.join(diagnostic_dir, filename), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def dps2_lgfc(
    net,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: torch.Tensor,
    lgfc_config: LGFCConfig,
    num_steps: int,
    inner_loops: int,
    zeta: float,
    pad: int,
    patch_size: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
    save_diagnostics: bool = False,
    randn_like=torch.randn_like,
):
    """Run configurable PaDIS sampling with post-inner-loop LGFC."""

    net.eval()
    width = latents.shape[-1]
    patch_count = width // patch_size + 1
    spaced = np.linspace(
        0, (patch_count - 1) * patch_size, patch_count, dtype=int
    )

    x_init = inverseop.adjoint(measurement).detach()
    height, fov_width = x_init.shape[-2:]
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)
    t_steps = _ve_schedule(
        net, x.device, num_steps, sigma_min, sigma_max, rho
    )
    diagnostics = [] if save_diagnostics else None

    for outer_index, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps
    ):
        del t_next
        outer_step = outer_index + 1
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        # LGFC 1/3: reset current-outer reference
        reference = None

        for _ in range(inner_loops):
            indices = getIndices(spaced, patch_count, pad, patch_size)
            x = x.detach().requires_grad_(True)
            x_noisy = x + t_cur * randn_like(x)
            x_real_noisy = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
            denoised_real = denoisedFromPatches(
                net,
                x_real_noisy,
                t_cur,
                latents_pos,
                None,
                indices,
                t_goal=0,
                wrong=False,
            )
            denoised = torch.complex(
                denoised_real[:, 0], denoised_real[:, 1]
            ).unsqueeze(1)

            # LGFC 2/3: update reference from current denoised prediction
            denoised_fov = denoised[
                :, :, pad : pad + height, pad : pad + fov_width
            ]
            reference = update_reference(
                reference,
                denoised_fov,
                lgfc_config.reference_mode,
                lgfc_config.ema_beta,
            )

            score = (denoised - x_noisy) / (t_cur ** 2)
            cropped_x0hat = denoised_real[
                :, :, pad : pad + width, pad : pad + width
            ]
            residual = measurement - inverseop.forward(cropped_x0hat)
            residual_flat = residual.reshape(x.shape[0], -1)
            sse_individual = torch.norm(residual_flat, dim=-1) ** 2
            sse = torch.sum(sse_individual)
            likelihood_grad = torch.autograd.grad(sse, x)[0]
            x = x - (
                zeta / torch.sqrt(sse_individual)[:, None, None, None]
            ) * likelihood_grad

            if outer_index < num_steps - 1:
                x = x + (alpha / 2) * score + torch.sqrt(alpha) * randn_like(x)
            else:
                x = x + (alpha / 2) * score

        should_correct = (
            outer_step >= lgfc_config.start_outer
            and (outer_step - lgfc_config.start_outer)
            % lgfc_config.correction_every
            == 0
        )
        if should_correct:
            # LGFC 3/3: correct the state after all inner loops
            x_fov = x[
                :, :, pad : pad + height, pad : pad + fov_width
            ].detach()
            x_corrected, stats = apply_lgfc(
                x_fov=x_fov,
                reference=reference,
                maps=inverseop.maps,
                mask=inverseop.mask,
                unsampled_weight=lgfc_config.unsampled_weight,
                max_relative_update=lgfc_config.max_relative_update,
                collect_diagnostics=save_diagnostics,
            )
            x_next = x.detach().clone()
            x_next[
                :, :, pad : pad + height, pad : pad + fov_width
            ] = x_corrected
            x = x_next

            if save_diagnostics:
                diagnostics.append(_diagnostic_row(outer_step, t_cur, stats))

    if save_diagnostics:
        _save_diagnostics(diagnostics, save_dir, tag)

    reconstruction = x[
        :, :, pad : pad + width, pad : pad + width
    ].detach().squeeze(1)
    return reconstruction, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
