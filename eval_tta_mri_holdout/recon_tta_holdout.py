"""Baseline PaDIS reconstruction with four late cross-mask TTA events."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm

from denoise_padding import denoisedFromPatches, getIndices

try:
    from .scan_tta_holdout import (
        HoldoutScanTTAAdapter,
        TTA_INTERVAL,
        TTA_MAX_DIFFUSION,
    )
except ImportError:
    from scan_tta_holdout import (
        HoldoutScanTTAAdapter,
        TTA_INTERVAL,
        TTA_MAX_DIFFUSION,
    )


def _ve_base_schedule(
    net: torch.nn.Module,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    index = torch.arange(num_steps, dtype=torch.float64, device=device)
    schedule = (
        sigma_max ** (1.0 / rho)
        + index / (num_steps - 1.0)
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    return torch.cat(
        [net.round_sigma(schedule), torch.zeros(1, dtype=torch.float64, device=device)]
    )


def dps2_tta_holdout(
    net: torch.nn.Module,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: torch.Tensor,
    *,
    adapter: HoldoutScanTTAAdapter,
    optimizer: torch.optim.Adam,
    num_steps: int = 78,
    inner_loops: int = 10,
    sigma_min: float = 0.003,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    zeta: float = 3.0,
    pad: int = 64,
    psize: int = 64,
    subject_seed: int = 123,
    randn_like=torch.randn_like,
    device: str = "cuda",
) -> Tuple[torch.Tensor, List[Dict], Dict]:
    """Keep all 780 baseline updates intact and use full Ω for reconstruction."""
    net.eval()
    adapter.freeze()
    image_size = latents.shape[-1]
    patches = image_size // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)
    t_steps = _ve_base_schedule(
        net, num_steps, sigma_min, sigma_max, rho, torch.device(device)
    )

    diagnostics: List[Dict] = []
    counters = {
        "baseline_denoiser_calls": 0,
        "tta_denoiser_calls": 0,
        "tta_backprops": 0,
        "cg_iterations_total": 0,
    }
    event = 0

    for outer_index, (t_cur, _t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps
    ):
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur.square()
        diffusion_index = num_steps - outer_index

        if (
            diffusion_index <= TTA_MAX_DIFFUSION
            and diffusion_index % TTA_INTERVAL == 0
        ):
            event += 1
            summary = adapter.refine_event(
                fixed_x=x,
                sigma=t_cur,
                latents_pos=latents_pos,
                measurement=measurement,
                inverseop=inverseop,
                optimizer=optimizer,
                spaced=spaced,
                patches=patches,
                pad=pad,
                psize=psize,
                event=event,
                outer_step=outer_index + 1,
                diffusion_index=diffusion_index,
                subject_seed=subject_seed,
            )
            diagnostics.extend(summary.rows)
            counters["tta_denoiser_calls"] += summary.denoiser_calls
            counters["tta_backprops"] += summary.backprops
            counters["cg_iterations_total"] += summary.cg_iterations

        # This block follows eval/recon.py::dps2 and always uses full Ω.
        for _inner_index in range(inner_loops):
            indices = getIndices(spaced, patches, pad, psize)
            x = x.detach().requires_grad_(True)
            x_noisy = x + t_cur * randn_like(x)
            x_real_noisy = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
            d_real = denoisedFromPatches(
                net,
                x_real_noisy,
                t_cur,
                latents_pos,
                None,
                indices,
                t_goal=0,
                wrong=False,
            )
            counters["baseline_denoiser_calls"] += 1

            d_complex = torch.complex(d_real[:, 0], d_real[:, 1]).unsqueeze(1)
            score = (d_complex - x_noisy) / t_cur.square()
            d_crop = d_real if pad == 0 else d_real[
                :, :, pad:pad + image_size, pad:pad + image_size
            ]
            residual = measurement - inverseop.forward(d_crop)
            residual_flat = residual.reshape(x.shape[0], -1)
            sse_ind = torch.norm(residual_flat, dim=-1).square()
            sse = torch.sum(sse_ind)
            likelihood_grad = torch.autograd.grad(outputs=sse, inputs=x)[0]
            x = x - (
                zeta / torch.sqrt(sse_ind)[:, None, None, None]
            ) * likelihood_grad

            if outer_index < num_steps - 1:
                x = x + (alpha / 2) * score + torch.sqrt(alpha) * randn_like(x)
            else:
                x = x + (alpha / 2) * score

    counters["tta_events"] = event
    return (
        x[:, :, pad:pad + image_size, pad:pad + image_size].detach().squeeze(1),
        diagnostics,
        counters,
    )
