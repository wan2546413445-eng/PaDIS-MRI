"""Global cross-conditioned PaDIS diffusion reconstruction."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import tqdm

try:
    from .cg_sense import proximal_cg
    from .gcc_adapter import GCCScanAdapter
except ImportError:
    from cg_sense import proximal_cg
    from gcc_adapter import GCCScanAdapter


GCC_TTA_EVENTS = (40, 30, 20, 10)


def _ve_base_schedule(
    net: torch.nn.Module,
    num_sigmas: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    index = torch.arange(num_sigmas, dtype=torch.float64, device=device)
    schedule = (
        sigma_max ** (1.0 / rho)
        + (index / (num_sigmas - 1.0))
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    return torch.cat([
        net.round_sigma(schedule),
        torch.zeros(1, dtype=torch.float64, device=device),
    ])


def formal_budget(
    num_steps: int = 78,
    inner_loops: int = 10,
    events: Sequence[int] = GCC_TTA_EVENTS,
    refinement_iters: int = 5,
    cg_iters: int = 5,
) -> Dict[str, int]:
    formal_calls = int(num_steps * inner_loops)
    tta_calls = int(len(tuple(events)) * refinement_iters)
    return {
        "formal_denoiser_calls": formal_calls,
        "tta_denoiser_calls": tta_calls,
        "denoiser_calls": formal_calls + tta_calls,
        "tta_backprops": tta_calls,
        "full_cg_calls": formal_calls,
        "tta_cg_calls": tta_calls,
        "total_cg_iterations": int(
            (formal_calls + tta_calls) * cg_iters
        ),
    }


def conditional_score_from_clean(
    conditional_crop: torch.Tensor,
    unconditional_padded: torch.Tensor,
    x_noisy: torch.Tensor,
    sigma: torch.Tensor,
    pad: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Embed the conditional clean crop and use it as the score target."""
    if conditional_crop.ndim != 4 or conditional_crop.shape[1] != 1:
        raise ValueError("conditional_crop must have shape [B,1,H,W]")
    if unconditional_padded.shape != x_noisy.shape:
        raise ValueError("unconditional_padded and x_noisy shapes must match")

    height, width = conditional_crop.shape[-2:]
    conditional_padded = unconditional_padded.clone()
    conditional_padded[
        :, :, pad:pad + height, pad:pad + width
    ] = conditional_crop
    score = (conditional_padded - x_noisy) / sigma.square()
    return score, conditional_padded


def reconstruct_gcc(
    net: torch.nn.Module,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: torch.Tensor,
    *,
    adapter: GCCScanAdapter,
    optimizer: torch.optim.Adam,
    num_steps: int = 78,
    inner_loops: int = 10,
    sigma_min: float = 0.003,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    pad: int = 64,
    psize: int = 64,
    tta_events: Sequence[int] = GCC_TTA_EVENTS,
    refinement_iters: int = 5,
    cg_iters: int = 5,
    gamma: float = 1.0,
    subject_seed: int = 123,
    randn_like=torch.randn_like,
    device: str = "cuda",
) -> Tuple[torch.Tensor, List[Dict], Dict[str, int]]:
    """Run holdout TTA and full-measurement conditional diffusion jointly."""
    if num_steps < 2 or inner_loops < 1:
        raise ValueError("num_steps >= 2 and inner_loops >= 1 are required")
    if pad != 64 or psize != 64 or latents.shape[-1] != 384:
        raise ValueError("Formal GCC-PaDIS requires image=384, pad=64, patch=64")
    if tuple(tta_events) != GCC_TTA_EVENTS:
        raise ValueError("Formal GCC-PaDIS TTA events are [40,30,20,10]")
    if float(gamma) != adapter.gamma:
        raise ValueError("TTA and formal reconstruction must share gamma")

    # Imported lazily so CPU-only operator smoke tests do not require the full
    # baseline evaluation dependency stack.
    from denoise_padding import denoisedFromPatches, getIndices

    net.eval()
    adapter.freeze()
    image_size = int(latents.shape[-1])
    patches = image_size // psize + 1
    spaced = np.linspace(
        0, (patches - 1) * psize, patches, dtype=int
    )

    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(
        x_init, (pad, pad, pad, pad), "constant", 0
    )
    t_steps = _ve_base_schedule(
        net,
        num_steps,
        sigma_min,
        sigma_max,
        rho,
        torch.device(device),
    )

    diagnostics: List[Dict] = []
    counters = {
        "formal_denoiser_calls": 0,
        "tta_denoiser_calls": 0,
        "denoiser_calls": 0,
        "tta_backprops": 0,
        "full_cg_calls": 0,
        "tta_cg_calls": 0,
        "total_cg_iterations": 0,
    }
    event_number = 0

    for outer_index, (sigma, _next_sigma) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps
    ):
        sigma = sigma.float()
        alpha = 0.5 * sigma.square()
        diffusion_index = num_steps - outer_index

        if diffusion_index in tta_events:
            event_number += 1
            adaptation = adapter.refine(
                current_x=x,
                sigma=sigma,
                latents_pos=latents_pos,
                measurement=measurement,
                maps=inverseop.maps,
                full_mask=inverseop.mask,
                optimizer=optimizer,
                spaced=spaced,
                patches=patches,
                pad=pad,
                psize=psize,
                refinement_iters=refinement_iters,
                cg_iters=cg_iters,
                event=event_number,
                subject_seed=subject_seed,
            )
            diagnostics.extend(adaptation.rows)
            counters["tta_denoiser_calls"] += adaptation.denoiser_calls
            counters["denoiser_calls"] += adaptation.denoiser_calls
            counters["tta_backprops"] += adaptation.backprops
            counters["tta_cg_calls"] += adaptation.cg_calls
            counters["total_cg_iterations"] += adaptation.cg_iterations

        for _inner_index in range(inner_loops):
            # These three draws deliberately remain on the baseline streams:
            # patch offset, noisy-state noise, and diffusion noise.
            indices = getIndices(spaced, patches, pad, psize)
            x = x.detach()
            x_noisy = x + sigma * randn_like(x)
            x_real_noisy = torch.view_as_real(
                x_noisy.squeeze(1)
            ).permute(0, 3, 1, 2)

            with torch.no_grad():
                denoised_real = denoisedFromPatches(
                    net,
                    x_real_noisy,
                    sigma,
                    latents_pos,
                    None,
                    indices,
                    t_goal=0,
                    wrong=False,
                )
                unconditional_padded = torch.complex(
                    denoised_real[:, 0], denoised_real[:, 1]
                ).unsqueeze(1)
                unconditional_crop = unconditional_padded[
                    :,
                    :,
                    pad:pad + image_size,
                    pad:pad + image_size,
                ]

                conditional_crop, _ = proximal_cg(
                    unconditional_crop,
                    measurement,
                    inverseop.maps,
                    inverseop.mask,
                    iterations=cg_iters,
                    gamma=gamma,
                    differentiable=False,
                )
                score_cond, _conditional_padded = (
                    conditional_score_from_clean(
                        conditional_crop,
                        unconditional_padded,
                        x_noisy,
                        sigma,
                        pad,
                    )
                )

                if outer_index < num_steps - 1:
                    x = (
                        x
                        + (alpha / 2) * score_cond
                        + torch.sqrt(alpha) * randn_like(x)
                    )
                else:
                    x = x + (alpha / 2) * score_cond

            counters["formal_denoiser_calls"] += 1
            counters["denoiser_calls"] += 1
            counters["full_cg_calls"] += 1
            counters["total_cg_iterations"] += cg_iters

    expected = formal_budget(
        num_steps=num_steps,
        inner_loops=inner_loops,
        events=GCC_TTA_EVENTS,
        refinement_iters=refinement_iters,
        cg_iters=cg_iters,
    )
    if counters != expected:
        raise RuntimeError(
            f"GCC-PaDIS budget mismatch: actual={counters}, expected={expected}"
        )

    reconstruction = x[
        :, :, pad:pad + image_size, pad:pad + image_size
    ].detach().squeeze(1)
    return reconstruction, diagnostics, counters
