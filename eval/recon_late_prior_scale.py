"""Late-stage deterministic prior scaling for PaDIS-MRI."""

import csv
import os
from typing import Optional, Tuple

import numpy as np
import torch
import tqdm

from denoise_padding import denoisedFromPatches, getIndices
from utils import makeFigures


def _ve_base_schedule(net, num_sigmas, sigma_min, sigma_max, rho, device):
    idx = torch.arange(num_sigmas, dtype=torch.float64, device=device)
    t = (
        sigma_max ** (1.0 / rho)
        + idx / (num_sigmas - 1.0)
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t = net.round_sigma(t)
    return torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)])


def dps2_late_prior_scale(
    net,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: Optional[torch.Tensor],
    num_steps: int = 104,
    inner_loops: int = 10,
    sigma_min: float = 0.003,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    zeta: float = 3.0,
    pad: int = 64,
    psize: int = 64,
    randn_like=torch.randn_like,
    verbose: bool = False,
    clean: Optional[torch.Tensor] = None,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
    save_intermediate: bool = False,
    intermediate_every: int = 10,
    late_prior_steps: int = 0,
    late_prior_scale: float = 1.0,
    save_scale_diagnostics: bool = False,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    """
    Scale only the deterministic prior drift in the final outer steps:

        x <- x + beta * (alpha / 2) * score + sqrt(alpha) * noise

    Measurement consistency, random partitions, schedule and random draws are
    otherwise unchanged. beta=1 reproduces the original update.
    """
    if measurement is None:
        raise ValueError("MRI measurement is required")

    num_steps = int(num_steps)
    inner_loops = int(inner_loops)
    late_prior_steps = int(late_prior_steps)
    late_prior_scale = float(late_prior_scale)

    if num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    if inner_loops < 1:
        raise ValueError("inner_loops must be >= 1")
    if not 0 <= late_prior_steps <= num_steps:
        raise ValueError(
            f"late_prior_steps must be in [0,{num_steps}], got {late_prior_steps}"
        )
    if not np.isfinite(late_prior_scale) or late_prior_scale < 0:
        raise ValueError("late_prior_scale must be finite and non-negative")

    net.eval()
    w = latents.shape[-1]
    patches = w // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)
    t_steps = _ve_base_schedule(net, num_steps, sigma_min, sigma_max, rho, device)

    late_start = num_steps - late_prior_steps
    print(
        "[Late Prior Scale] "
        f"last_outer_steps={late_prior_steps} | beta={late_prior_scale:g} | "
        f"first_scaled_outer_step="
        f"{late_start + 1 if late_prior_steps > 0 else 'disabled'}"
    )

    noisypsnr = denoisedpsnr = 0.0
    noisyssim = denoisedssim = 0.0
    noisynrmse = denoisednrmse = 0.0

    safe_tag = tag if tag is not None else "sample"
    intermediate_every = max(1, int(intermediate_every))
    intermediate_rows = []
    intermediate_dir = None
    if save_intermediate and save_dir is not None:
        intermediate_dir = os.path.join(save_dir, "intermediate_vis", safe_tag)
        os.makedirs(intermediate_dir, exist_ok=True)

    scale_rows = []
    scale_csv_path = None
    if save_scale_diagnostics and save_dir is not None:
        diag_dir = os.path.join(save_dir, "late_prior_scale_diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        scale_csv_path = os.path.join(
            diag_dir, f"{safe_tag}_beta{late_prior_scale:g}.csv"
        )

    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=len(t_steps) - 1,
    ):
        del t_next
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2
        in_scaled_region = late_prior_steps > 0 and i >= late_start
        beta = late_prior_scale if in_scaled_region else 1.0

        if i == late_start and late_prior_steps > 0:
            print(
                f"[Late Prior Scale] Entering scaled region at "
                f"outer step {i + 1}/{num_steps}, beta={beta:g}"
            )

        for j in range(inner_loops):
            # Preserve original RNG order: patch offset before Gaussian noise.
            indices = getIndices(spaced, patches, pad, psize)
            x = x.detach().requires_grad_(True)

            x_noisy = x + t_cur * randn_like(x)
            x_real_noisy = torch.view_as_real(
                x_noisy.squeeze(1)
            ).permute(0, 3, 1, 2)

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

            d_cplx = torch.complex(d_real[:, 0], d_real[:, 1]).unsqueeze(1)
            score = (d_cplx - x_noisy) / (t_cur ** 2)

            cropped_x0hat = (
                d_real if pad == 0 else d_real[:, :, pad:pad + w, pad:pad + w]
            )
            ax = inverseop.forward(cropped_x0hat)
            residual = measurement - ax
            residual_flat = residual.reshape(x.shape[0], -1)
            sse_ind = torch.norm(residual_flat, dim=-1) ** 2
            sse = torch.sum(sse_ind)

            likelihood_grad = torch.autograd.grad(outputs=sse, inputs=x)[0]
            dc_update = -(
                zeta / torch.sqrt(sse_ind)[:, None, None, None]
            ) * likelihood_grad
            prior_update_unscaled = (alpha / 2) * score
            prior_update_scaled = beta * prior_update_unscaled

            x = x + dc_update
            if i < num_steps - 1:
                x = x + prior_update_scaled + torch.sqrt(alpha) * randn_like(x)
            else:
                x = x + prior_update_scaled

            if save_scale_diagnostics:
                with torch.no_grad():
                    dc_norm = torch.linalg.vector_norm(
                        dc_update.reshape(dc_update.shape[0], -1), dim=1
                    ).mean()
                    prior_norm_unscaled = torch.linalg.vector_norm(
                        prior_update_unscaled.reshape(
                            prior_update_unscaled.shape[0], -1
                        ),
                        dim=1,
                    ).mean()
                    prior_norm_scaled = torch.linalg.vector_norm(
                        prior_update_scaled.reshape(
                            prior_update_scaled.shape[0], -1
                        ),
                        dim=1,
                    ).mean()
                    scale_rows.append(
                        {
                            "outer_step": i + 1,
                            "inner_loop": j + 1,
                            "sigma": float(t_cur.detach().cpu()),
                            "in_scaled_region": bool(in_scaled_region),
                            "beta": float(beta),
                            "sse": float(sse.detach().cpu()),
                            "dc_update_l2": float(dc_norm.cpu()),
                            "prior_update_l2_unscaled": float(
                                prior_norm_unscaled.cpu()
                            ),
                            "prior_update_l2_scaled": float(
                                prior_norm_scaled.cpu()
                            ),
                            "prior_to_dc_ratio_unscaled": float(
                                (prior_norm_unscaled / dc_norm.clamp_min(1e-20)).cpu()
                            ),
                            "prior_to_dc_ratio_scaled": float(
                                (prior_norm_scaled / dc_norm.clamp_min(1e-20)).cpu()
                            ),
                        }
                    )

            if verbose:
                print(
                    f"step {i + 1}/{num_steps}, inner {j + 1}/{inner_loops}, "
                    f"beta={beta:g}"
                )

        should_save = (
            save_intermediate
            and intermediate_dir is not None
            and clean is not None
            and (
                i == 0
                or ((i + 1) % intermediate_every == 0)
                or (i == num_steps - 1)
            )
        )
        if should_save:
            current_recon = x[:, :, pad:pad + w, pad:pad + w].detach().squeeze(1)
            (
                noisy_psnr,
                recon_psnr,
                noisy_ssim,
                recon_ssim,
                noisy_nrmse,
                recon_nrmse,
            ) = makeFigures(
                noisy2=x_init.squeeze(1),
                denoised2=current_recon,
                orig2=clean.squeeze(1),
                i=i + 1,
                out_dir=intermediate_dir,
                tag=f"{safe_tag}_step{i + 1:03d}",
                plot=True,
            )
            intermediate_rows.append(
                {
                    "step": i + 1,
                    "num_steps": num_steps,
                    "beta": beta,
                    "noisy_psnr": noisy_psnr,
                    "recon_psnr": recon_psnr,
                    "noisy_ssim": noisy_ssim,
                    "recon_ssim": recon_ssim,
                    "noisy_nrmse": noisy_nrmse,
                    "recon_nrmse": recon_nrmse,
                }
            )

    if intermediate_rows and intermediate_dir is not None:
        path = os.path.join(intermediate_dir, "intermediate_metrics.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(intermediate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(intermediate_rows)

    if scale_rows and scale_csv_path is not None:
        with open(scale_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(scale_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scale_rows)
        print(f"[Late Prior Scale] Saved diagnostics to {scale_csv_path}")

    return (
        x[:, :, pad:pad + w, pad:pad + w].detach().squeeze(1),
        noisypsnr,
        denoisedpsnr,
        noisyssim,
        denoisedssim,
        noisynrmse,
        denoisednrmse,
    )
