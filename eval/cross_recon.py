import numpy as np
import torch
import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os
import sys
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dnnlib.util import configure_bart

configure_bart()

from bart import bart

from inverse_operators import *
from cross_denoise_padding import denoisedFromCrossPatchSets, cross_getIndices as getIndices
from utils import fftmod, makeFigures


def _ve_base_schedule(
        net,
        num_sigmas: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        device: torch.device,
) -> torch.Tensor:
    """
    Build a base VE schedule of length `num_sigmas` (ex. 104), and appends a terminal zero (so length becomes num_sigmas+1).
    """
    idx = torch.arange(num_sigmas, dtype=torch.float64, device=device)
    t = (sigma_max ** (1.0 / rho) + (idx / (num_sigmas - 1.0)) * (
                sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    t = net.round_sigma(t)
    return torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)], dim=0)


def cross_dps2(
        net,
        latents: torch.Tensor,
        latents_pos: torch.Tensor,
        inverseop,
        measurement: Optional[torch.Tensor],
        num_steps: int = 104,  # num noising steps
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
        device: str = 'cuda',
        save_dir: Optional[str] = None,
        tag: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_every: int = 10,
        cp_k: int = 8,
        cp_local_k: int = 3,
        cp_global_k: int = 4,
        cp_context_mode: str = 'global_random',
        cp_overlap_ratio: float = 0.5,

        cp_eval_batch_size: int = 2,
        cp_debug: bool = False,
        memory_safe: bool = True,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    """
    PaDIS (patch DPS) MRI reconstruction with 10 inner sub-steps per sigma
    (total ~1040 updates for num_steps=104).

    Optional diagnostic mode:
    - save_intermediate=True:
        Save author-style intermediate visualizations and metrics
        after selected outer diffusion steps.
    - intermediate_every:
        Save every N outer steps, while always saving step 1 and final step.
    """

    net.eval()
    w = latents.shape[-1]
    patches = w // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)  # complex

    t_steps = _ve_base_schedule(net, num_steps, sigma_min, sigma_max, rho, device)

    noisypsnr = denoisedpsnr = noisyssim = denoisedssim = noisynrmse = denoisednrmse = 0.0

    # ------------------------------------------------------------------
    # Intermediate visualization / author-style metric diagnostics
    # ------------------------------------------------------------------
    intermediate_rows = []
    intermediate_dir = None
    safe_tag = tag if tag is not None else "sample"
    intermediate_every = max(1, int(intermediate_every))

    if save_intermediate and save_dir is not None:
        intermediate_dir = os.path.join(save_dir, "intermediate_vis", safe_tag)
        os.makedirs(intermediate_dir, exist_ok=True)
        print(f"[intermediate] Enabled: {intermediate_dir} | every={intermediate_every}")

    # ------------------------------------------------------------------
    # Main posterior sampling loop
    # ------------------------------------------------------------------
    for i, (t_cur, t_next) in tqdm.tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])),
            total=len(t_steps) - 1
    ):
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        for j in range(inner_loops):
            indices = getIndices(spaced, patches, pad, psize)
            x = x.detach().requires_grad_(True)

            # VE noise injection
            x_noisy = x + (t_cur * randn_like(x))

            if memory_safe:
                with torch.no_grad():
                    x_real_noisy = torch.view_as_real(
                        x_noisy.detach().squeeze(1)
                    ).permute(0, 3, 1, 2)  # (B,2,H_pad,W_pad)
                    D_real = denoisedFromCrossPatchSets(
                        net,
                        x_real_noisy,
                        t_cur,
                        latents_pos,
                        None,
                        indices,
                        cp_k=cp_k, cp_local_k=cp_local_k, cp_global_k=cp_global_k,
                        cp_context_mode=cp_context_mode, cp_overlap_ratio=cp_overlap_ratio,
                        cp_eval_batch_size=cp_eval_batch_size,
                        cp_debug=cp_debug
                    )
                D_real = D_real.detach()
                D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)
                score = (D_cplx - x_noisy.detach()) / (t_cur ** 2)

                cropped_x = x[:, :, pad:pad + w, pad:pad + w]
                cropped_x_real = torch.view_as_real(cropped_x.squeeze(1)).permute(0, 3, 1, 2)
                Ax = inverseop.forward(cropped_x_real)
                residual = measurement - Ax
                residual_flat = residual.reshape(x.shape[0], -1)
                sse_ind = torch.norm(residual_flat, dim=-1) ** 2
                sse = torch.sum(sse_ind)
                likelihood_grad = torch.autograd.grad(outputs=sse, inputs=x)[0]
            else:
                x_real_noisy = torch.view_as_real(
                    x_noisy.squeeze(1)
                ).permute(0, 3, 1, 2)  # (B,2,H_pad,W_pad)

                # Patch denoise -> 2ch real-valued (Re, Im)
                D_real = denoisedFromCrossPatchSets(
                    net,
                    x_real_noisy,
                    t_cur,
                    latents_pos,
                    None,
                    indices,
                    cp_k=cp_k, cp_local_k=cp_local_k, cp_global_k=cp_global_k, cp_context_mode=cp_context_mode, cp_overlap_ratio=cp_overlap_ratio, cp_eval_batch_size=cp_eval_batch_size,
                    cp_debug=cp_debug
                )

                # Score function calculation
                D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)
                score = (D_cplx - x_noisy) / (t_cur ** 2)
                cropped_x0hat = D_real if pad == 0 else D_real[:, :, pad:pad + w, pad:pad + w]

                # Forward + residual
                Ax = inverseop.forward(cropped_x0hat)
                residual = measurement - Ax
                residual_flat = residual.reshape(x.shape[0], -1)
                sse_ind = torch.norm(residual_flat, dim=-1) ** 2
                sse = torch.sum(sse_ind)

                # Gradient of SSE w.r.t. x
                likelihood_grad = torch.autograd.grad(outputs=sse, inputs=x)[0]

            # Measurement-consistency update
            x = x - (zeta / torch.sqrt(sse_ind)[:, None, None, None]) * likelihood_grad

            # Diffusion step
            if i < num_steps - 1:
                x = x + (alpha / 2) * score + torch.sqrt(alpha) * randn_like(x)
            else:
                x = x + (alpha / 2) * score

            if verbose:
                with torch.no_grad():
                    print(f"step {i + 1}/{num_steps} -> ||x||={torch.linalg.norm(x).item():.4f}")

        # ------------------------------------------------------------------
        # Save intermediate visualization after the full inner loop of this
        # outer diffusion step.
        # ------------------------------------------------------------------
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

            intermediate_rows.append({
                "step": int(i + 1),
                "num_steps": int(num_steps),
                "noisy_psnr": float(noisy_psnr),
                "recon_psnr": float(recon_psnr),
                "noisy_ssim": float(noisy_ssim),
                "recon_ssim": float(recon_ssim),
                "noisy_nrmse": float(noisy_nrmse),
                "recon_nrmse": float(recon_nrmse),
            })

    # ------------------------------------------------------------------
    # Save intermediate metric CSV
    # ------------------------------------------------------------------
    if intermediate_rows and intermediate_dir is not None:
        csv_path = os.path.join(intermediate_dir, "intermediate_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(intermediate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(intermediate_rows)

        print(f"[intermediate] Saved diagnostics to {intermediate_dir}")

    return (
        x[:, :, pad:pad + w, pad:pad + w].detach().squeeze(1),
        noisypsnr,
        denoisedpsnr,
        noisyssim,
        denoisedssim,
        noisynrmse,
        denoisednrmse,
    )


@torch.no_grad()
def dps_uncond(
        net,
        batch_size=1,
        resolution=384,
        psize=96,
        pad=96,
        num_steps=50,
        sigma_min=0.003,
        sigma_max=10.0,
        rho=7,
        device='cuda',
        randn_like=torch.randn_like,
        cp_k: int = 8,
        cp_local_k: int = 3,
        cp_global_k: int = 4,
        cp_context_mode: str = 'global_random',
        cp_overlap_ratio: float = 0.5,
        cp_eval_batch_size: int = 2,

):
    net.eval()

    # --- init ---
    shape = (batch_size, 1, resolution, resolution)
    x = sigma_max * randn_like(torch.zeros(shape, dtype=torch.complex64, device=device))
    if pad > 0:
        x = F.pad(x, (pad, pad, pad, pad), 'constant', 0)

    # --- schedule ---大到小noise非线性递减
    idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) +
               idx / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
               ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros(1, dtype=torch.float64, device=device)])

    # --- positional grid (if your model needs it) ---
    R = resolution + 2 * pad
    x_lin = torch.linspace(-1, 1, R, device=device)
    y_lin = torch.linspace(-1, 1, R, device=device)
    x_pos = x_lin.view(1, -1).repeat(R, 1)
    y_pos = y_lin.view(-1, 1).repeat(1, R)
    latents_pos = torch.stack([x_pos, y_pos], dim=0).unsqueeze(0)  # [1,2,R,R]

    patches = (resolution // psize) + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    # --- VE-DPS loop (no measurement term) ---
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        print(i)
        alpha = 0.5 * t_cur ** 2
        for _ in range(4):  # same inner loops as dps2
            x = x.detach().requires_grad_(True)

            # 1) VE noise injection
            eps = randn_like(x)
            x_noisy = x + t_cur * eps

            # 2) denoise patches
            xr = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
            D_real = denoisedFromCrossPatchSets(net, xr, t_cur, latents_pos, None,
                                                getIndices(spaced, patches, pad, psize),
                                                cp_k=cp_k, cp_local_k=cp_local_k, cp_global_k=cp_global_k,
                                                cp_context_mode=cp_context_mode, cp_overlap_ratio=cp_overlap_ratio,
                                                cp_eval_batch_size=cp_eval_batch_size)

            D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)

            # 3) score
            score = (D_cplx - x) / (t_cur ** 2)

            # 4) diffusion update
            if i < num_steps - 1:
                x = x + (alpha / 2) * score + torch.sqrt(alpha) * randn_like(x)
            else:
                x = x + (alpha / 2) * score

    if pad > 0:
        x = x[:, :, pad:-pad, pad:-pad]
    return x.detach()


def dps_edm(
        net,
        measurement: torch.Tensor,
        clean: torch.Tensor,
        inverseop,
        num_steps: int = 104,
        repeats_per_sigma: int = 10,
        sigma_min: float = 0.003,
        sigma_max: float = 10.0,
        rho: float = 7.0,
        zeta: float = 3.0,
        pad: int = 0,  # use SAME pad as PaDIS to align FOV
        device: str = 'cuda',
        randn_like=torch.randn_like,
        verbose: bool = False,
        save_dir: Optional[str] = None,
        tag: Optional[str] = None,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    """
    EDM-style DPS reconstruction (whole-image denoising) with measurement consistency. Matches PaDIS-MRI noising schedule.
    """
    net.eval()

    x_init = inverseop.adjoint(measurement).to(device)
    x = x_init.clone()

    t_base = _ve_base_schedule(net, num_steps, sigma_min, sigma_max, rho, device)
    t_rep = torch.repeat_interleave(t_base[:-1], repeats_per_sigma)
    t_steps = torch.cat([t_rep, t_base[-1:]], dim=0)

    noisypsnr = denoisedpsnr = noisyssim = denoisedssim = noisynrmse = denoisednrmse = 0.0
    total_iters = t_steps.numel() - 1

    # 3) main Euler‐Maruyama loop
    for i in tqdm.tqdm(range(total_iters)):
        t_cur = t_steps[i].float()
        alpha = 0.5 * (t_cur ** 2)

        x = x.detach().requires_grad_(True)

        # VE noise injection
        x_noisy = x + t_cur * randn_like(x)

        # model call in 2ch real
        x_real = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
        den_real = net(x_real, t_cur)  # [B,2,H,W]

        # score calc
        den_cplx = torch.complex(den_real[:, 0], den_real[:, 1]).unsqueeze(1)
        score = (den_cplx - x_noisy) / (t_cur ** 2)

        # measurement‐consistency
        Ax = inverseop.forward(den_real)
        residual = measurement - Ax
        sse = torch.sum(torch.abs(residual).view(residual.shape[0], -1) ** 2, dim=1)  # [B]
        # gradient of SSE w.r.t. x_cplx
        grad_l = torch.autograd.grad(outputs=sse, inputs=x)[0]

        # measurement‐consistency update
        x_mid = x + (alpha / 2) * score - (zeta / torch.sqrt(sse)[:, None, None, None]) * grad_l

        # diffusion update
        if i < total_iters - 1:
            x = x_mid + torch.sqrt(alpha) * randn_like(x_mid)
        else:
            x = x_mid

        if verbose:
            with torch.no_grad():
                print(f"step {i + 1}/{num_steps} -> ||x||={torch.linalg.norm(x).item():.4f}")

        # if save_dir is not None and i == num_steps-1:
        #     noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse = makeFigures(
        #                                                                                         x_init.squeeze(1),
        #                                                                                         x.squeeze(1).detach(),
        #                                                                                         clean.squeeze(1),
        #                                                                                         i,
        #                                                                                         save_dir,
        #                                                                                         tag)

    return x.detach(), noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse


@torch.no_grad()
def dps_uncond_edm(
        net,
        batch_size=1,
        resolution=384,
        num_steps=96,
        sigma_min=0.003,
        sigma_max=10.0,
        rho=7,
        device='cuda',
        randn_like=torch.randn_like,
):
    """
    Unconditional generation using a plain EDM model trained on full images (no patching).
    """

    # 1) initialize noise
    shape = (batch_size, net.img_channels, resolution, resolution)
    x = sigma_max * randn_like(torch.empty(shape, device=device))

    # 2) build noise schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
                      sigma_max ** (1 / rho)
                      + (step_indices / (num_steps - 1)) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
              ) ** rho
    # append final zero
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    # 3) Euler–Maruyama loop
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        sigma = t_cur.float()
        dt = (t_next - t_cur).float()

        # score estimate via denoising network
        denoised = net(x, sigma)  # [B,C,H,W]
        score = (x - denoised) / sigma  # [B,C,H,W]

        # Euler update
        x = x + score * dt

    x = x.squeeze(0)
    real = x[0]
    imag = x[1]
    complex = torch.complex(real, imag)
    return complex
