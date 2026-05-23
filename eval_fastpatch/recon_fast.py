import numpy as np
import torch
import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os
import sys
import csv
import time
from contextlib import nullcontext

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dnnlib.util import configure_bart

configure_bart()

from bart import bart

from eval.inverse_operators import *
from denoise_padding_fast import denoisedFromPatches, getIndices
from eval.utils import fftmod, makeFigures


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

def dps2(
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
        device: str = 'cuda',
        save_dir: Optional[str] = None,
        tag: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_every: int = 10,
        posterior_mode: str = "original",
        cheap_dc_mode: str = "adjoint",
        phase_full_every: Tuple[int, int, int] = (8, 4, 2),
        final_full_steps: int = 5,
        stagnation_window: int = 3,
        stagnation_ratio: float = 0.98,
        cheap_dc_scale: float = 2.0,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:

    net.eval()

    if posterior_mode not in ("original", "anchored"):
        raise ValueError(f"Unsupported posterior_mode: {posterior_mode}")

    if cheap_dc_mode != "adjoint":
        raise ValueError(f"Unsupported cheap_dc_mode: {cheap_dc_mode}")

    if len(phase_full_every) != 3:
        raise ValueError("phase_full_every must contain exactly three integers.")

    def _cuda_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _full_every_for_step(step_idx: int) -> int:
        if step_idx < 0.25 * num_steps:
            return max(1, int(phase_full_every[0]))
        if step_idx < 0.75 * num_steps:
            return max(1, int(phase_full_every[1]))
        return max(1, int(phase_full_every[2]))

    def _should_force_full_from_history(history) -> bool:
        if stagnation_window <= 0:
            return False
        if len(history) < stagnation_window + 1:
            return False
        prev = history[-(stagnation_window + 1):-1]
        if len(prev) == 0:
            return False
        best_prev = min(prev)
        return history[-1] > stagnation_ratio * best_prev

    w = latents.shape[-1]
    patches = w // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)

    t_steps = _ve_base_schedule(net, num_steps, sigma_min, sigma_max, rho, device)

    noisypsnr = denoisedpsnr = noisyssim = denoisedssim = noisynrmse = denoisednrmse = 0.0

    intermediate_rows = []
    intermediate_dir = None
    safe_tag = tag if tag is not None else "sample"
    intermediate_every = max(1, int(intermediate_every))

    if save_intermediate and save_dir is not None:
        intermediate_dir = os.path.join(save_dir, "intermediate_vis", safe_tag)
        os.makedirs(intermediate_dir, exist_ok=True)
        print(f"[intermediate] Enabled: {intermediate_dir} | every={intermediate_every}")

    total_updates = 0
    full_updates = 0
    cheap_updates = 0
    dc_err_history = []
    dc_err_sum = 0.0
    likelihood_backward_runtime = 0.0
    cheap_adjoint_runtime = 0.0
    run_start = time.perf_counter()

    measurement_norm = (
        torch.norm(measurement.reshape(measurement.shape[0], -1), dim=-1)
        + 1e-12
    )

    for i, (t_cur, t_next) in tqdm.tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])),
            total=len(t_steps) - 1
    ):
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        for j in range(inner_loops):
            if posterior_mode == "original":
                use_full = True
            else:
                use_full = False
                full_every = _full_every_for_step(i)

                if (j % full_every) == 0:
                    use_full = True

                if i >= num_steps - int(final_full_steps):
                    use_full = True

                if _should_force_full_from_history(dc_err_history):
                    use_full = True

            indices = getIndices(spaced, patches, pad, psize)

            if use_full:
                x = x.detach().requires_grad_(True)
                grad_context = nullcontext()
            else:
                x = x.detach()
                grad_context = torch.no_grad()

            with grad_context:
                x_noisy = x + (t_cur * randn_like(x))
                x_real_noisy = torch.view_as_real(
                    x_noisy.squeeze(1)
                ).permute(0, 3, 1, 2).contiguous()

                D_real = denoisedFromPatches(
                    net,
                    x_real_noisy,
                    t_cur,
                    latents_pos,
                    None,
                    indices,
                    t_goal=0,
                    wrong=False
                )

                D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)
                score = (D_cplx - x_noisy) / (t_cur ** 2)

                cropped_x0hat = (
                    D_real
                    if pad == 0
                    else D_real[:, :, pad:pad + w, pad:pad + w]
                )

                Ax = inverseop.forward(cropped_x0hat)
                residual = measurement - Ax
                residual_flat = residual.reshape(x.shape[0], -1)
                sse_ind = torch.norm(residual_flat, dim=-1) ** 2
                sse = torch.sum(sse_ind)

            dc_err = torch.sqrt(sse_ind.detach())
            dc_err_norm = (dc_err / measurement_norm).mean().item()
            dc_err_history.append(dc_err_norm)
            dc_err_sum += dc_err_norm

            if use_full:
                _cuda_sync()
                t0 = time.perf_counter()
                likelihood_grad = torch.autograd.grad(outputs=sse, inputs=x)[0]
                _cuda_sync()
                likelihood_backward_runtime += time.perf_counter() - t0
                full_updates += 1
            else:
                with torch.no_grad():
                    residual_for_grad = Ax - measurement

                    _cuda_sync()
                    t0 = time.perf_counter()
                    grad_crop = cheap_dc_scale * inverseop.adjoint(residual_for_grad)
                    grad_padded = torch.zeros_like(x)

                    if pad == 0:
                        grad_padded = grad_crop
                    else:
                        grad_padded[:, :, pad:pad + w, pad:pad + w] = grad_crop

                    likelihood_grad = grad_padded
                    _cuda_sync()
                    cheap_adjoint_runtime += time.perf_counter() - t0

                cheap_updates += 1

            total_updates += 1

            update_context = nullcontext() if use_full else torch.no_grad()
            with update_context:
                scale = zeta / (torch.sqrt(sse_ind.detach())[:, None, None, None] + 1e-12)
                x = x - scale * likelihood_grad

                if i < num_steps - 1:
                    x = x + (alpha / 2) * score + torch.sqrt(alpha) * randn_like(x)
                else:
                    x = x + (alpha / 2) * score

            if verbose:
                with torch.no_grad():
                    print(f"step {i + 1}/{num_steps} -> ||x||={torch.linalg.norm(x).item():.4f}")

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

    if intermediate_rows and intermediate_dir is not None:
        csv_path = os.path.join(intermediate_dir, "intermediate_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(intermediate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(intermediate_rows)

        print(f"[intermediate] Saved diagnostics to {intermediate_dir}")

    _cuda_sync()
    total_runtime = time.perf_counter() - run_start
    mean_dc_err = dc_err_sum / max(total_updates, 1)
    final_dc_err = dc_err_history[-1] if dc_err_history else float("nan")
    full_ratio = full_updates / max(total_updates, 1)

    print(
        f"[PaDIS dps2] posterior_mode={posterior_mode} "
        f"total_updates={total_updates} full_updates={full_updates} "
        f"cheap_updates={cheap_updates} full_ratio={full_ratio:.4f} "
        f"mean_dc_err={mean_dc_err:.6e} final_dc_err={final_dc_err:.6e} "
        f"total_runtime={total_runtime:.3f}s "
        f"likelihood_backward_runtime={likelihood_backward_runtime:.3f}s "
        f"cheap_adjoint_runtime={cheap_adjoint_runtime:.3f}s"
    )

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
            D_real = denoisedFromPatches(net, xr, t_cur, latents_pos, None,
                                         getIndices(spaced, patches, pad, psize),
                                         t_goal=0, wrong=False)
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
