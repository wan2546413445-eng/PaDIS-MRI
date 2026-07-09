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
from denoise_padding_patchdiag import denoisedFromPatches, getIndices
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
    t = (sigma_max ** (1.0 / rho) + (idx / (num_sigmas - 1.0)) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    t = net.round_sigma(t)
    return torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)], dim=0)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, None)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def get_boundary_mask_padded(
    indices,
    padded_hw,
    pad: int,
    crop_w: int,
    boundary_width: int = 4,
    device="cuda",
) -> torch.Tensor:
    """
    Build patch-boundary mask in padded coordinates, then crop back to
    the original FOV [pad:pad+crop_w, pad:pad+crop_w].

    True  = patch boundary
    False = patch interior
    """
    H, W = int(padded_hw[0]), int(padded_hw[1])
    boundary_width = int(boundary_width)

    mask = torch.ones((H, W), dtype=torch.bool, device=device)

    for z in indices:
        r0, r1, c0, c1 = map(int, z)

        r0 = max(0, min(r0, H))
        r1 = max(0, min(r1, H))
        c0 = max(0, min(c0, W))
        c1 = max(0, min(c1, W))

        rr0 = r0 + boundary_width
        rr1 = r1 - boundary_width
        cc0 = c0 + boundary_width
        cc1 = c1 - boundary_width

        if rr0 < rr1 and cc0 < cc1:
            mask[rr0:rr1, cc0:cc1] = False

    return mask[pad:pad + crop_w, pad:pad + crop_w]


def compute_boundary_magnitude_mse(
    recon_complex: torch.Tensor,
    gt_complex: torch.Tensor,
    indices,
    padded_hw,
    pad: int,
    crop_w: int,
    boundary_width: int = 4,
    fg_threshold: float = 0.05,
) -> dict:
    """
    Magnitude-domain MSE on patch boundary and patch interior.

    recon_complex: [B,H,W] or [B,1,H,W], cropped FOV
    gt_complex:    [B,H,W] or [B,1,H,W], cropped FOV
    indices:       patch coordinates in padded image
    """
    if recon_complex.ndim == 4 and recon_complex.shape[1] == 1:
        recon_complex = recon_complex.squeeze(1)
    if gt_complex.ndim == 4 and gt_complex.shape[1] == 1:
        gt_complex = gt_complex.squeeze(1)

    gt_complex = gt_complex.to(recon_complex.device)

    recon_mag = torch.abs(recon_complex)
    gt_mag = torch.abs(gt_complex)
    err = (recon_mag - gt_mag) ** 2

    boundary_mask = get_boundary_mask_padded(
        indices=indices,
        padded_hw=padded_hw,
        pad=pad,
        crop_w=crop_w,
        boundary_width=boundary_width,
        device=err.device,
    )

    interior_mask = ~boundary_mask

    if fg_threshold is not None and fg_threshold > 0:
        gt_max = gt_mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        fg_mask = gt_mag > (float(fg_threshold) * gt_max)
    else:
        fg_mask = torch.ones_like(err, dtype=torch.bool)

    boundary_valid = boundary_mask.unsqueeze(0).expand_as(err) & fg_mask
    interior_valid = interior_mask.unsqueeze(0).expand_as(err) & fg_mask

    boundary_count = int(boundary_valid.sum().item())
    interior_count = int(interior_valid.sum().item())

    mse_boundary = float(err[boundary_valid].mean().item()) if boundary_count > 0 else float("nan")
    mse_interior = float(err[interior_valid].mean().item()) if interior_count > 0 else float("nan")
    ratio = mse_boundary / (mse_interior + 1e-12)

    return {
        "mse_boundary": mse_boundary,
        "mse_interior": mse_interior,
        "ratio": ratio,
        "boundary_count": boundary_count,
        "interior_count": interior_count,
    }


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
    boundary_stat: bool = False,
    boundary_every: int = 10,
    boundary_width: int = 4,
    boundary_fg_threshold: float = 0.05,
    boundary_csv: Optional[str] = None,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    """
    PaDIS patch-DPS reconstruction with optional patch-boundary diagnostics.

    This file is a diagnostics fork of recon.py. It keeps score computation and DPS updates unchanged.\n    The imported denoise_padding_patchdiag.py defaults to original hard stitching,\n    and can optionally enable PADIS_PATCH_AGG=edge_atten for ablation.

    Boundary stat can be enabled by function arguments or environment variables:
        PADIS_BOUNDARY_STAT=1
        PADIS_BOUNDARY_EVERY=10
        PADIS_BOUNDARY_WIDTH=4
        PADIS_BOUNDARY_FG_THRESHOLD=0.05
        PADIS_BOUNDARY_CSV=/path/to/file.csv
    """

    net.eval()
    w = latents.shape[-1]
    patches = w // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)

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
    # Boundary/interior diagnostic statistics.
    # ------------------------------------------------------------------
    boundary_stat = bool(boundary_stat) or _env_flag("PADIS_BOUNDARY_STAT", False)
    boundary_every = int(os.environ.get("PADIS_BOUNDARY_EVERY", boundary_every))
    boundary_width = int(os.environ.get("PADIS_BOUNDARY_WIDTH", boundary_width))
    boundary_fg_threshold = float(os.environ.get("PADIS_BOUNDARY_FG_THRESHOLD", boundary_fg_threshold))
    boundary_csv = os.environ.get("PADIS_BOUNDARY_CSV", boundary_csv)

    boundary_every = max(1, boundary_every)
    boundary_rows = []

    if boundary_stat and clean is None:
        print("[BoundaryStat] clean=None, disabled.")
        boundary_stat = False

    # ------------------------------------------------------------------
    # Main posterior sampling loop
    # ------------------------------------------------------------------
    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=len(t_steps) - 1,
    ):
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        for j in range(inner_loops):
            indices = getIndices(spaced, patches, pad, psize)
            x = x.detach().requires_grad_(True)

            # VE noise injection
            x_noisy = x + (t_cur * randn_like(x))
            x_real_noisy = torch.view_as_real(
                x_noisy.squeeze(1)
            ).permute(0, 3, 1, 2)

            # Patch denoise -> 2ch real-valued (Re, Im)
            D_real = denoisedFromPatches(
                net,
                x_real_noisy,
                t_cur,
                latents_pos,
                None,
                indices,
                t_goal=0,
                wrong=False,
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
                    print(f"step {i+1}/{num_steps} -> ||x||={torch.linalg.norm(x).item():.4f}")

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
                tag=f"{safe_tag}_step{i+1:03d}",
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
        # Patch-boundary diagnostic statistics.
        # Uses the final random-grid indices from the last inner loop.
        # ------------------------------------------------------------------
        if boundary_stat and (
            i == 0
            or ((i + 1) % boundary_every == 0)
            or (i == num_steps - 1)
        ):
            with torch.no_grad():
                current_recon = x[:, :, pad:pad + w, pad:pad + w].detach().squeeze(1)

                stat = compute_boundary_magnitude_mse(
                    recon_complex=current_recon,
                    gt_complex=clean,
                    indices=indices,
                    padded_hw=x.shape[-2:],
                    pad=pad,
                    crop_w=w,
                    boundary_width=boundary_width,
                    fg_threshold=boundary_fg_threshold,
                )

                row = {
                    "step": int(i + 1),
                    "num_steps": int(num_steps),
                    "boundary_width": int(boundary_width),
                    "fg_threshold": float(boundary_fg_threshold),
                    **stat,
                }
                boundary_rows.append(row)

                print(
                    f"[BoundaryStat] step={i+1:03d}/{num_steps} | "
                    f"MSE_boundary={stat['mse_boundary']:.6e} | "
                    f"MSE_interior={stat['mse_interior']:.6e} | "
                    f"ratio={stat['ratio']:.3f} | "
                    f"n_b={stat['boundary_count']} | "
                    f"n_i={stat['interior_count']}"
                )

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

    # ------------------------------------------------------------------
    # Save boundary statistic CSV
    # ------------------------------------------------------------------
    if boundary_rows:
        if boundary_csv is None:
            if save_dir is not None:
                boundary_csv = os.path.join(save_dir, f"{safe_tag}_boundary_stat.csv")

        if boundary_csv is not None:
            csv_dir = os.path.dirname(boundary_csv)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)

            with open(boundary_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(boundary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(boundary_rows)

            print(f"[BoundaryStat] Saved CSV: {boundary_csv}")

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

    #--- init ---
    shape = (batch_size, 1, resolution, resolution)
    x = sigma_max * randn_like(torch.zeros(shape, dtype=torch.complex64, device=device))
    if pad > 0:
        x = F.pad(x, (pad, pad, pad, pad), 'constant', 0)

    #--- schedule ---
    idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max**(1/rho) +
               idx/(num_steps-1)*(sigma_min**(1/rho)-sigma_max**(1/rho))
              )**rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros(1, dtype=torch.float64, device=device)])

    #--- positional grid (if your model needs it) ---
    R = resolution + 2*pad
    x_lin = torch.linspace(-1, 1, R, device=device)
    y_lin = torch.linspace(-1, 1, R, device=device)
    x_pos = x_lin.view(1, -1).repeat(R, 1)
    y_pos = y_lin.view(-1, 1).repeat(1, R)
    latents_pos = torch.stack([x_pos, y_pos], dim=0).unsqueeze(0)

    patches = (resolution // psize) + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    #--- VE-DPS loop (no measurement term) ---
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        print(i)
        alpha = 0.5 * t_cur**2
        for _ in range(4):
            x = x.detach().requires_grad_(True)

            eps = randn_like(x)
            x_noisy = x + t_cur * eps

            xr = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
            D_real = denoisedFromPatches(
                net,
                xr,
                t_cur,
                latents_pos,
                None,
                getIndices(spaced, patches, pad, psize),
                t_goal=0,
                wrong=False,
            )
            D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)

            score = (D_cplx - x) / (t_cur**2)

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
    pad: int = 0,
    device: str = 'cuda',
    randn_like=torch.randn_like,
    verbose: bool = False,
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    """
    EDM-style DPS reconstruction (whole-image denoising) with measurement consistency.
    Matches PaDIS-MRI noising schedule.
    """
    net.eval()

    x_init = inverseop.adjoint(measurement).to(device)
    x = x_init.clone()

    t_base = _ve_base_schedule(net, num_steps, sigma_min, sigma_max, rho, device)
    t_rep = torch.repeat_interleave(t_base[:-1], repeats_per_sigma)
    t_steps = torch.cat([t_rep, t_base[-1:]], dim=0)

    noisypsnr = denoisedpsnr = noisyssim = denoisedssim = noisynrmse = denoisednrmse = 0.0
    total_iters = t_steps.numel() - 1

    for i in tqdm.tqdm(range(total_iters)):
        t_cur = t_steps[i].float()
        alpha = 0.5 * (t_cur**2)

        x = x.detach().requires_grad_(True)

        x_noisy = x + t_cur * randn_like(x)

        x_real = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
        den_real = net(x_real, t_cur)

        den_cplx = torch.complex(den_real[:, 0], den_real[:, 1]).unsqueeze(1)
        score = (den_cplx - x_noisy) / (t_cur**2)

        Ax = inverseop.forward(den_real)
        residual = measurement - Ax
        sse = torch.sum(torch.abs(residual).view(residual.shape[0], -1)**2, dim=1)
        grad_l = torch.autograd.grad(outputs=sse, inputs=x)[0]

        x_mid = x + (alpha / 2) * score - (zeta / torch.sqrt(sse)[:, None, None, None]) * grad_l

        if i < total_iters - 1:
            x = x_mid + torch.sqrt(alpha) * randn_like(x_mid)
        else:
            x = x_mid

        if verbose:
            with torch.no_grad():
                print(f"step {i+1}/{num_steps} -> ||x||={torch.linalg.norm(x).item():.4f}")

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

    shape = (batch_size, net.img_channels, resolution, resolution)
    x = sigma_max * randn_like(torch.empty(shape, device=device))

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1/rho)
        + (step_indices / (num_steps - 1)) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        sigma = t_cur.float()
        dt = (t_next - t_cur).float()

        denoised = net(x, sigma)
        score = (x - denoised) / sigma
        x = x + score * dt

    x = x.squeeze(0)
    real = x[0]
    imag = x[1]
    complex = torch.complex(real, imag)
    return complex
