
import numpy as np
import torch
import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os
import sys
import csv
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(REPO_DIR, "eval")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, EVAL_DIR)


from dnnlib.util import configure_bart
configure_bart()

from bart import bart


from inverse_operators import *
from utils import makeFigures
from recon import _ve_base_schedule
from denoise_padding_globalcond import denoisedFromPatches_globalcond, getIndices


def dps2_globalcond(
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
    global_context_size: int = 96,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:

    net.eval()
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

    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=len(t_steps) - 1,
    ):
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        for j in range(inner_loops):
            indices = getIndices(spaced, patches, pad, psize)
            x = x.detach().requires_grad_(True)

            x_noisy = x + (t_cur * randn_like(x))
            x_real_noisy = torch.view_as_real(
                x_noisy.squeeze(1)
            ).permute(0, 3, 1, 2)

            D_real = denoisedFromPatches_globalcond(
                net,
                x_real_noisy,
                t_cur,
                latents_pos,
                None,
                indices,
                t_goal=0,
                wrong=False,
                global_context_size=global_context_size,
            )

            D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)
            score = (D_cplx - x_noisy) / (t_cur ** 2)

            cropped_x0hat = D_real if pad == 0 else D_real[:, :, pad:pad + w, pad:pad + w]

            Ax = inverseop.forward(cropped_x0hat)
            residual = measurement - Ax
            residual_flat = residual.reshape(x.shape[0], -1)
            sse_ind = torch.norm(residual_flat, dim=-1) ** 2
            sse = torch.sum(sse_ind)

            likelihood_grad = torch.autograd.grad(outputs=sse, inputs=x)[0]
            x = x - (zeta / torch.sqrt(sse_ind)[:, None, None, None]) * likelihood_grad

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

    return (
        x[:, :, pad:pad + w, pad:pad + w].detach().squeeze(1),
        noisypsnr,
        denoisedpsnr,
        noisyssim,
        denoisedssim,
        noisynrmse,
        denoisednrmse,
    )