import os
import sys
import numpy as np
import torch
import tqdm
from typing import Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from denoise_padding_context_center import denoisedFromPatches, getIndices


def _ve_base_schedule(
    net,
    num_sigmas: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    idx = torch.arange(num_sigmas, dtype=torch.float64, device=device)
    t = (
        sigma_max ** (1.0 / rho)
        + (idx / (num_sigmas - 1.0))
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t = net.round_sigma(t)
    return torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)], dim=0)


def dps2_context_center(
    net,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: Optional[torch.Tensor],
    num_steps: int = 78,
    inner_loops: int = 10,
    sigma_min: float = 0.003,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    zeta: float = 3.0,
    pad: int = 64,
    psize: int = 64,
    context_margin: int = 16,
    chunk: int = 8,
    randn_like=torch.randn_like,
    verbose: bool = False,
    clean: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:

    net.eval()
    device = torch.device(device)

    w = latents.shape[-1]
    patches = w // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)

    t_steps = _ve_base_schedule(
        net=net,
        num_sigmas=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )

    noisypsnr = denoisedpsnr = noisyssim = denoisedssim = noisynrmse = denoisednrmse = 0.0

    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=len(t_steps) - 1,
    ):
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        for j in range(inner_loops):
            indices = getIndices(
                spaced=spaced,
                patches=patches,
                pad=pad,
                psize=psize,
                context_margin=context_margin,
            )

            x = x.detach().requires_grad_(True)

            x_noisy = x + (t_cur * randn_like(x))
            x_real_noisy = torch.view_as_real(
                x_noisy.squeeze(1)
            ).permute(0, 3, 1, 2)

            D_real = denoisedFromPatches(
                net=net,
                x=x_real_noisy,
                t_hat=t_cur,
                latents_pos=latents_pos,
                class_labels=None,
                indices=indices,
                pad=pad,
                t_goal=0,
                wrong=False,
                context_margin=context_margin,
                chunk=chunk,
            )

            D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)
            score = (D_cplx - x_noisy) / (t_cur ** 2)

            cropped_x0hat = D_real if pad == 0 else D_real[:, :, pad:pad+w, pad:pad+w]

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
                    print(
                        f"step {i+1}/{num_steps}, inner {j+1}/{inner_loops}, "
                        f"||x||={torch.linalg.norm(x).item():.4f}"
                    )

    return (
        x[:, :, pad:pad+w, pad:pad+w].detach().squeeze(1),
        noisypsnr,
        denoisedpsnr,
        noisyssim,
        denoisedssim,
        noisynrmse,
        denoisednrmse,
    )
