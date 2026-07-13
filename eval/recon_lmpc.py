"""
Late-Stage Multi-Partition Consensus (LMPC) reconstruction for PaDIS-MRI.

The original training code and original eval/recon.py are not modified.
During the final N outer diffusion steps, K complementary partitions evaluate
the same x and the same injected Gaussian noise. Their denoised whole-image
estimates are averaged before one data-consistency update and one diffusion
update are applied.
"""

import csv
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import tqdm

from denoise_padding_lmpc import denoisedFromPatches, get_indices_lmpc
from utils import makeFigures


Offset = Tuple[int, int]


def _ve_base_schedule(
    net,
    num_sigmas: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device,
) -> torch.Tensor:
    idx = torch.arange(num_sigmas, dtype=torch.float64, device=device)
    t = (
        sigma_max ** (1.0 / rho)
        + (idx / (num_sigmas - 1.0))
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t = net.round_sigma(t)
    return torch.cat(
        [t, torch.zeros(1, dtype=torch.float64, device=device)], dim=0
    )


def _complementary_offsets(
    pad: int,
    k: int,
    mode: str,
) -> List[Offset]:
    """
    Return K complementary offsets.

    random_complementary:
        Draw one random anchor, then add half-pad shifts modulo pad. This keeps
        the original random-partition character while covering complementary
        relative patch positions.

    fixed:
        Always use anchors based at (0,0), useful for deterministic debugging.
    """
    if k not in (1, 2, 4):
        raise ValueError(f"late_consensus_k must be 1, 2, or 4, got {k}")

    if pad <= 0:
        return [(0, 0)]

    if mode not in ("random_complementary", "fixed"):
        raise ValueError(
            "late_consensus_offset_mode must be "
            "'random_complementary' or 'fixed'"
        )

    if mode == "random_complementary":
        base_a = random.randint(0, pad - 1)
        base_b = random.randint(0, pad - 1)
    else:
        base_a, base_b = 0, 0

    half = pad // 2
    if k == 1:
        shifts = [(0, 0)]
    elif k == 2:
        shifts = [(0, 0), (half, half)]
    else:
        shifts = [(0, 0), (half, 0), (0, half), (half, half)]

    return [
        ((base_a + da) % pad, (base_b + db) % pad)
        for da, db in shifts
    ]


def dps2_lmpc(
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
    late_consensus_steps: int = 0,
    late_consensus_k: int = 1,
    late_consensus_offset_mode: str = "random_complementary",
    save_consensus_diagnostics: bool = False,
):
    """
    PaDIS-MRI reconstruction with optional late-stage multi-partition consensus.

    LMPC is active for the final ``late_consensus_steps`` outer diffusion
    levels. At each inner loop in that region:

      1. Keep the same current state x.
      2. Inject Gaussian noise once to obtain x_noisy.
      3. Denoise x_noisy using K complementary patch partitions.
      4. Average the K denoised whole-image estimates.
      5. Perform one measurement-consistency update.
      6. Perform one diffusion update.

    Setting late_consensus_steps=0 or late_consensus_k=1 preserves the original
    single-random-partition behavior.
    """
    if measurement is None:
        raise ValueError("dps2_lmpc requires an MRI measurement")

    num_steps = int(num_steps)
    inner_loops = int(inner_loops)
    late_consensus_steps = int(late_consensus_steps)
    late_consensus_k = int(late_consensus_k)

    if num_steps < 2:
        raise ValueError(f"num_steps must be >= 2, got {num_steps}")
    if inner_loops < 1:
        raise ValueError(f"inner_loops must be >= 1, got {inner_loops}")
    if not 0 <= late_consensus_steps <= num_steps:
        raise ValueError(
            f"late_consensus_steps must be in [0,{num_steps}], "
            f"got {late_consensus_steps}"
        )
    if late_consensus_k not in (1, 2, 4):
        raise ValueError(
            f"late_consensus_k must be 1, 2, or 4, got {late_consensus_k}"
        )
    if late_consensus_offset_mode not in ("random_complementary", "fixed"):
        raise ValueError(
            "late_consensus_offset_mode must be "
            "'random_complementary' or 'fixed'"
        )

    net.eval()
    w = latents.shape[-1]
    patches = w // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(
        x_init, (pad, pad, pad, pad), "constant", 0
    )

    t_steps = _ve_base_schedule(
        net, num_steps, sigma_min, sigma_max, rho, device
    )

    late_start = num_steps - late_consensus_steps
    lmpc_active = late_consensus_steps > 0 and late_consensus_k > 1

    print(
        "[LMPC] "
        f"active={lmpc_active} | "
        f"last_outer_steps={late_consensus_steps} | "
        f"k={late_consensus_k} | "
        f"offset_mode={late_consensus_offset_mode} | "
        f"first_LMPC_outer_step="
        f"{late_start + 1 if lmpc_active else 'disabled'}"
    )

    noisypsnr = denoisedpsnr = 0.0
    noisyssim = denoisedssim = 0.0
    noisynrmse = denoisednrmse = 0.0

    intermediate_rows = []
    intermediate_dir = None
    safe_tag = tag if tag is not None else "sample"
    intermediate_every = max(1, int(intermediate_every))

    if save_intermediate and save_dir is not None:
        intermediate_dir = os.path.join(
            save_dir, "intermediate_vis", safe_tag
        )
        os.makedirs(intermediate_dir, exist_ok=True)
        print(
            f"[intermediate] Enabled: {intermediate_dir} "
            f"| every={intermediate_every}"
        )

    diagnostic_rows = []
    diagnostic_path = None
    if save_consensus_diagnostics and save_dir is not None:
        diagnostic_dir = os.path.join(save_dir, "lmpc_diagnostics")
        os.makedirs(diagnostic_dir, exist_ok=True)
        diagnostic_path = os.path.join(
            diagnostic_dir, f"{safe_tag}_lmpc.csv"
        )

    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=len(t_steps) - 1,
    ):
        del t_next
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2
        use_consensus = lmpc_active and i >= late_start

        if use_consensus and i == late_start:
            print(
                f"[LMPC] Entering consensus region at outer step "
                f"{i + 1}/{num_steps}"
            )

        for j in range(inner_loops):
            x = x.detach().requires_grad_(True)

            # Offset generation occurs before torch noise generation, matching
            # the original logical order. In random-complementary mode it uses
            # exactly one random anchor per inner loop.
            if use_consensus:
                offsets = _complementary_offsets(
                    pad=pad,
                    k=late_consensus_k,
                    mode=late_consensus_offset_mode,
                )
            else:
                # None delegates to the original random-offset behavior.
                offsets = [None]

            # All partitions must see the same state and the same noise.
            x_noisy = x + t_cur * randn_like(x)
            x_real_noisy = torch.view_as_real(
                x_noisy.squeeze(1)
            ).permute(0, 3, 1, 2)

            d_sum = None
            detached_mag_sq_sum = None

            for offset in offsets:
                indices = get_indices_lmpc(
                    spaced=spaced,
                    patches=patches,
                    pad=pad,
                    psize=psize,
                    offset=offset,
                )

                d_part = denoisedFromPatches(
                    net,
                    x_real_noisy,
                    t_cur,
                    latents_pos,
                    None,
                    indices,
                    t_goal=0,
                    wrong=False,
                )

                d_sum = d_part if d_sum is None else d_sum + d_part

                if save_consensus_diagnostics and use_consensus:
                    d_det = d_part.detach()
                    if pad == 0:
                        d_det = d_det[:, :, :w, :w]
                    else:
                        d_det = d_det[
                            :, :, pad : pad + w, pad : pad + w
                        ]
                    mag_sq = torch.sum(d_det ** 2, dim=1)
                    detached_mag_sq_sum = (
                        mag_sq
                        if detached_mag_sq_sum is None
                        else detached_mag_sq_sum + mag_sq
                    )

            partition_count = len(offsets)
            d_real = d_sum / float(partition_count)

            if save_consensus_diagnostics and use_consensus:
                d_mean_det = d_real.detach()
                if pad == 0:
                    d_mean_det = d_mean_det[:, :, :w, :w]
                else:
                    d_mean_det = d_mean_det[
                        :, :, pad : pad + w, pad : pad + w
                    ]

                mean_mag_sq = torch.sum(d_mean_det ** 2, dim=1)
                variance_map = (
                    detached_mag_sq_sum / float(partition_count)
                    - mean_mag_sq
                ).clamp_min(0.0)

                denoiser_disagreement_rms = torch.sqrt(
                    variance_map.mean()
                )
                denoiser_mean_rms = torch.sqrt(mean_mag_sq.mean())
                sigma_value = float(t_cur.detach().cpu())
                score_disagreement_rms = (
                    denoiser_disagreement_rms
                    / max(sigma_value ** 2, 1e-20)
                )
                prior_update_disagreement_rms = (
                    float(alpha.detach().cpu()) / 2.0
                ) * score_disagreement_rms

                diagnostic_rows.append(
                    {
                        "outer_step": i + 1,
                        "inner_loop": j + 1,
                        "sigma": sigma_value,
                        "partition_count": partition_count,
                        "offsets": ";".join(
                            f"{a}:{b}" for a, b in offsets
                        ),
                        "denoiser_disagreement_rms": float(
                            denoiser_disagreement_rms.cpu()
                        ),
                        "denoiser_mean_rms": float(
                            denoiser_mean_rms.cpu()
                        ),
                        "relative_denoiser_disagreement": float(
                            (
                                denoiser_disagreement_rms
                                / denoiser_mean_rms.clamp_min(1e-20)
                            ).cpu()
                        ),
                        "score_disagreement_rms": float(
                            score_disagreement_rms.cpu()
                        ),
                        "prior_update_disagreement_rms": float(
                            prior_update_disagreement_rms.cpu()
                        ),
                    }
                )

            d_cplx = torch.complex(
                d_real[:, 0], d_real[:, 1]
            ).unsqueeze(1)
            score = (d_cplx - x_noisy) / (t_cur ** 2)

            cropped_x0hat = (
                d_real
                if pad == 0
                else d_real[:, :, pad : pad + w, pad : pad + w]
            )

            ax = inverseop.forward(cropped_x0hat)
            residual = measurement - ax
            residual_flat = residual.reshape(x.shape[0], -1)
            sse_ind = torch.norm(residual_flat, dim=-1) ** 2
            sse = torch.sum(sse_ind)

            likelihood_grad = torch.autograd.grad(
                outputs=sse, inputs=x
            )[0]

            x = x - (
                zeta / torch.sqrt(sse_ind)[:, None, None, None]
            ) * likelihood_grad

            if i < num_steps - 1:
                x = (
                    x
                    + (alpha / 2) * score
                    + torch.sqrt(alpha) * randn_like(x)
                )
            else:
                x = x + (alpha / 2) * score

            if verbose:
                with torch.no_grad():
                    print(
                        f"step {i + 1}/{num_steps}, "
                        f"inner {j + 1}/{inner_loops} "
                        f"-> ||x||={torch.linalg.norm(x).item():.4f}"
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
            current_recon = (
                x[:, :, pad : pad + w, pad : pad + w]
                .detach()
                .squeeze(1)
            )

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
                    "step": int(i + 1),
                    "num_steps": int(num_steps),
                    "noisy_psnr": float(noisy_psnr),
                    "recon_psnr": float(recon_psnr),
                    "noisy_ssim": float(noisy_ssim),
                    "recon_ssim": float(recon_ssim),
                    "noisy_nrmse": float(noisy_nrmse),
                    "recon_nrmse": float(recon_nrmse),
                }
            )

    if intermediate_rows and intermediate_dir is not None:
        csv_path = os.path.join(
            intermediate_dir, "intermediate_metrics.csv"
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(intermediate_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(intermediate_rows)
        print(f"[intermediate] Saved diagnostics to {intermediate_dir}")

    if diagnostic_rows and diagnostic_path is not None:
        with open(
            diagnostic_path, "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(diagnostic_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(diagnostic_rows)
        print(f"[LMPC] Saved partition diagnostics to {diagnostic_path}")

    return (
        x[:, :, pad : pad + w, pad : pad + w]
        .detach()
        .squeeze(1),
        noisypsnr,
        denoisedpsnr,
        noisyssim,
        denoisedssim,
        noisynrmse,
        denoisednrmse,
    )
