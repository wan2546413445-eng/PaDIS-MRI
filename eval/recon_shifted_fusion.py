
import numpy as np
import torch
import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os
import sys
import csv
from torch.utils.checkpoint import checkpoint
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dnnlib.util import configure_bart
configure_bart()

from bart import bart

from inverse_operators import *
from recon import _ve_base_schedule
from denoise_padding import getIndices, denoisedFromPatches
from utils import fftmod, makeFigures


def dps2_shifted_fusion(
    net,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: Optional[torch.Tensor],
    num_steps: int = 78,  # num noising steps
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
    fusion_start_update: int = 520,
    shift: int = 32,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    """
    PaDIS (patch DPS) MRI reconstruction with shifted uniform fusion after a configurable update
    (total 780 updates for num_steps=78 and inner_loops=10).

    Optional diagnostic mode:
    - save_intermediate=True:
        Save author-style intermediate visualizations and metrics
        after selected outer diffusion steps.
    - intermediate_every:
        Save every N outer steps, while always saving step 1 and final step.
    """

    net.eval()
    w = latents.shape[-1]#w = 原始图像宽度 = N
    patches = w // psize + 1#每个方向上切多少个 patch 起点
    #如patches=384//64+1=6+1=7.
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)
    #spaced = np.linspace(0, 384, 7, dtype=int)
    #spaced = [0, 64, 128, 192, 256, 320, 384]；没有随机 offset 时的 patch 起点列表

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

    uniform_weight = torch.ones(
        1,
        psize,
        psize,
        dtype=x_init.real.dtype,
        device=x_init.device,
    )

    def _indices_from_offset(offset_row: int, offset_col: int):
        return [
            [spaced[p] + offset_row, spaced[p] + offset_row + psize, spaced[q] + offset_col, spaced[q] + offset_col + psize]
            for p in range(patches)
            for q in range(patches)
        ]

    def _net_forward(x_input, sigma, pos_input):
        return net(x_input, sigma, pos_input, None)

    def _denoise_partition(x_real_noisy, indices):
        channels = x_real_noisy.shape[1]
        patch_count = len(indices)
        x_input = torch.zeros(
            patch_count, channels, psize, psize,
            dtype=x_real_noisy.dtype, device=x_real_noisy.device
        )
        pos_input = torch.zeros(
            patch_count, 2, psize, psize,
            dtype=latents_pos.dtype, device=latents_pos.device
        )
        for patch_idx, z in enumerate(indices):
            x_input[patch_idx] = x_real_noisy[0, :, z[0]:z[1], z[2]:z[3]]
            pos_input[patch_idx] = latents_pos[0, :, z[0]:z[1], z[2]:z[3]]
        return checkpoint(
            _net_forward,
            x_input,
            t_cur,
            pos_input,
            use_reentrant=False,
        ).to(x_real_noisy.dtype)

    # ------------------------------------------------------------------
    # Main posterior sampling loop
    # ------------------------------------------------------------------
    for i, (t_cur, t_next) in tqdm.tqdm(
        #同时获取t_cur噪声强度，t_next下一个强度
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=len(t_steps) - 1
    ):
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2
        for j in range(inner_loops):
            #inner_loops = 10；每个噪声层内部重复更新次数
            #每个 inner loop 会随机采样一次 patch offset，因此同一个噪声层下可以看到不同的 patch 划分。
            indices = getIndices(spaced, patches, pad, psize)
            #spaced = [0, 64, 128, 192, 256, 320, 384]；patches = 7；pad = 64；psize = 64
            #每次 inner loop 都随机选一个偏移 (a,b)，然后根据这个偏移裁剪整张图的 patch。
            x = x.detach().requires_grad_(True)
            # VE noise injection
            #Score-based SDE 的反向采样不是一步预测 clean image，而是在不同噪声水平下逐步根据 score 修正图像。
            x_noisy = x + (t_cur * randn_like(x))
            x_real_noisy = torch.view_as_real(
                x_noisy.squeeze(1)
            ).permute(0, 3, 1, 2)
            # [B, 1, H, W]--[B, H, W]--[B, H, W, 2]--(B,2,H_pad,W_pad)
            # Patch denoise -> 2ch real-valued (Re, Im)
            global_update = i * inner_loops + j
            if global_update < fusion_start_update:
                D_real = denoisedFromPatches(#调用算法2
                    net,
                    x_real_noisy,
                    t_cur,
                    latents_pos,
                    None,
                    indices,
                    t_goal=0,
                    wrong=False
                )
            else:
                indices_1 = indices
                offset_row_1 = indices_1[0][0]
                offset_col_1 = indices_1[0][2]
                offset_row_2 = (offset_row_1 + shift) % pad if pad > 0 else 0
                offset_col_2 = (offset_col_1 + shift) % pad if pad > 0 else 0
                indices_2 = _indices_from_offset(offset_row_2, offset_col_2)

                denoised_1 = _denoise_partition(x_real_noisy, indices_1)
                denoised_2 = _denoise_partition(x_real_noisy, indices_2)

                numerator = torch.zeros_like(x_real_noisy)
                weight_sum = torch.zeros(
                    x_real_noisy.shape[0], 1, x_real_noisy.shape[2], x_real_noisy.shape[3],
                    dtype=x_real_noisy.dtype, device=x_real_noisy.device
                )
                weight = uniform_weight
                for denoised_patches, partition_indices in ((denoised_1, indices_1), (denoised_2, indices_2)):
                    for patch_idx, z in enumerate(partition_indices):
                        numerator[0, :, z[0]:z[1], z[2]:z[3]] += denoised_patches[patch_idx] * weight
                        weight_sum[0, :, z[0]:z[1], z[2]:z[3]] += weight

                fused = numerator / (weight_sum + 1e-12)
                D_real = torch.zeros_like(fused)
                if pad == 0:
                    D_real = fused
                else:
                    h_pad, w_pad = fused.shape[-2:]
                    D_real[:, :, pad:h_pad-pad, pad:w_pad-pad] = fused[:, :, pad:h_pad-pad, pad:w_pad-pad]

            # Score function calculation
            D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)#real/imag 转回 complex
            score = (D_cplx - x_noisy) / (t_cur ** 2)
            #D_real裁掉padding；取长度为 w =N的中心区域
            cropped_x0hat = D_real if pad == 0 else D_real[:, :, pad:pad+w, pad:pad+w]#clean image estimate

            # Forward + residual
            Ax = inverseop.forward(cropped_x0hat)#A=PFS
            residual = measurement - Ax#当前与测量值的残差
            residual_flat = residual.reshape(x.shape[0], -1)#把r拉平
            sse_ind = torch.norm(residual_flat, dim=-1) ** 2#每个 batch 单独算平方范数
            sse = torch.sum(sse_ind)#batch 内所有样本求和；sum of squared errors

            # Gradient of SSE w.r.t. x；要让 SSE 变小，当前变量 x 应该往哪个方向调整。
            likelihood_grad = torch.autograd.grad(outputs=sse, inputs=x)[0]

            # Measurement-consistency update
            # zeta：data consistency 强度超参数，默认设置为3.0
            # sqrt(sse_ind)：对 batch 中每个样本单独归一化。
            x = x - (zeta / torch.sqrt(sse_ind)[:, None, None, None]) * likelihood_grad

            # Diffusion step：当前图像如何更符合 learned MRI image prior
            if i < num_steps - 1:#如果还没到最后一个噪声层，就继续加随机噪声
                x = x + (alpha / 2) * score + torch.sqrt(alpha) * randn_like(x)
            else:#如果已经是最后一步，就不再加噪声
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
            current_recon = x[:, :, pad:pad+w, pad:pad+w].detach().squeeze(1)

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
        x[:, :, pad:pad+w, pad:pad+w].detach().squeeze(1),
        noisypsnr,
        denoisedpsnr,
        noisyssim,
        denoisedssim,
        noisynrmse,
        denoisednrmse,
    )
