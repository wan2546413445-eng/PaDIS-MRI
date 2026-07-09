import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import PIL.Image
import scipy.io
from diffusers import AutoencoderKL
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import sys

# Diagnostics fork of denoise_padding.py.
# Default behavior is identical to the original hard residual stitching.
# Set PADIS_PATCH_AGG=edge_atten to enable edge residual attenuation.
torch.manual_seed(2)
#随机生成 offset a,b，spaced：grid,psize:patch size
#每次 inner loop 都随机改变 patch 网格的起始位置，避免 patch 边界永远固定在同一批像素上
def getIndices(spaced, patches, pad, psize, freezeindex = False):
    a, b = 0, 0  # Default values when pad = 0;patch 网格整体往下移动 a 个像素，往右移动 b 个像素
    if pad > 0:
        a = random.randint(0, pad-1)#a∈{0,1,…,63}
        b = random.randint(0, pad-1)#b∈{0,1,…,63}
    if freezeindex:
        a = 0
        b = 0
    indices = []
    for p in range(patches):#p=0,1,2,3,4,5,6
        for q in range(patches):#q=0,1,2,3,4,5,6
            indices.append([
                spaced[p]+a,
                spaced[p]+a+psize,
                spaced[q]+b,
                spaced[q]+b+psize])
    #[row_start, row_end, col_start, col_end]
    return indices

def denoisedFromPatches(net, x, t_hat, latents_pos, class_labels, indices, pad=96, t_goal = -1, avg=1, spaced=[], wrong=False):
    if len(spaced) > 1:
        indices = getIndices(spaced, 5, 24, 56)
    if wrong:
        x_hat = x
    else:
        x_hat = torch.clone(x)
        
        
    channels = len(x_hat[0,:,0,0])#2[real,imag]
    N = len(x_hat[0,0,0,:])#512
    psize = indices[0][1] - indices[0][0]
    #假设indices[0] = [10, 74, 20, 84]，就是74-10=10
    patches = len(indices)#总 patch 数；如7*7=49
    pad = int((N - np.sqrt(patches)*psize))
    #pad=512−7×64=512−448=64.
    output = torch.zeros_like(x_hat)
    x_input = torch.zeros(patches, channels, psize, psize).to(torch.device('cuda'))
    #[49, 2, 64, 64]
    pos_input = torch.zeros(patches, 2, psize, psize).to(torch.device('cuda'))
    #[49, 2, 64, 64]
    for i in range(patches):#i=0,1,…,48
        z = indices[i]#取第 i 个 patch 的坐标，如z = [10, 74, 20, 84]
        x_input[i,:,:,:] = torch.squeeze(x_hat[0,:,z[0]:z[1], z[2]:z[3]])
        #裁 image patch，x_hat[0, :, 10:74, 20:84]；[2, 64, 64]
        pos_input[i,:,:,:] = torch.squeeze(latents_pos[:,:,z[0]:z[1], z[2]:z[3]])
        #裁 position patch，latents_pos[:, :, 10:74, 20:84]；[2, 64, 64]
    #输入x_input.shape = [49, 2, 64, 64]；pos_input.shape = [49, 2, 64, 64]
    # Eval-only memory saving for heavy patch networks such as MSRUNet.
    # This keeps pad=64 and psize=64 unchanged.
    # It only splits the 49 patch forward into smaller chunks.
    patch_batch = int(os.environ.get("PADIS_PATCH_BATCH", patches))
    use_ckpt = os.environ.get("PADIS_PATCH_CHECKPOINT", "0") == "1"

    if patch_batch >= patches and not use_ckpt:
        bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)
    else:
        bigout_list = []
        for start in range(0, patches, patch_batch):
            end = min(start + patch_batch, patches)
            xi = x_input[start:end]
            pi = pos_input[start:end]
            ci = None if class_labels is None else class_labels[start:end]

            if use_ckpt:
                def forward_chunk(xi_, pi_):
                    return net(xi_, t_hat, pi_, ci)
                out = checkpoint(forward_chunk, xi, pi, use_reentrant=False)
            else:
                out = net(xi, t_hat, pi, ci)

            bigout_list.append(out.to(torch.float64))

        bigout = torch.cat(bigout_list, dim=0)
    #输出bigout.shape = [49, 2, 64, 64]，net就是denoised patch estimate

    # ------------------------------------------------------------------
    # Patch residual aggregation.
    #
    # Default mode keeps the original PaDIS-MRI hard residual stitching:
    #     x_hat[z_i] <- x_hat[z_i] + (D_i - x_i)
    #
    # Optional diagnostic ablation:
    #     PADIS_PATCH_AGG=edge_atten
    #     PADIS_EDGE_SIGMA=0.6
    #     PADIS_EDGE_FLOOR=0.5
    #
    # This attenuates only patch-edge residual updates. It does not change
    # the random-grid sampling or the network forward pass.
    # ------------------------------------------------------------------
    agg_mode = os.environ.get("PADIS_PATCH_AGG", "hard").strip().lower()

    if agg_mode in ("edge_atten", "edge_attenuation"):
        sigma_w = float(os.environ.get("PADIS_EDGE_SIGMA", "0.6"))
        edge_floor = float(os.environ.get("PADIS_EDGE_FLOOR", "0.5"))
        edge_floor = max(0.0, min(edge_floor, 1.0))

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, psize, device=x_hat.device, dtype=x_hat.dtype),
            torch.linspace(-1, 1, psize, device=x_hat.device, dtype=x_hat.dtype),
            indexing="ij",
        )

        window2d = torch.exp(-(xx**2 + yy**2) / (2 * sigma_w**2))
        window2d = window2d / window2d.max().clamp_min(1e-12)
        window2d = edge_floor + (1.0 - edge_floor) * window2d
        window = window2d.unsqueeze(0).repeat(channels, 1, 1)

        for i in range(patches):
            z = indices[i]
            x_patch = x_hat[0, :, z[0]:z[1], z[2]:z[3]]
            patch_residual = bigout[i, :, :, :].to(x_hat.dtype) - x_patch
            output[0, :, z[0]:z[1], z[2]:z[3]] += patch_residual * window
    else:
        for i in range(patches):
            z = indices[i]  # 取第 i 个 patch 的空间坐标: [row_start, row_end, col_start, col_end]
            # 取出去噪前的 noisy patch，shape 为 [2, P, P]，两个通道分别为 real/imag
            x_patch = x_hat[0, :, z[0]:z[1], z[2]:z[3]]
            # 将第 i 个 denoised patch 写回 output 的对应空间位置
            output[0, :, z[0]:z[1], z[2]:z[3]] += bigout[i, :, :, :]
            # 计算该 patch 的去噪残差:
            # output[z_i] = denoised_patch - noisy_patch
            output[0, :, z[0]:z[1], z[2]:z[3]] -= x_patch

    # 将 patch residual 加回原图:
    # x_hat[z_i] = noisy_patch + (denoised_patch - noisy_patch) = denoised_patch
    # 因此在当前 non-overlapping partition 内，相当于把每个 noisy patch 替换为 denoised patch
    x_hat = x_hat + output
    # 构造返回图像。Algorithm 1 调用时 t_goal=0，因此 temp 初始为全 0。
    # 随后只把中心原始 FOV 区域替换为 denoised result，padding 外围区域保持为 0。
    temp = t_goal + torch.randn_like(x_hat) * t_goal
    temp[:, :, pad:N - pad, pad:N - pad] = x_hat[:, :, pad:N - pad, pad:N - pad]
    return temp
