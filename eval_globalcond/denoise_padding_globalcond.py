import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import scipy.io
from diffusers import AutoencoderKL
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
torch.manual_seed(2)


def getIndices(spaced, patches, pad, psize, freezeindex=False):
    a, b = 0, 0
    if pad > 0:
        a = random.randint(0, pad - 1)
        b = random.randint(0, pad - 1)
    if freezeindex:
        a = 0
        b = 0

    indices = []
    for p in range(patches):
        for q in range(patches):
            indices.append([spaced[p] + a, spaced[p] + a + psize,
                            spaced[q] + b, spaced[q] + b + psize])
    return indices


def build_global_context_from_noisy(
    x_real_noisy,
    sigma,
    global_context_size=96,
    sigma_data=0.5,
    eps=1e-8,
):
    """
    Match training GC condition:
        context = upsample(downsample(magnitude(c_in * x_t_full), S), full_size)

    x_real_noisy: [B, 2, H, W], real/imag noisy full padded image.
    sigma: scalar or tensor.
    """
    sigma = torch.as_tensor(sigma, device=x_real_noisy.device, dtype=x_real_noisy.dtype).reshape(1, 1, 1, 1)
    c_in = 1.0 / torch.sqrt(torch.as_tensor(sigma_data, device=x_real_noisy.device, dtype=x_real_noisy.dtype) ** 2 + sigma ** 2)

    source = c_in * x_real_noisy
    context = torch.sqrt(torch.sum(source ** 2, dim=1, keepdim=True) + eps)

    h, w = context.shape[-2:]
    s = min(int(global_context_size), h, w)

    if s < h or s < w:
        low = F.interpolate(context, size=(s, s), mode="area")
        context = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)

    mean = context.mean(dim=(1, 2, 3), keepdim=True)
    std = context.std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    context = (context - mean) / std
    return context


def denoisedFromPatches_globalcond(
    net,
    x,
    t_hat,
    latents_pos,
    class_labels,
    indices,
    pad=96,
    t_goal=-1,
    avg=1,
    spaced=[],
    wrong=False,
    global_context_size=96,
):
    """
    GC-PaDIS patch denoiser.

    Original PaDIS condition:
        x_pos = [pos_x, pos_y]

    GC-PaDIS condition:
        x_pos = [pos_x, pos_y, lowres_global_context]
    """
    if len(spaced) > 1:
        indices = getIndices(spaced, 5, 24, 56)

    if wrong:
        x_hat = x
    else:
        x_hat = torch.clone(x)

    channels = len(x_hat[0, :, 0, 0])
    N = len(x_hat[0, 0, 0, :])
    psize = indices[0][1] - indices[0][0]
    patches = len(indices)
    pad = int((N - np.sqrt(patches) * psize))

    output = torch.zeros_like(x_hat)

    x_input = torch.zeros(
        patches, channels, psize, psize,
        device=x_hat.device,
        dtype=x_hat.dtype,
    )

    # 2 position channels + 1 magnitude low-res global context channel.
    pos_input = torch.zeros(
        patches, 3, psize, psize,
        device=x_hat.device,
        dtype=x_hat.dtype,
    )

    context_full = build_global_context_from_noisy(
        x_real_noisy=x_hat,
        sigma=t_hat,
        global_context_size=global_context_size,
    )

    for i in range(patches):
        z = indices[i]
        x_input[i, :, :, :] = torch.squeeze(x_hat[0, :, z[0]:z[1], z[2]:z[3]])
        pos_input[i, 0:2, :, :] = torch.squeeze(latents_pos[:, :, z[0]:z[1], z[2]:z[3]])
        pos_input[i, 2:3, :, :] = torch.squeeze(context_full[:, :, z[0]:z[1], z[2]:z[3]], dim=0)

    bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)

    for i in range(patches):
        z = indices[i]
        x_patch = x_hat[0, :, z[0]:z[1], z[2]:z[3]]
        output[0, :, z[0]:z[1], z[2]:z[3]] += bigout[i, :, :, :]
        output[0, :, z[0]:z[1], z[2]:z[3]] -= x_patch

    x_hat = x_hat + output

    temp = t_goal + torch.randn_like(x_hat) * t_goal
    temp[:, :, pad:N - pad, pad:N - pad] = x_hat[:, :, pad:N - pad, pad:N - pad]
    return temp