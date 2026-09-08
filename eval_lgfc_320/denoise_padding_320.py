"""Patch denoising for the 320-ROI / 384-measurement LGFC experiment."""

import os
import random
import torch
from torch.utils.checkpoint import checkpoint

TARGET_SIZE = 320
MEASUREMENT_SIZE = 384
CANVAS_SIZE = 448
OUTER_ZERO = 32
TARGET_START = 64
TARGET_END = TARGET_START + TARGET_SIZE
MEAS_START = OUTER_ZERO
MEAS_END = MEAS_START + MEASUREMENT_SIZE


def get_indices(spaced, patches, pad, psize, freezeindex=False):
    """Mirror eval/denoise_padding.py RNG/index semantics."""
    a = b = 0
    if pad > 0:
        a = random.randint(0, pad - 1)
        b = random.randint(0, pad - 1)
    if freezeindex:
        a = b = 0

    indices = []
    for p in range(patches):
        for q in range(patches):
            indices.append([
                int(spaced[p] + a),
                int(spaced[p] + a + psize),
                int(spaced[q] + b),
                int(spaced[q] + b + psize),
            ])
    return indices


def denoised_from_patches_320(net, x, t_hat, latents_pos, class_labels, indices):
    """Build a 448x448 denoiser output for a central 320 target.

    Regions:
      - [0:32] and [416:448] outer padding: zero.
      - original 384 measurement FOV [32:416]: identity by default.
      - central 320 target [64:384]: PaDIS denoised output.

    The identity context keeps the original 384 MRI likelihood well-defined
    and differentiable, while the learned PaDIS prior is applied only to the
    central 320 target.
    """
    if x.ndim != 4 or x.shape[-2:] != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(
            f"x must end in {CANVAS_SIZE}x{CANVAS_SIZE}, got {tuple(x.shape)}"
        )
    if latents_pos.shape[-2:] != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(
            f"latents_pos must end in {CANVAS_SIZE}x{CANVAS_SIZE}, "
            f"got {tuple(latents_pos.shape)}"
        )

    x_hat = torch.clone(x)
    channels = x_hat.shape[1]
    psize = indices[0][1] - indices[0][0]
    num_patches = len(indices)

    x_input = torch.zeros(
        (num_patches, channels, psize, psize),
        dtype=x_hat.dtype,
        device=x_hat.device,
    )
    pos_input = torch.zeros(
        (num_patches, 2, psize, psize),
        dtype=latents_pos.dtype,
        device=latents_pos.device,
    )

    for i, (h0, h1, w0, w1) in enumerate(indices):
        x_input[i] = x_hat[0, :, h0:h1, w0:w1]
        pos_input[i] = latents_pos[0, :, h0:h1, w0:w1]

    patch_batch = int(os.environ.get("PADIS_PATCH_BATCH", num_patches))
    use_ckpt = os.environ.get("PADIS_PATCH_CHECKPOINT", "0") == "1"

    if patch_batch >= num_patches and not use_ckpt:
        bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)
    else:
        bigout_list = []
        for start in range(0, num_patches, patch_batch):
            end = min(start + patch_batch, num_patches)
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

    residual_update = torch.zeros_like(x_hat)
    for i, (h0, h1, w0, w1) in enumerate(indices):
        residual_update[0, :, h0:h1, w0:w1] += (
            bigout[i] - x_hat[0, :, h0:h1, w0:w1]
        )
    denoised = x_hat + residual_update

    output = torch.zeros_like(x_hat)
    output[:, :, MEAS_START:MEAS_END, MEAS_START:MEAS_END] = (
        x[:, :, MEAS_START:MEAS_END, MEAS_START:MEAS_END]
    )
    output[:, :, TARGET_START:TARGET_END, TARGET_START:TARGET_END] = (
        denoised[:, :, TARGET_START:TARGET_END, TARGET_START:TARGET_END]
    )
    return output
