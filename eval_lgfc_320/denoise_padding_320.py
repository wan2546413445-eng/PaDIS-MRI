"""Patch denoising for the 320-ROI / 384-measurement LGFC experiment."""

import os
import random

import torch
from torch.utils.checkpoint import checkpoint


TARGET_SIZE = 320
MEASUREMENT_SIZE = 384
CANVAS_SIZE = 448

# Geometry inside the 448 canvas.
OUTER_ZERO = 32
MEAS_START = OUTER_ZERO
MEAS_END = MEAS_START + MEASUREMENT_SIZE       # 416
TARGET_START = 64
TARGET_END = TARGET_START + TARGET_SIZE        # 384


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


def denoised_from_patches_320(
    net,
    x_noisy_real,
    x_state_real,
    t_hat,
    latents_pos,
    class_labels,
    indices,
):
    """Return separate outputs for MRI x0 and diffusion score.

    Geometry in the 448x448 canvas:

      [0:32] / [416:448]
          Original zero-padding context retained from the 512 PaDIS canvas.

      [32:416]
          Original complete 384 MRI FOV.

      [64:384]
          Central 320 target that receives the learned PaDIS denoiser.

    Two outputs are deliberately separated:

    1) x0_for_dc
       - outer zero padding: 0
       - 32-pixel real MRI context ring: current posterior state x
       - central 320 target: learned denoiser output D_theta(x_noisy)

       This is used by the unchanged 384 MRI likelihood and LGFC reference.

    2) denoised_for_score
       - outer zero padding: 0, preserving the original PaDIS zero-padding score
       - 32-pixel real MRI context ring: x_noisy, hence score == 0 there
       - central 320 target: learned denoiser output D_theta(x_noisy)

       Therefore the learned diffusion prior acts only on the central 320 ROI.
    """
    for name, value in (
        ("x_noisy_real", x_noisy_real),
        ("x_state_real", x_state_real),
    ):
        if value.ndim != 4 or value.shape[-2:] != (CANVAS_SIZE, CANVAS_SIZE):
            raise ValueError(
                f"{name} must end in {CANVAS_SIZE}x{CANVAS_SIZE}, "
                f"got {tuple(value.shape)}"
            )

    if latents_pos.shape[-2:] != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(
            f"latents_pos must end in {CANVAS_SIZE}x{CANVAS_SIZE}, "
            f"got {tuple(latents_pos.shape)}"
        )

    x_hat = torch.clone(x_noisy_real)
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
        bigout = net(
            x_input,
            t_hat,
            pos_input,
            class_labels,
        ).to(torch.float64)
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

                out = checkpoint(
                    forward_chunk,
                    xi,
                    pi,
                    use_reentrant=False,
                )
            else:
                out = net(xi, t_hat, pi, ci)

            bigout_list.append(out.to(torch.float64))

        bigout = torch.cat(bigout_list, dim=0)

    # Same residual/scatter semantics as the original non-overlapping PaDIS
    # partition. Only the central 320 region is consumed as learned x0.
    residual_update = torch.zeros_like(x_hat)
    for i, (h0, h1, w0, w1) in enumerate(indices):
        residual_update[0, :, h0:h1, w0:w1] += (
            bigout[i] - x_hat[0, :, h0:h1, w0:w1]
        )
    denoised_full = x_hat + residual_update

    # ------------------------------------------------------------
    # x0 used by the original 384 MRI data-consistency likelihood:
    #
    # measurement context ring -> current posterior state
    # target 320              -> learned denoiser output
    # outer zero padding      -> zero
    # ------------------------------------------------------------
    x0_for_dc = torch.zeros_like(x_hat)

    x0_for_dc[
        :,
        :,
        MEAS_START:MEAS_END,
        MEAS_START:MEAS_END,
    ] = x_state_real[
        :,
        :,
        MEAS_START:MEAS_END,
        MEAS_START:MEAS_END,
    ]

    x0_for_dc[
        :,
        :,
        TARGET_START:TARGET_END,
        TARGET_START:TARGET_END,
    ] = denoised_full[
        :,
        :,
        TARGET_START:TARGET_END,
        TARGET_START:TARGET_END,
    ]

    # ------------------------------------------------------------
    # D used only to form diffusion score:
    #
    # context ring -> x_noisy  => score exactly zero
    # target 320   -> D_theta  => normal learned PaDIS score
    # outer pad    -> zero     => preserve original zero-padding behavior
    # ------------------------------------------------------------
    denoised_for_score = torch.zeros_like(x_hat)

    denoised_for_score[
        :,
        :,
        MEAS_START:MEAS_END,
        MEAS_START:MEAS_END,
    ] = x_noisy_real[
        :,
        :,
        MEAS_START:MEAS_END,
        MEAS_START:MEAS_END,
    ]

    denoised_for_score[
        :,
        :,
        TARGET_START:TARGET_END,
        TARGET_START:TARGET_END,
    ] = denoised_full[
        :,
        :,
        TARGET_START:TARGET_END,
        TARGET_START:TARGET_END,
    ]

    return x0_for_dc, denoised_for_score
