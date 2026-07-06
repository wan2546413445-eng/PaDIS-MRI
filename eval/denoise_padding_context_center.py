# ---------------------------------------------------------------
# Context-center inference helper for PaDIS-MRI.
#
# target patch: 64x64
# input patch : (64 + 2 * context_margin) x (64 + 2 * context_margin)
# output      : use only center 64x64 region
#
# This file does not replace eval/denoise_padding.py.
# ---------------------------------------------------------------

import random
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

torch.manual_seed(2)


def getIndices(spaced, patches, pad, psize, freezeindex=False, context_margin=0):
    """
    Generate target patch indices [y0, y1, x0, x1].

    For context-center inference, the target patch must have enough margin
    around it, so random offset is restricted to [margin, pad-margin-1].
    """
    margin = int(context_margin)

    a, b = 0, 0
    if pad > 0:
        low = margin
        high = pad - margin - 1
        if high >= low:
            a = random.randint(low, high)
            b = random.randint(low, high)
        else:
            a = max(0, min(margin, pad - 1))
            b = max(0, min(margin, pad - 1))

    if freezeindex:
        a = margin if pad > margin else 0
        b = margin if pad > margin else 0

    indices = []
    for p in range(patches):
        for q in range(patches):
            indices.append([
                spaced[p] + a,
                spaced[p] + a + psize,
                spaced[q] + b,
                spaced[q] + b + psize,
            ])
    return indices


def denoisedFromPatches(
    net,
    x,
    t_hat,
    latents_pos,
    class_labels,
    indices,
    pad=64,
    t_goal=-1,
    avg=1,
    spaced=[],
    wrong=False,
    context_margin=0,
    chunk=8,
):
    """
    Context-center denoising.

    x: [1, 2, Hpad, Wpad]
    latents_pos: [1, 2, Hpad, Wpad]

    For each target 64x64 patch:
        input to net: 96x96 if context_margin=16
        write back: only center 64x64
    """

    if len(spaced) > 1:
        indices = getIndices(
            spaced,
            5,
            24,
            56,
            context_margin=context_margin,
        )

    x_hat = x if wrong else torch.clone(x)

    margin = int(context_margin)
    channels = x_hat.shape[1]
    N = x_hat.shape[-1]

    target_psize = indices[0][1] - indices[0][0]
    input_psize = target_psize + 2 * margin
    patches = len(indices)

    crop_pad = int((N - np.sqrt(patches) * target_psize))

    output = torch.zeros_like(x_hat)

    x_input = torch.zeros(
        patches,
        channels,
        input_psize,
        input_psize,
        dtype=x_hat.dtype,
        device=x_hat.device,
    )
    pos_input = torch.zeros(
        patches,
        2,
        input_psize,
        input_psize,
        dtype=latents_pos.dtype,
        device=latents_pos.device,
    )

    for i in range(patches):
        y0, y1, x0, x1 = indices[i]

        hy0 = y0 - margin
        hy1 = y1 + margin
        hx0 = x0 - margin
        hx1 = x1 + margin

        if hy0 < 0 or hx0 < 0 or hy1 > N or hx1 > N:
            raise ValueError(
                f"Context-center patch out of bounds: "
                f"target={[y0, y1, x0, x1]}, "
                f"context_margin={margin}, full_size={N}"
            )

        x_input[i, :, :, :] = x_hat[0, :, hy0:hy1, hx0:hx1]
        pos_input[i, :, :, :] = latents_pos[0, :, hy0:hy1, hx0:hx1]

    # Important: context patch is larger than ordinary PaDIS patch.
    # Use chunked forward to reduce peak memory.
    outs = []
    chunk = max(1, int(chunk))

    # Checkpointed chunk forward:
    # We still need gradient wrt x through the denoiser for DPS,
    # but do not want to store all U-Net activations for 49 context patches.
    for s in range(0, patches, chunk):
        e = min(s + chunk, patches)

        x_chunk = x_input[s:e]
        pos_chunk = pos_input[s:e]

        def _net_forward(xc, pc):
            return net(xc, t_hat, pc, class_labels)

        out = torch_checkpoint(
            _net_forward,
            x_chunk,
            pos_chunk,
            use_reentrant=False,
        )
        outs.append(out)

    bigout = torch.cat(outs, dim=0).to(torch.float64)

    for i in range(patches):
        y0, y1, x0, x1 = indices[i]
        x_patch = x_hat[0, :, y0:y1, x0:x1]

        if margin > 0:
            if bigout.shape[-1] == input_psize:
                pred_center = bigout[
                    i,
                    :,
                    margin:margin + target_psize,
                    margin:margin + target_psize,
                ]
            elif bigout.shape[-1] == target_psize:
                pred_center = bigout[i]
            else:
                raise RuntimeError(
                    f"Unexpected net output size {bigout.shape[-2:]}; "
                    f"expected {(input_psize, input_psize)} or "
                    f"{(target_psize, target_psize)}."
                )
        else:
            pred_center = bigout[i]

        output[0, :, y0:y1, x0:x1] += pred_center
        output[0, :, y0:y1, x0:x1] -= x_patch

    x_hat = x_hat + output

    temp = t_goal + torch.randn_like(x_hat) * t_goal
    temp[:, :, crop_pad:N - crop_pad, crop_pad:N - crop_pad] = \
        x_hat[:, :, crop_pad:N - crop_pad, crop_pad:N - crop_pad]

    return temp
