# ---------------------------------------------------------------
# Context-center inference helper for PaDIS-MRI.
#
# New file:
#   eval/denoise_padding_context_center.py
#
# This file does not replace eval/denoise_padding.py.
# ---------------------------------------------------------------

import random
import numpy as np
import torch

torch.manual_seed(2)


def getIndices(spaced, patches, pad, psize, freezeindex=False, context_margin=0):
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
    pad=96,
    t_goal=-1,
    avg=1,
    spaced=[],
    wrong=False,
    context_margin=0,
):
    if len(spaced) > 1:
        indices = getIndices(spaced, 5, 24, 56, context_margin=context_margin)

    x_hat = x if wrong else torch.clone(x)

    margin = int(context_margin)
    channels = len(x_hat[0, :, 0, 0])
    N = len(x_hat[0, 0, 0, :])

    target_psize = indices[0][1] - indices[0][0]
    input_psize = target_psize + 2 * margin
    patches = len(indices)

    crop_pad = int((N - np.sqrt(patches) * target_psize))

    output = torch.zeros_like(x_hat)
    x_input = torch.zeros(patches, channels, input_psize, input_psize, device=x_hat.device)
    pos_input = torch.zeros(patches, 2, input_psize, input_psize, device=x_hat.device)

    for i in range(patches):
        y0, y1, x0, x1 = indices[i]

        hy0 = y0 - margin
        hy1 = y1 + margin
        hx0 = x0 - margin
        hx1 = x1 + margin

        if hy0 < 0 or hx0 < 0 or hy1 > N or hx1 > N:
            raise ValueError(
                f"Context-center patch out of bounds: "
                f"target={[y0, y1, x0, x1]}, context_margin={margin}, full_size={N}"
            )

        x_input[i, :, :, :] = torch.squeeze(x_hat[0, :, hy0:hy1, hx0:hx1])
        pos_input[i, :, :, :] = torch.squeeze(latents_pos[:, :, hy0:hy1, hx0:hx1])

    bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)

    for i in range(patches):
        y0, y1, x0, x1 = indices[i]
        x_patch = x_hat[0, :, y0:y1, x0:x1]

        if margin > 0:
            pred_center = bigout[i, :, margin:margin + target_psize, margin:margin + target_psize]
        else:
            pred_center = bigout[i, :, :, :]

        output[0, :, y0:y1, x0:x1] += pred_center
        output[0, :, y0:y1, x0:x1] -= x_patch

    x_hat = x_hat + output

    temp = t_goal + torch.randn_like(x_hat) * t_goal
    temp[:, :, crop_pad:N - crop_pad, crop_pad:N - crop_pad] = x_hat[:, :, crop_pad:N - crop_pad, crop_pad:N - crop_pad]
    return temp
