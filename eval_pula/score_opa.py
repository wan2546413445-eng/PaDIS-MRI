"""OPA patch denoising on the current FOV state."""

import random

import numpy as np
import torch
import torch.nn.functional as F


def real_channels(image):
    return torch.view_as_real(image.squeeze(1)).permute(0, 3, 1, 2).contiguous()


def patch_shifts(mode, steps, pad, seed):
    # Use PaDIS's frozen grid first in both modes and keep an independent grid RNG.
    if mode == "fixed":
        return [(0, 0)] * steps
    rng = random.Random(seed)
    return [(0, 0)] + [divmod(n, pad) for n in rng.sample(range(1, pad * pad), steps - 1)]


class PatchScore:
    def __init__(self, net, image_size, pad, psize, device):
        from denoise_padding import denoisedFromPatches, getIndices

        self.net = net
        self.pad = pad
        self.image_size = image_size
        self.denoise_patches = denoisedFromPatches
        patches = image_size // psize + 1
        self.indices = getIndices(np.arange(patches) * psize, patches, 0, psize)
        coords = torch.linspace(-1, 1, image_size + 2 * pad, device=device)
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        self.positions = torch.stack((x, y)).unsqueeze(0)

    def denoise_padded(self, padded_state, sigma, shift):
        """Patch denoise a padded complex state; gradients are preserved for DPS."""
        row, col = shift
        indices = [(a + row, b + row, c + col, d + col) for a, b, c, d in self.indices]
        query = real_channels(padded_state)
        sigma_arg = query.new_tensor(sigma)
        denoised = self.denoise_patches(
            self.net, query, sigma_arg, self.positions, None, indices, t_goal=0, wrong=False,
        )
        return torch.complex(denoised[:, 0], denoised[:, 1]).unsqueeze(1)

    @torch.no_grad()
    def denoise(self, state, sigma, shift):
        padded = F.pad(state, (self.pad,) * 4)
        denoised = self.denoise_padded(padded, sigma, shift)
        fov = slice(self.pad, self.pad + self.image_size)
        return denoised[:, :, fov, fov]


def _rms(value):
    return value.abs().square().mean().sqrt().item()


def compare_grids(state, sigma, shifts, predictions):
    # Each partition covers the entire FOV. Pairwise RMS is on their common FOV.
    rows = [{"shift": shift, "denoised_rms": _rms(pred),
             "score_rms": _rms((pred - state) / sigma ** 2)}
            for shift, pred in zip(shifts, predictions)]
    pairs = []
    for i, left in enumerate(predictions):
        for j in range(i + 1, len(predictions)):
            difference = _rms(left - predictions[j])
            pairs.append({"grids": [i, j], "overlap_denoised_rms": difference,
                          "overlap_score_rms": difference / sigma ** 2})
    return {"sigma": sigma, "grids": rows, "pairs": pairs}
