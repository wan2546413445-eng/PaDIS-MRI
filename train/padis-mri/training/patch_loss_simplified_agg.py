"""Single-patch EDM loss with simplified anatomy-gradient guided sampling.

This loss changes only the patch-location distribution.  Each source image
still produces exactly one patch, and the standard Patch_EDMLoss noise and
denoising objective are inherited unchanged.
"""

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from torch_utils import persistence, training_stats
from training.agg_sampler import sample_single_locations
from training.patch_loss import Patch_EDMLoss
from training.patch_overlap_loss import _extract_patches, _position_maps


@persistence.persistent_class
class SimplifiedAggPatchEDMLoss(Patch_EDMLoss):
    """Apply simplified AGG sampling while retaining one patch per image."""

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        agg_rho=0.65,
        agg_gamma=1.0,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        if not 0.0 <= agg_rho <= 1.0:
            raise ValueError('agg_rho must be in [0, 1]')
        if agg_gamma < 0.0:
            raise ValueError('agg_gamma must be non-negative')

        self.agg_rho = float(agg_rho)
        self.agg_gamma = float(agg_gamma)

        print(
            '[SimplifiedAggPatchEDMLoss] '
            f'rho={self.agg_rho}, gamma={self.agg_gamma}, '
            'patches_per_source=1'
        )

    def pachify(self, images, patch_size, padding=None):
        """Sample and extract exactly one AGG patch from every source image."""
        resolution = int(images.shape[2])
        if padding is not None and int(padding) > 0:
            padding = int(padding)
            padded = torch.zeros(
                images.shape[0],
                images.shape[1],
                images.shape[2] + 2 * padding,
                images.shape[3] + 2 * padding,
                dtype=images.dtype,
                device=images.device,
            )
            padded[:, :, padding:-padding, padding:-padding] = images
        else:
            padded = images

        top, left = sample_single_locations(
            padded,
            int(patch_size),
            self.agg_rho,
            self.agg_gamma,
        )
        patches = _extract_patches(
            padded, top, left, int(patch_size)
        )
        positions = _position_maps(
            top,
            left,
            padded.shape[0],
            int(patch_size),
            resolution,
            padded.device,
        )

        if patches.shape[0] != images.shape[0]:
            raise RuntimeError(
                'simplified AGG must return exactly one patch per source image'
            )

        training_stats.report(
            'Sampler/agg_rho', patches.new_tensor(self.agg_rho)
        )
        training_stats.report(
            'Sampler/agg_gamma', patches.new_tensor(self.agg_gamma)
        )
        training_stats.report(
            'Sampler/patches_per_source', patches.new_tensor(1.0)
        )
        training_stats.report(
            'Sampler/patch_size', patches.new_tensor(float(patch_size))
        )
        return patches, positions

