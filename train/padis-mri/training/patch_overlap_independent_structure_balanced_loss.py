"""PaDIS-MRI balanced structure-aware overlap consistency.

This module adds two independent experiments on top of the validated CG-OL
only64 setup:

1. gradient:
   Mix the original pixel overlap loss with Sobel-gradient consistency.

2. wavelet:
   Mix the original pixel overlap loss with Haar high-frequency consistency
   over LH, HL, and HH bands.

Both experiments:
- activate only for 64x64 patches;
- use the same sigma within each overlapping pair;
- use independent Gaussian noise realizations;
- use the interior prediction as a detached teacher;
- keep the EDM main loss unchanged;
- use fixed lambda_overlap=0.3 by default.

To avoid increasing the effective auxiliary-loss magnitude, the structural
component is detached-scale-aligned to the original pixel overlap loss for
each sample. With structure_mix=0.5, the scalar auxiliary loss remains close
to the original CG-OL scale while half of its gradient direction is replaced
by the selected structural consistency gradient.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from torch_utils import persistence, training_stats
from training.patch_loss import Patch_EDMLoss
from training.patch_overlap_loss import (
    _boundary_distances,
    _extract_patches,
    _overlap_views,
    _position_maps,
    _sample_overlap_geometry,
)


@persistence.persistent_class
class IndependentNoiseStructureOverlapPatchEDMLoss(Patch_EDMLoss):
    """Balanced pixel/structure overlap loss for the active patch size."""

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        lambda_overlap=0.3,
        active_patch_size=64,
        structure_mode='gradient',
        structure_mix=0.5,
        structure_scale_min=0.1,
        structure_scale_max=10.0,
    ):
        super().__init__(
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=sigma_data,
        )

        if lambda_overlap < 0:
            raise ValueError('lambda_overlap must be non-negative')
        if structure_mode not in {'gradient', 'wavelet'}:
            raise ValueError(
                "structure_mode must be either 'gradient' or 'wavelet'"
            )
        if not 0.0 <= structure_mix <= 1.0:
            raise ValueError('structure_mix must be in [0, 1]')
        if structure_scale_min <= 0:
            raise ValueError('structure_scale_min must be positive')
        if structure_scale_max < structure_scale_min:
            raise ValueError(
                'structure_scale_max must be >= structure_scale_min'
            )

        self.lambda_overlap = float(lambda_overlap)
        self.active_patch_size = int(active_patch_size)
        self.structure_mode = str(structure_mode)
        self.structure_mix = float(structure_mix)
        self.structure_scale_min = float(structure_scale_min)
        self.structure_scale_max = float(structure_scale_max)

    @staticmethod
    def _sobel_gradients(x):
        """Return channel-wise Sobel x/y gradients with replicate padding."""
        channels = x.shape[1]
        kernel_x = x.new_tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]) / 8.0
        kernel_y = x.new_tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ]) / 8.0

        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(
            channels, 1, 1, 1
        )
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(
            channels, 1, 1, 1
        )

        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(x_pad, kernel_x, groups=channels)
        grad_y = F.conv2d(x_pad, kernel_y, groups=channels)
        return grad_x, grad_y

    @staticmethod
    def _haar_high_frequency(x):
        """Return orthonormal Haar LH, HL, and HH coefficients."""
        height, width = x.shape[-2:]
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                'Haar overlap dimensions must both be even, '
                f'but got {height}x{width}'
            )

        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return lh, hl, hh

    def _structure_per_pair(
        self,
        boundary,
        interior,
        confidence,
        edm_weight,
    ):
        """Compute one structural loss value for every overlapping pair."""
        interior = interior.detach()

        if self.structure_mode == 'gradient':
            boundary_gx, boundary_gy = self._sobel_gradients(boundary)
            interior_gx, interior_gy = self._sobel_gradients(interior)

            structure_pixel = 0.5 * (
                (boundary_gx - interior_gx).square()
                + (boundary_gy - interior_gy).square()
            )
            structure_pixel = edm_weight * confidence * structure_pixel
            return structure_pixel.mean(dim=(1, 2, 3), keepdim=True)

        boundary_bands = self._haar_high_frequency(boundary)
        interior_bands = self._haar_high_frequency(interior)

        confidence_half = F.avg_pool2d(
            confidence,
            kernel_size=2,
            stride=2,
        )
        band_error = torch.stack(
            [
                (boundary_band - interior_band).square()
                for boundary_band, interior_band
                in zip(boundary_bands, interior_bands)
            ],
            dim=0,
        ).mean(dim=0)

        structure_pixel = edm_weight * confidence_half * band_error
        return structure_pixel.mean(dim=(1, 2, 3), keepdim=True)

    def __call__(
        self,
        net,
        images,
        patch_size,
        resolution,
        labels=None,
        augment_pipe=None,
    ):
        # Preserve the original PaDIS Patch_EDMLoss for 16x16 and 32x32.
        if int(patch_size) != self.active_patch_size:
            return super().__call__(
                net=net,
                images=images,
                patch_size=patch_size,
                resolution=resolution,
                labels=labels,
                augment_pipe=augment_pipe,
            )

        if augment_pipe is not None:
            raise ValueError(
                'Structure-overlap experiments require --augment=0'
            )

        pair_batch = images.shape[0]
        if pair_batch < 1:
            raise ValueError('No source image is available in this batch')

        clean_full = images
        labels_pair_source = labels

        top1, left1, top2, left2, horizontal, overlap = (
            _sample_overlap_geometry(clean_full, int(patch_size))
        )
        clean1 = _extract_patches(
            clean_full, top1, left1, int(patch_size)
        )
        clean2 = _extract_patches(
            clean_full, top2, left2, int(patch_size)
        )

        pos1 = _position_maps(
            top1,
            left1,
            pair_batch,
            int(patch_size),
            resolution,
            images.device,
        )
        pos2 = _position_maps(
            top2,
            left2,
            pair_batch,
            int(patch_size),
            resolution,
            images.device,
        )

        # Same sigma within a pair, independent noise realization.
        rnd_normal = torch.randn(
            [pair_batch, 1, 1, 1],
            device=images.device,
        )
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        edm_weight = (
            sigma.square() + self.sigma_data ** 2
        ) / (sigma * self.sigma_data).square()

        noisy1 = clean1 + sigma * torch.randn_like(clean1)
        noisy2 = clean2 + sigma * torch.randn_like(clean2)

        noisy_pair = torch.cat([noisy1, noisy2], dim=0)
        sigma_pair = torch.cat([sigma, sigma], dim=0)
        pos_pair = torch.cat([pos1, pos2], dim=0)
        labels_pair = (
            torch.cat([labels_pair_source, labels_pair_source], dim=0)
            if labels_pair_source is not None
            else None
        )

        denoised_pair = net(
            noisy_pair,
            sigma_pair,
            x_pos=pos_pair,
            class_labels=labels_pair,
            augment_labels=None,
        )
        denoised1 = denoised_pair[:pair_batch]
        denoised2 = denoised_pair[pair_batch:]

        # Keep the original two-view EDM supervision unchanged.
        edm_loss_1 = edm_weight * (denoised1 - clean1).square()
        edm_loss_2 = edm_weight * (denoised2 - clean2).square()
        edm_loss = 0.5 * (edm_loss_1 + edm_loss_2)
        training_stats.report('Loss/edm', edm_loss.detach())
        training_stats.report(
            'Loss/sigma_mean',
            sigma.detach().reshape(-1).mean(),
        )

        if self.lambda_overlap == 0:
            zero = edm_loss.detach().new_zeros([])
            for key in [
                'Loss/overlap_pixel_raw',
                'Loss/overlap_structure_raw',
                'Loss/overlap_structure_balanced',
                'Loss/overlap_combined_raw',
                'Loss/overlap_combined_weighted',
                'Loss/overlap_structure_scale_mean',
                'Loss/overlap_structure_scale_min',
                'Loss/overlap_structure_scale_max',
            ]:
                training_stats.report(key, zero)
            return edm_loss

        denoised1_overlap, denoised2_overlap = _overlap_views(
            denoised1,
            denoised2,
            horizontal,
            overlap,
        )

        d1, d2 = _boundary_distances(
            horizontal,
            int(patch_size),
            overlap,
            images.device,
            images.dtype,
        )
        interior_is_1 = d1 >= d2
        interior = torch.where(
            interior_is_1,
            denoised1_overlap,
            denoised2_overlap,
        )
        boundary = torch.where(
            interior_is_1,
            denoised2_overlap,
            denoised1_overlap,
        )

        confidence_raw = (d1 - d2).abs()
        confidence = confidence_raw / confidence_raw.mean(
            dim=(1, 2, 3),
            keepdim=True,
        ).clamp_min(1e-12)

        # Original CG-OL pixel consistency.
        pixel_map = (
            edm_weight
            * confidence
            * (boundary - interior.detach()).square()
        )
        pixel_per_pair = pixel_map.mean(
            dim=(1, 2, 3),
            keepdim=True,
        )

        # Selected structural consistency.
        structure_per_pair = self._structure_per_pair(
            boundary=boundary,
            interior=interior,
            confidence=confidence,
            edm_weight=edm_weight,
        )

        # Match the structural scalar scale to the original pixel loss.
        # The ratio is detached, so the model cannot manipulate the scale.
        structure_scale = (
            pixel_per_pair.detach()
            / structure_per_pair.detach().clamp_min(1e-12)
        ).clamp(
            self.structure_scale_min,
            self.structure_scale_max,
        )
        structure_balanced = structure_scale * structure_per_pair

        combined_per_pair = (
            (1.0 - self.structure_mix) * pixel_per_pair
            + self.structure_mix * structure_balanced
        )

        auxiliary_loss = combined_per_pair.expand_as(edm_loss)
        weighted_auxiliary = self.lambda_overlap * auxiliary_loss

        pixel_flat = pixel_per_pair.detach().reshape(-1)
        structure_flat = structure_per_pair.detach().reshape(-1)
        structure_balanced_flat = structure_balanced.detach().reshape(-1)
        combined_flat = combined_per_pair.detach().reshape(-1)
        weighted_flat = (
            self.lambda_overlap * combined_per_pair.detach()
        ).reshape(-1)
        scale_flat = structure_scale.detach().reshape(-1)

        training_stats.report(
            'Loss/overlap_pixel_raw',
            pixel_flat,
        )
        training_stats.report(
            'Loss/overlap_structure_raw',
            structure_flat,
        )
        training_stats.report(
            'Loss/overlap_structure_balanced',
            structure_balanced_flat,
        )
        training_stats.report(
            'Loss/overlap_combined_raw',
            combined_flat,
        )
        training_stats.report(
            'Loss/overlap_combined_weighted',
            weighted_flat,
        )
        training_stats.report(
            'Loss/overlap_structure_scale_mean',
            scale_flat.mean(),
        )
        training_stats.report(
            'Loss/overlap_structure_scale_min',
            scale_flat.min(),
        )
        training_stats.report(
            'Loss/overlap_structure_scale_max',
            scale_flat.max(),
        )
        training_stats.report(
            'Loss/overlap_structure_mix',
            edm_loss.detach().new_tensor(self.structure_mix),
        )
        training_stats.report(
            'Loss/overlap_lambda',
            edm_loss.detach().new_tensor(self.lambda_overlap),
        )

        return edm_loss + weighted_auxiliary
