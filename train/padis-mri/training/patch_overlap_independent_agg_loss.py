"""Simplified AGG sampling with independent-noise overlap consistency."""

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from torch_utils import persistence, training_stats
from training.agg_sampler import (
    sample_overlap_locations,
    sample_single_locations,
)
from training.patch_loss import Patch_EDMLoss
from training.patch_overlap_loss import (
    _boundary_distances,
    _extract_patches,
    _overlap_views,
    _position_maps,
)


@persistence.persistent_class
class AggIndependentNoiseOverlapPatch_EDMLoss(Patch_EDMLoss):
    """Combine structure-guided patch sampling and overlap consistency."""

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        agg_rho=0.65,
        agg_gamma=1.0,
        lambda_oc=0.3,
        active_patch_size=64,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        if not 0.0 <= agg_rho <= 1.0:
            raise ValueError('agg_rho must be in [0, 1]')
        if agg_gamma < 0.0:
            raise ValueError('agg_gamma must be non-negative')
        if lambda_oc < 0.0:
            raise ValueError('lambda_oc must be non-negative')
        if active_patch_size < 2:
            raise ValueError('active_patch_size must be at least 2')

        self.agg_rho = float(agg_rho)
        self.agg_gamma = float(agg_gamma)
        self.lambda_oc = float(lambda_oc)
        self.active_patch_size = int(active_patch_size)

        print(
            '[AggIndependentNoiseOverlapPatch_EDMLoss] '
            f'rho={self.agg_rho}, gamma={self.agg_gamma}, '
            f'lambda_oc={self.lambda_oc}, '
            f'overlap_patch={self.active_patch_size}'
        )

    def pachify(self, images, patch_size, padding=None):
        """Sample structure-guided patches for the standard EDM branches."""
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
            padded, int(patch_size), self.agg_rho, self.agg_gamma
        )
        patches = _extract_patches(padded, top, left, int(patch_size))
        positions = _position_maps(
            top,
            left,
            padded.shape[0],
            int(patch_size),
            resolution,
            padded.device,
        )
        training_stats.report(
            'Sampler/agg_rho', patches.new_tensor(self.agg_rho)
        )
        return patches, positions

    def __call__(
        self,
        net,
        images,
        patch_size,
        resolution,
        labels=None,
        augment_pipe=None,
    ):
        patch_size = int(patch_size)
        if patch_size != self.active_patch_size:
            return super().__call__(
                net=net,
                images=images,
                patch_size=patch_size,
                resolution=resolution,
                labels=labels,
                augment_pipe=augment_pipe,
            )

        if augment_pipe is not None:
            raise ValueError('AGG overlap training requires --augment=0')
        pair_batch = images.shape[0]
        if pair_batch < 1:
            raise ValueError('current batch contains no source image')

        top1, left1, top2, left2, horizontal, overlap = (
            sample_overlap_locations(
                images, patch_size, self.agg_rho, self.agg_gamma
            )
        )
        clean1 = _extract_patches(images, top1, left1, patch_size)
        clean2 = _extract_patches(images, top2, left2, patch_size)
        pos1 = _position_maps(
            top1, left1, pair_batch, patch_size, resolution, images.device
        )
        pos2 = _position_maps(
            top2, left2, pair_batch, patch_size, resolution, images.device
        )

        sigma = (
            torch.randn([pair_batch, 1, 1, 1], device=images.device)
            * self.P_std
            + self.P_mean
        ).exp()
        weight = (
            sigma.square() + self.sigma_data ** 2
        ) / (sigma * self.sigma_data).square()

        noisy1 = clean1 + sigma * torch.randn_like(clean1)
        noisy2 = clean2 + sigma * torch.randn_like(clean2)
        labels_pair = (
            torch.cat([labels, labels], dim=0) if labels is not None else None
        )
        denoised_pair = net(
            torch.cat([noisy1, noisy2], dim=0),
            torch.cat([sigma, sigma], dim=0),
            x_pos=torch.cat([pos1, pos2], dim=0),
            class_labels=labels_pair,
            augment_labels=None,
        )
        denoised1 = denoised_pair[:pair_batch]
        denoised2 = denoised_pair[pair_batch:]

        edm_loss_1 = weight * (denoised1 - clean1).square()
        edm_loss_2 = weight * (denoised2 - clean2).square()
        edm_loss = 0.5 * (edm_loss_1 + edm_loss_2)
        training_stats.report('Loss/edm', edm_loss.detach())
        training_stats.report(
            'Sampler/agg_rho', edm_loss.new_tensor(self.agg_rho)
        )

        if self.lambda_oc == 0.0:
            zero = edm_loss.detach().new_zeros([])
            training_stats.report('Loss/overlap_center_raw', zero)
            training_stats.report('Loss/overlap_center_raw_ratio', zero)
            training_stats.report('Loss/overlap_center_weighted', zero)
            training_stats.report('Loss/overlap_center_ratio', zero)
            return edm_loss

        overlap1, overlap2 = _overlap_views(
            denoised1, denoised2, horizontal, overlap
        )
        distance1, distance2 = _boundary_distances(
            horizontal,
            patch_size,
            overlap,
            images.device,
            images.dtype,
        )
        interior_is_1 = distance1 >= distance2
        denoised_interior = torch.where(interior_is_1, overlap1, overlap2)
        denoised_boundary = torch.where(interior_is_1, overlap2, overlap1)

        confidence_raw = (distance1 - distance2).abs()
        confidence = confidence_raw / confidence_raw.mean(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(1e-12)
        auxiliary_pixel = (
            weight
            * confidence
            * (denoised_boundary - denoised_interior.detach()).square()
        )
        auxiliary_per_pair = auxiliary_pixel.mean(
            dim=(1, 2, 3), keepdim=True
        )
        auxiliary_loss = auxiliary_per_pair.expand_as(edm_loss)
        weighted_auxiliary = self.lambda_oc * auxiliary_loss

        edm_mean = edm_loss.detach().mean().clamp_min(1e-12)
        raw_ratio = auxiliary_loss.detach().mean() / edm_mean
        weighted_ratio = weighted_auxiliary.detach().mean() / edm_mean
        training_stats.report(
            'Loss/overlap_center_raw', auxiliary_per_pair.detach()
        )
        training_stats.report('Loss/overlap_center_raw_ratio', raw_ratio)
        training_stats.report(
            'Loss/overlap_center_weighted', weighted_auxiliary.detach()
        )
        training_stats.report('Loss/overlap_center_ratio', weighted_ratio)
        training_stats.report(
            'Loss/overlap_center_lambda',
            edm_loss.detach().new_tensor(self.lambda_oc),
        )
        return edm_loss + weighted_auxiliary
