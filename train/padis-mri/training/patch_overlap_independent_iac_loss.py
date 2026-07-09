"""PaDIS-MRI independent-noise, center-guided overlap consistency loss
with internal-aware anatomical weighting.

This file is a drop-in experimental variant of patch_overlap_independent_loss.py.

Core change relative to IndependentNoiseOverlapPatch_EDMLoss:
    aux_pixel = weight * confidence * anatomy_weight
                * (D_boundary - D_interior.detach()).square()

The anatomy_weight is computed from the clean full image:
    internal foreground: anatomy_internal_weight
    outer foreground band: anatomy_outer_weight
    background: anatomy_bg_weight

The purpose is to keep the original center-boundary overlap consistency inside
reliable brain/internal regions, while suppressing strong outer-contour/skull
regions that can dominate the auxiliary loss.
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


def _disk_kernel(radius, device, dtype):
    """Return a 2D disk structuring element with shape [1, 1, K, K]."""
    radius = int(radius)
    if radius <= 0:
        return torch.ones([1, 1, 1, 1], device=device, dtype=dtype)

    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    kernel = ((xx.square() + yy.square()) <= float(radius * radius)).to(dtype)
    return kernel.view(1, 1, kernel.shape[0], kernel.shape[1])


def _binary_erosion(mask, radius):
    """Binary erosion for mask [B, 1, H, W] implemented with conv2d."""
    radius = int(radius)
    if radius <= 0:
        return mask

    kernel = _disk_kernel(radius, mask.device, mask.dtype)
    required = kernel.sum()
    conv = F.conv2d(mask.to(dtype=mask.dtype), kernel, padding=radius)
    return (conv >= required - 1e-6).to(mask.dtype)


def _magnitude_image(x):
    """Compute magnitude image [B, 1, H, W] from [B, C, H, W]."""
    if x.shape[1] == 1:
        return x.abs()
    return torch.sqrt(x.square().sum(dim=1, keepdim=True).clamp_min(0))


def _make_internal_aware_weight_full(
    clean_full,
    fg_threshold=0.05,
    erosion_radius=8,
    internal_weight=1.0,
    outer_weight=0.2,
    bg_weight=0.0,
):
    """Build full-image anatomical weight map [B, 1, H, W].

    The foreground is estimated from the clean image magnitude. The internal
    region is foreground eroded by erosion_radius. The outer band is
    foreground - internal. Background is outside foreground.
    """
    mag = _magnitude_image(clean_full).detach()

    fg = (mag > float(fg_threshold)).to(clean_full.dtype)
    internal = _binary_erosion(fg, int(erosion_radius))
    outer = (fg - internal).clamp_min(0.0)
    bg = (1.0 - fg).clamp_min(0.0)

    anatomy_weight = (
        float(internal_weight) * internal
        + float(outer_weight) * outer
        + float(bg_weight) * bg
    )

    return anatomy_weight, fg, internal, outer


def _make_internal_aware_weight_overlap(
    clean_full,
    top1,
    left1,
    top2,
    left2,
    horizontal,
    overlap,
    patch_size,
    fg_threshold=0.05,
    erosion_radius=8,
    internal_weight=1.0,
    outer_weight=0.2,
    bg_weight=0.0,
):
    """Return anatomy weight on the physical overlap region.

    Output shape is [B, 1, patch_size, overlap] after the same orientation
    convention as _overlap_views().
    """
    weight_full, fg_full, internal_full, outer_full = _make_internal_aware_weight_full(
        clean_full=clean_full,
        fg_threshold=fg_threshold,
        erosion_radius=erosion_radius,
        internal_weight=internal_weight,
        outer_weight=outer_weight,
        bg_weight=bg_weight,
    )

    weight1 = _extract_patches(weight_full, top1, left1, int(patch_size))
    weight2 = _extract_patches(weight_full, top2, left2, int(patch_size))

    weight1_overlap, weight2_overlap = _overlap_views(weight1, weight2, horizontal, overlap)

    # They should correspond to the same physical pixels. Average for numerical
    # robustness and to avoid orientation-specific indexing mistakes.
    anatomy_weight = 0.5 * (weight1_overlap + weight2_overlap)

    fg1 = _extract_patches(fg_full, top1, left1, int(patch_size))
    fg2 = _extract_patches(fg_full, top2, left2, int(patch_size))
    internal1 = _extract_patches(internal_full, top1, left1, int(patch_size))
    internal2 = _extract_patches(internal_full, top2, left2, int(patch_size))
    outer1 = _extract_patches(outer_full, top1, left1, int(patch_size))
    outer2 = _extract_patches(outer_full, top2, left2, int(patch_size))

    fg_o1, fg_o2 = _overlap_views(fg1, fg2, horizontal, overlap)
    internal_o1, internal_o2 = _overlap_views(internal1, internal2, horizontal, overlap)
    outer_o1, outer_o2 = _overlap_views(outer1, outer2, horizontal, overlap)

    fg_overlap = 0.5 * (fg_o1 + fg_o2)
    internal_overlap = 0.5 * (internal_o1 + internal_o2)
    outer_overlap = 0.5 * (outer_o1 + outer_o2)

    return anatomy_weight, fg_overlap, internal_overlap, outer_overlap


@persistence.persistent_class
class IndependentNoiseInternalAwareOverlapPatch_EDMLoss(Patch_EDMLoss):
    """Internal-aware variant of center-guided overlap consistency.

    It keeps the original independent-noise CG-OL logic, but suppresses the
    overlap auxiliary loss on outer contour and background regions.
    """

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        lambda_overlap=1.0,
        active_patch_size=64,
        anatomy_fg_threshold=0.05,
        anatomy_erosion_radius=8,
        anatomy_internal_weight=1.0,
        anatomy_outer_weight=0.2,
        anatomy_bg_weight=0.0,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)

        if lambda_overlap < 0:
            raise ValueError('lambda_overlap must be non-negative')
        if anatomy_erosion_radius < 0:
            raise ValueError('anatomy_erosion_radius must be non-negative')
        if min(anatomy_internal_weight, anatomy_outer_weight, anatomy_bg_weight) < 0:
            raise ValueError('anatomy weights must be non-negative')

        self.lambda_overlap = float(lambda_overlap)
        self.active_patch_size = int(active_patch_size)

        self.anatomy_fg_threshold = float(anatomy_fg_threshold)
        self.anatomy_erosion_radius = int(anatomy_erosion_radius)
        self.anatomy_internal_weight = float(anatomy_internal_weight)
        self.anatomy_outer_weight = float(anatomy_outer_weight)
        self.anatomy_bg_weight = float(anatomy_bg_weight)

    def __call__(
        self,
        net,
        images,
        patch_size,
        resolution,
        labels=None,
        augment_pipe=None,
    ):
        # Non-active patch sizes use the original PaDIS Patch_EDMLoss.
        if int(patch_size) != self.active_patch_size:
            return super().__call__(
                net=net,
                images=images,
                patch_size=patch_size,
                resolution=resolution,
                labels=labels,
                augment_pipe=augment_pipe,
            )

        # Geometry correspondence between the two patches would be invalid if
        # random geometric augmentation is applied independently.
        if augment_pipe is not None:
            raise ValueError('Internal-aware independent overlap experiment requires --augment=0')

        pair_batch = images.shape[0]
        clean_full = images
        labels_pair_source = labels

        if pair_batch < 1:
            raise ValueError('Current batch has no usable images')

        # Sample one half-shift overlapping patch pair per full image.
        top1, left1, top2, left2, horizontal, overlap = _sample_overlap_geometry(
            clean_full,
            int(patch_size),
        )

        clean1 = _extract_patches(clean_full, top1, left1, int(patch_size))
        clean2 = _extract_patches(clean_full, top2, left2, int(patch_size))

        pos1 = _position_maps(top1, left1, pair_batch, int(patch_size), resolution, images.device)
        pos2 = _position_maps(top2, left2, pair_batch, int(patch_size), resolution, images.device)

        # Same sigma, independent Gaussian noise realizations.
        rnd_normal = torch.randn([pair_batch, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma.square() + self.sigma_data ** 2) / (sigma * self.sigma_data).square()

        noisy1 = clean1 + sigma * torch.randn_like(clean1)
        noisy2 = clean2 + sigma * torch.randn_like(clean2)

        noisy_pair = torch.cat([noisy1, noisy2], dim=0)
        sigma_pair = torch.cat([sigma, sigma], dim=0)
        pos_pair = torch.cat([pos1, pos2], dim=0)
        labels_pair = (
            torch.cat([labels_pair_source, labels_pair_source], dim=0)
            if labels_pair_source is not None else None
        )

        denoised_pair = net(
            noisy_pair,
            sigma_pair,
            x_pos=pos_pair,
            class_labels=labels_pair,
            augment_labels=None,
        )
        D1 = denoised_pair[:pair_batch]
        D2 = denoised_pair[pair_batch:]

        # Main EDM denoising objective, scale-matched to the original paired CG-OL.
        edm_loss_1 = weight * (D1 - clean1).square()
        edm_loss_2 = weight * (D2 - clean2).square()
        edm_loss = 0.5 * (edm_loss_1 + edm_loss_2)
        training_stats.report('Loss/edm', edm_loss.detach())

        if self.lambda_overlap == 0:
            zero = edm_loss.detach().new_zeros([])
            training_stats.report('Loss/overlap_center_raw', zero)
            training_stats.report('Loss/overlap_center_raw_ratio', zero)
            training_stats.report('Loss/overlap_center_weighted', zero)
            training_stats.report('Loss/overlap_center_ratio', zero)
            training_stats.report('Loss/overlap_anatomy_weight_mean', zero)
            training_stats.report('Loss/overlap_internal_fraction', zero)
            training_stats.report('Loss/overlap_outer_fraction', zero)
            return edm_loss

        # Overlap predictions for the same physical region.
        D1_overlap, D2_overlap = _overlap_views(D1, D2, horizontal, overlap)

        # Center-boundary guidance: the view farther from the patch boundary is
        # treated as the more reliable interior prediction.
        d1, d2 = _boundary_distances(
            horizontal,
            int(patch_size),
            overlap,
            images.device,
            images.dtype,
        )
        interior_is_1 = d1 >= d2
        D_interior = torch.where(interior_is_1, D1_overlap, D2_overlap)
        D_boundary = torch.where(interior_is_1, D2_overlap, D1_overlap)

        confidence_raw = (d1 - d2).abs()
        confidence_mean = confidence_raw.mean(dim=(1, 2, 3), keepdim=True)
        confidence = confidence_raw / confidence_mean.clamp_min(1e-12)

        # New: internal-aware anatomical weighting on the overlap region.
        anatomy_weight, fg_overlap, internal_overlap, outer_overlap = _make_internal_aware_weight_overlap(
            clean_full=clean_full,
            top1=top1,
            left1=left1,
            top2=top2,
            left2=left2,
            horizontal=horizontal,
            overlap=overlap,
            patch_size=int(patch_size),
            fg_threshold=self.anatomy_fg_threshold,
            erosion_radius=self.anatomy_erosion_radius,
            internal_weight=self.anatomy_internal_weight,
            outer_weight=self.anatomy_outer_weight,
            bg_weight=self.anatomy_bg_weight,
        )

        # Only update boundary prediction; interior prediction is a fixed target.
        aux_pixel = (
            weight
            * confidence
            * anatomy_weight
            * (D_boundary - D_interior.detach()).square()
        )

        aux_per_pair = aux_pixel.mean(dim=(1, 2, 3), keepdim=True)
        auxiliary_loss = aux_per_pair.expand_as(edm_loss)
        weighted_auxiliary = self.lambda_overlap * auxiliary_loss

        edm_mean = edm_loss.detach().mean().clamp_min(1e-12)
        raw_ratio = auxiliary_loss.detach().mean() / edm_mean
        weighted_ratio = weighted_auxiliary.detach().mean() / edm_mean

        training_stats.report('Loss/overlap_center_raw', aux_per_pair.detach())
        training_stats.report('Loss/overlap_center_raw_ratio', raw_ratio)
        training_stats.report('Loss/overlap_center_weighted', weighted_auxiliary.detach())
        training_stats.report('Loss/overlap_center_ratio', weighted_ratio)
        training_stats.report(
            'Loss/overlap_center_lambda',
            edm_loss.detach().new_tensor(self.lambda_overlap),
        )

        training_stats.report(
            'Loss/overlap_anatomy_weight_mean',
            anatomy_weight.detach().mean(),
        )
        training_stats.report(
            'Loss/overlap_internal_fraction',
            (internal_overlap.detach() > 0.5).to(edm_loss.dtype).mean(),
        )
        training_stats.report(
            'Loss/overlap_outer_fraction',
            (outer_overlap.detach() > 0.5).to(edm_loss.dtype).mean(),
        )
        training_stats.report(
            'Loss/overlap_fg_fraction',
            (fg_overlap.detach() > 0.5).to(edm_loss.dtype).mean(),
        )

        return edm_loss + weighted_auxiliary
