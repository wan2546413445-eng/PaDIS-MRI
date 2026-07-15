"""AGG-mild sampling + independent-noise center-guided overlap loss.

- 16x16 and 32x32: anatomy-gated gradient-aware single-patch sampling.
- active_patch_size (default 64): anatomy-gated gradient-aware overlapping-pair
  sampling, followed by the independent-noise center-guided consistency loss.
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
)


@persistence.persistent_class
class AggIndependentNoiseOverlapPatch_EDMLoss(Patch_EDMLoss):
    """Combine AGG-mild sampling with independent overlap consistency."""

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        lambda_overlap=0.3,
        active_patch_size=64,
        agg_sample_prob=0.65,
        agg_thr_ratio=0.05,
        agg_base=0.08,
        agg_radius_scale=1.15,
        agg_tau_scale=0.30,
        agg_min_pixels=64,
        agg_grad_alpha=1.0,
        agg_grad_clip_q=0.95,
        agg_gate_base=0.25,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)

        if lambda_overlap < 0:
            raise ValueError('lambda_overlap must be >= 0')
        if not 0.0 <= agg_sample_prob <= 1.0:
            raise ValueError('agg_sample_prob must be in [0, 1]')
        if not 0.0 < agg_grad_clip_q <= 1.0:
            raise ValueError('agg_grad_clip_q must be in (0, 1]')
        if not 0.0 <= agg_gate_base <= 1.0:
            raise ValueError('agg_gate_base must be in [0, 1]')
        if active_patch_size < 2:
            raise ValueError('active_patch_size must be >= 2')

        self.lambda_overlap = float(lambda_overlap)
        self.active_patch_size = int(active_patch_size)

        self.agg_sample_prob = float(agg_sample_prob)
        self.agg_thr_ratio = float(agg_thr_ratio)
        self.agg_base = float(agg_base)
        self.agg_radius_scale = float(agg_radius_scale)
        self.agg_tau_scale = float(agg_tau_scale)
        self.agg_min_pixels = int(agg_min_pixels)
        self.agg_grad_alpha = float(agg_grad_alpha)
        self.agg_grad_clip_q = float(agg_grad_clip_q)
        self.agg_gate_base = float(agg_gate_base)

        print(
            '[AggIndependentNoiseOverlapPatch_EDMLoss] enabled: '
            f'active_patch={self.active_patch_size}, '
            f'lambda_overlap={self.lambda_overlap}, '
            f'prob={self.agg_sample_prob}, '
            f'thr={self.agg_thr_ratio}, '
            f'base={self.agg_base}, '
            f'radius_scale={self.agg_radius_scale}, '
            f'tau_scale={self.agg_tau_scale}, '
            f'grad_alpha={self.agg_grad_alpha}, '
            f'grad_clip_q={self.agg_grad_clip_q}, '
            f'gate_base={self.agg_gate_base}'
        )

    def _agg_score_map(self, image, patch_size):
        """Return AGG score for every valid top-left position of one image.

        Args:
            image: [C, H, W], already containing the dataset-level zero padding.
            patch_size: square patch size.

        Returns:
            Tensor [H-P+1, W-P+1], or None when anatomical support is invalid.
        """
        patch_size = int(patch_size)
        _, height, width = image.shape
        if patch_size > height or patch_size > width:
            raise ValueError('patch_size is larger than the input image')

        use_channels = min(2, image.shape[0])
        magnitude = image[:use_channels].float().square().sum(dim=0).sqrt()
        peak = magnitude.amax().clamp_min(1e-8)
        support = (magnitude > self.agg_thr_ratio * peak).float()
        mass = support.sum()
        if int(mass.item()) < self.agg_min_pixels:
            return None

        rows = torch.arange(height, device=image.device, dtype=torch.float32)
        cols = torch.arange(width, device=image.device, dtype=torch.float32)
        center_y = (support.sum(dim=1) * rows).sum() / mass
        center_x = (support.sum(dim=0) * cols).sum() / mass

        radius = torch.sqrt(mass / torch.pi) * self.agg_radius_scale
        tau = torch.clamp(radius * self.agg_tau_scale, min=8.0)

        candidate_y = (
            torch.arange(height - patch_size + 1, device=image.device, dtype=torch.float32)
            + patch_size / 2.0
        )
        candidate_x = (
            torch.arange(width - patch_size + 1, device=image.device, dtype=torch.float32)
            + patch_size / 2.0
        )
        grid_y, grid_x = torch.meshgrid(candidate_y, candidate_x, indexing='ij')
        distance = torch.sqrt((grid_y - center_y).square() + (grid_x - center_x).square())
        outside = torch.clamp(distance - radius, min=0.0)
        radial_score = torch.exp(-0.5 * (outside / tau).square())

        grad_x = torch.zeros_like(magnitude)
        grad_y = torch.zeros_like(magnitude)
        grad_x[:, 1:] = magnitude[:, 1:] - magnitude[:, :-1]
        grad_y[1:, :] = magnitude[1:, :] - magnitude[:-1, :]
        gradient = torch.sqrt(grad_x.square() + grad_y.square())

        gradient_patch = F.avg_pool2d(
            gradient[None, None], kernel_size=patch_size, stride=1
        )[0, 0]
        support_fraction = F.avg_pool2d(
            support[None, None], kernel_size=patch_size, stride=1
        )[0, 0].clamp(0.0, 1.0)

        positive = gradient_patch.flatten()
        positive = positive[positive > 0]
        if positive.numel() < 8:
            gradient_normalized = torch.zeros_like(gradient_patch)
        else:
            clip_value = torch.quantile(
                positive, self.agg_grad_clip_q
            ).clamp_min(1e-8)
            gradient_normalized = (gradient_patch / clip_value).clamp(0.0, 1.0)

        anatomy_gate = (
            self.agg_gate_base
            + (1.0 - self.agg_gate_base) * support_fraction
        )
        score = (
            self.agg_base
            + radial_score
            * anatomy_gate
            * (1.0 + self.agg_grad_alpha * gradient_normalized)
        )
        score = torch.nan_to_num(
            score,
            nan=self.agg_base,
            posinf=1.0 + self.agg_base + self.agg_grad_alpha,
            neginf=self.agg_base,
        ).clamp_min(1e-8)
        return score

    def _sample_single_locations(self, images, patch_size):
        """Sample one AGG-mild patch location per source image."""
        patch_size = int(patch_size)
        batch, _, height, width = images.shape
        max_top = height - patch_size
        max_left = width - patch_size
        if min(max_top, max_left) < 0:
            raise ValueError('patch_size is larger than the input image')

        top = torch.randint(0, max_top + 1, [batch], device=images.device)
        left = torch.randint(0, max_left + 1, [batch], device=images.device)
        requested = torch.rand(batch, device=images.device) < self.agg_sample_prob
        used = torch.zeros(batch, device=images.device, dtype=torch.bool)

        for index in range(batch):
            if not bool(requested[index].item()):
                continue
            score = self._agg_score_map(images[index], patch_size)
            if score is None:
                continue
            flat_index = torch.multinomial(score.flatten(), 1)[0]
            score_width = score.shape[1]
            sampled_top = torch.div(flat_index, score_width, rounding_mode='floor')
            sampled_left = flat_index - sampled_top * score_width
            top[index] = sampled_top
            left[index] = sampled_left
            used[index] = True

        return top.long(), left.long(), used

    def pachify(self, images, patch_size, padding=None):
        """AGG-mild single-patch path used by non-active patch sizes."""
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

        top, left, agg_used = self._sample_single_locations(padded, int(patch_size))
        patches = _extract_patches(padded, top, left, int(patch_size))
        positions = _position_maps(
            top,
            left,
            padded.shape[0],
            int(patch_size),
            resolution,
            padded.device,
        )
        training_stats.report('Sampler/agg_single_fraction', agg_used.float().mean())
        return patches, positions

    def _sample_overlap_locations(self, images, patch_size):
        """Sample horizontal/vertical half-shift pairs using AGG pair scores."""
        patch_size = int(patch_size)
        device = images.device
        batch = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        overlap = patch_size // 2
        stride = patch_size - overlap
        if overlap <= 0:
            raise ValueError('patch_size must be at least 2')

        max_top_h = height - patch_size
        max_left_h = width - patch_size - stride
        max_top_v = height - patch_size - stride
        max_left_v = width - patch_size
        if min(max_top_h, max_left_h, max_top_v, max_left_v) < 0:
            raise ValueError(
                'image resolution is too small for half-shift overlapping patches'
            )

        horizontal = torch.rand(batch, device=device) < 0.5
        top_h = torch.randint(0, max_top_h + 1, [batch], device=device)
        left_h = torch.randint(0, max_left_h + 1, [batch], device=device)
        top_v = torch.randint(0, max_top_v + 1, [batch], device=device)
        left_v = torch.randint(0, max_left_v + 1, [batch], device=device)
        top1 = torch.where(horizontal, top_h, top_v)
        left1 = torch.where(horizontal, left_h, left_v)

        requested = torch.rand(batch, device=device) < self.agg_sample_prob
        used = torch.zeros(batch, device=device, dtype=torch.bool)

        for index in range(batch):
            if not bool(requested[index].item()):
                continue
            score = self._agg_score_map(images[index], patch_size)
            if score is None:
                continue

            if bool(horizontal[index].item()):
                # Candidate pair: (top, left) and (top, left + stride).
                pair_score = 0.5 * (score[:, :-stride] + score[:, stride:])
            else:
                # Candidate pair: (top, left) and (top + stride, left).
                pair_score = 0.5 * (score[:-stride, :] + score[stride:, :])

            flat_index = torch.multinomial(pair_score.flatten(), 1)[0]
            pair_width = pair_score.shape[1]
            sampled_top = torch.div(flat_index, pair_width, rounding_mode='floor')
            sampled_left = flat_index - sampled_top * pair_width
            top1[index] = sampled_top
            left1[index] = sampled_left
            used[index] = True

        top2 = torch.where(horizontal, top1, top1 + stride)
        left2 = torch.where(horizontal, left1 + stride, left1)
        return (
            top1.long(),
            left1.long(),
            top2.long(),
            left2.long(),
            horizontal,
            overlap,
            used,
        )

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

        # 16x16 and 32x32 retain the standard EDM objective, but their patch
        # locations are sampled by this class's AGG-mild pachify().
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
            raise ValueError('AGG + independent overlap requires --augment=0')

        pair_batch = images.shape[0]
        if pair_batch < 1:
            raise ValueError('current batch contains no source image')

        (
            top1,
            left1,
            top2,
            left2,
            horizontal,
            overlap,
            agg_used,
        ) = self._sample_overlap_locations(images, patch_size)

        clean1 = _extract_patches(images, top1, left1, patch_size)
        clean2 = _extract_patches(images, top2, left2, patch_size)
        pos1 = _position_maps(
            top1, left1, pair_batch, patch_size, resolution, images.device
        )
        pos2 = _position_maps(
            top2, left2, pair_batch, patch_size, resolution, images.device
        )

        rnd_normal = torch.randn(
            [pair_batch, 1, 1, 1], device=images.device
        )
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (
            sigma.square() + self.sigma_data ** 2
        ) / (sigma * self.sigma_data).square()

        noisy1 = clean1 + sigma * torch.randn_like(clean1)
        noisy2 = clean2 + sigma * torch.randn_like(clean2)

        noisy_pair = torch.cat([noisy1, noisy2], dim=0)
        sigma_pair = torch.cat([sigma, sigma], dim=0)
        pos_pair = torch.cat([pos1, pos2], dim=0)
        labels_pair = (
            torch.cat([labels, labels], dim=0) if labels is not None else None
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

        edm_loss_1 = weight * (denoised1 - clean1).square()
        edm_loss_2 = weight * (denoised2 - clean2).square()
        edm_loss = 0.5 * (edm_loss_1 + edm_loss_2)

        training_stats.report('Loss/edm', edm_loss.detach())
        training_stats.report('Sampler/agg_pair_fraction', agg_used.float().mean())

        if self.lambda_overlap == 0:
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
        confidence_mean = confidence_raw.mean(
            dim=(1, 2, 3), keepdim=True
        )
        confidence = confidence_raw / confidence_mean.clamp_min(1e-12)

        auxiliary_pixel = (
            weight
            * confidence
            * (denoised_boundary - denoised_interior.detach()).square()
        )
        auxiliary_per_pair = auxiliary_pixel.mean(
            dim=(1, 2, 3), keepdim=True
        )
        auxiliary_loss = auxiliary_per_pair.expand_as(edm_loss)
        weighted_auxiliary = self.lambda_overlap * auxiliary_loss

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
            edm_loss.detach().new_tensor(self.lambda_overlap),
        )

        return edm_loss + weighted_auxiliary
