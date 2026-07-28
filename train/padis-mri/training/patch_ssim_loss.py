"""Baseline Patch-EDM loss with a low-noise magnitude-SSIM auxiliary term."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from torch_utils import persistence, training_stats
from training.patch_loss import Patch_EDMLoss


def _validate_ssim_parameters(
    ssim_weight,
    ssim_sigma_max,
    ssim_window_size,
    ssim_window_sigma,
    ssim_k1,
    ssim_k2,
    ssim_data_range,
    magnitude_eps,
):
    """Validate the fixed stage-1 SSIM hyperparameters."""
    if ssim_weight < 0:
        raise ValueError('ssim_weight must be >= 0')
    if ssim_sigma_max <= 0:
        raise ValueError('ssim_sigma_max must be > 0')
    if (
        not isinstance(ssim_window_size, int)
        or isinstance(ssim_window_size, bool)
        or ssim_window_size <= 0
        or ssim_window_size % 2 == 0
    ):
        raise ValueError('ssim_window_size must be a positive odd integer')
    if ssim_window_size > 16:
        raise ValueError('ssim_window_size must be <= 16')
    if ssim_window_sigma <= 0:
        raise ValueError('ssim_window_sigma must be > 0')
    if ssim_k1 <= 0:
        raise ValueError('ssim_k1 must be > 0')
    if ssim_k2 <= 0:
        raise ValueError('ssim_k2 must be > 0')
    if ssim_data_range <= 0:
        raise ValueError('ssim_data_range must be > 0')
    if magnitude_eps <= 0:
        raise ValueError('magnitude_eps must be > 0')


def _gaussian_window(window_size, window_sigma, device, dtype):
    """Construct a normalized 2-D Gaussian window on the current device."""
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel_1d = torch.exp(-coordinates.square() / (2 * window_sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d).reshape(1, 1, window_size, window_size)


def magnitude_from_two_channels(value, magnitude_eps=1e-8):
    """Convert real/imaginary channels [B,2,H,W] to float32 magnitude."""
    if value.ndim != 4 or value.shape[1] != 2:
        raise ValueError(
            'magnitude SSIM expects a [B,2,H,W] real/imaginary tensor'
        )
    value = value.float()
    return torch.sqrt(
        value[:, 0:1].square()
        + value[:, 1:2].square()
        + float(magnitude_eps)
    )


def gaussian_ssim_2d(
    prediction,
    target,
    window_size=11,
    window_sigma=1.5,
    k1=0.01,
    k2=0.03,
    data_range=1.0,
):
    """Return one differentiable, single-scale SSIM value per sample."""
    if prediction.shape != target.shape:
        raise ValueError('prediction and target must have identical shapes')
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError('SSIM expects single-channel [B,1,H,W] tensors')
    if min(prediction.shape[-2:]) <= window_size // 2:
        raise ValueError('input is too small for reflect padding')

    prediction = prediction.float()
    target = target.float()
    window = _gaussian_window(
        window_size,
        window_sigma,
        prediction.device,
        prediction.dtype,
    )
    padding = window_size // 2

    def local_mean(value):
        value = F.pad(
            value,
            (padding, padding, padding, padding),
            mode='reflect',
        )
        return F.conv2d(value, window)

    mu_prediction = local_mean(prediction)
    mu_target = local_mean(target)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target

    variance_prediction = (
        local_mean(prediction.square()) - mu_prediction_sq
    ).clamp_min(0.0)
    variance_target = (
        local_mean(target.square()) - mu_target_sq
    ).clamp_min(0.0)
    covariance = local_mean(prediction * target) - mu_product

    c1 = float(k1 * data_range) ** 2
    c2 = float(k2 * data_range) ** 2
    numerator = (
        (2.0 * mu_product + c1)
        * (2.0 * covariance + c2)
    )
    denominator = (
        (mu_prediction_sq + mu_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return ssim_map.mean(dim=(1, 2, 3), keepdim=True)


@persistence.persistent_class
class MagnitudeSSIMPatchEDMLoss(Patch_EDMLoss):
    """Original Patch-EDM objective plus gated magnitude SSIM."""

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        ssim_weight=0.50,
        ssim_sigma_max=0.50,
        ssim_window_size=11,
        ssim_window_sigma=1.5,
        ssim_k1=0.01,
        ssim_k2=0.03,
        ssim_data_range=1.0,
        magnitude_eps=1e-8,
    ):
        super().__init__(
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=sigma_data,
        )
        _validate_ssim_parameters(
            ssim_weight=ssim_weight,
            ssim_sigma_max=ssim_sigma_max,
            ssim_window_size=ssim_window_size,
            ssim_window_sigma=ssim_window_sigma,
            ssim_k1=ssim_k1,
            ssim_k2=ssim_k2,
            ssim_data_range=ssim_data_range,
            magnitude_eps=magnitude_eps,
        )
        self.ssim_weight = float(ssim_weight)
        self.ssim_sigma_max = float(ssim_sigma_max)
        self.ssim_window_size = int(ssim_window_size)
        self.ssim_window_sigma = float(ssim_window_sigma)
        self.ssim_k1 = float(ssim_k1)
        self.ssim_k2 = float(ssim_k2)
        self.ssim_data_range = float(ssim_data_range)
        self.magnitude_eps = float(magnitude_eps)

    def _ssim_loss(self, prediction, target):
        prediction_magnitude = magnitude_from_two_channels(
            prediction,
            self.magnitude_eps,
        )
        target_magnitude = magnitude_from_two_channels(
            target,
            self.magnitude_eps,
        )
        ssim_value = gaussian_ssim_2d(
            prediction_magnitude,
            target_magnitude,
            window_size=self.ssim_window_size,
            window_sigma=self.ssim_window_sigma,
            k1=self.ssim_k1,
            k2=self.ssim_k2,
            data_range=self.ssim_data_range,
        )
        return ssim_value, torch.clamp_min(1.0 - ssim_value, 0.0)

    def __call__(
        self,
        net,
        images,
        patch_size,
        resolution,
        labels=None,
        augment_pipe=None,
    ):
        # 关闭辅助项时直接进入原始实现，保持随机数、前向和梯度严格一致。
        if self.ssim_weight == 0:
            return super().__call__(
                net=net,
                images=images,
                patch_size=patch_size,
                resolution=resolution,
                labels=labels,
                augment_pipe=augment_pipe,
            )

        # 以下随机操作顺序与原始 Patch_EDMLoss 完全一致。
        images, images_pos = self.pachify(images, patch_size)
        rnd_normal = torch.randn(
            [images.shape[0], 1, 1, 1],
            device=images.device,
        )
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (
            sigma.square() + self.sigma_data ** 2
        ) / (sigma * self.sigma_data).square()

        y, augment_labels = (
            augment_pipe(images)
            if augment_pipe is not None
            else (images, None)
        )
        noise = torch.randn_like(y) * sigma
        yn = y + noise
        D_yn = net(
            yn,
            sigma,
            x_pos=images_pos,
            class_labels=labels,
            augment_labels=augment_labels,
        )

        if D_yn.ndim != 4 or D_yn.shape[1] != 2:
            raise ValueError(
                'MagnitudeSSIMPatchEDMLoss requires two output channels'
            )
        if y.ndim != 4 or y.shape[1] != 2:
            raise ValueError(
                'MagnitudeSSIMPatchEDMLoss requires two target channels'
            )

        edm_loss = weight * (D_yn - y).square()
        ssim_value, ssim_loss = self._ssim_loss(D_yn, y)
        active_mask = (
            sigma.reshape(images.shape[0], 1, 1, 1)
            <= self.ssim_sigma_max
        ).float()
        gated_ssim_loss = active_mask * ssim_loss
        ssim_expanded = gated_ssim_loss.expand_as(edm_loss)
        weighted_ssim = self.ssim_weight * ssim_expanded

        edm_mean = edm_loss.detach().mean().clamp_min(1e-12)
        weighted_mean = weighted_ssim.detach().mean()
        active_count = active_mask.detach().sum()
        sigma_active_mean = (
            (sigma.detach() * active_mask.detach()).sum()
            / active_count.clamp_min(1.0)
        )

        training_stats.report('Loss/edm', edm_loss.detach())
        training_stats.report('Loss/ssim_raw', gated_ssim_loss.detach())
        training_stats.report('Loss/ssim_weighted', weighted_ssim.detach())
        training_stats.report(
            'Loss/ssim_weighted_ratio',
            weighted_mean / edm_mean,
        )
        active_values = ssim_value.detach().reshape(-1)[
            active_mask.detach().reshape(-1).bool()
        ]
        training_stats.report(
            'SSIM/prediction_to_target',
            active_values,
        )
        training_stats.report(
            'SSIM/prediction_to_target_all',
            ssim_value.detach(),
        )
        training_stats.report(
            'SSIM/active_fraction',
            active_mask.detach().mean(),
        )
        training_stats.report('SSIM/sigma_active_mean', sigma_active_mean)
        training_stats.report(
            'SSIM/weight',
            edm_loss.detach().new_tensor(self.ssim_weight),
        )
        training_stats.report(
            'SSIM/sigma_max',
            edm_loss.detach().new_tensor(self.ssim_sigma_max),
        )
        for supported_size in (16, 32, 64):
            values = (
                active_values
                if int(patch_size) == supported_size
                else []
            )
            training_stats.report(f'SSIM/patch{supported_size}', values)

        return edm_loss + weighted_ssim
