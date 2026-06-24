"""PaDIS-MRI 独立噪声、中心引导的重叠一致性损失。

本文件在 selective CG-OL 的基础上只增加一个可切换优化点：
gradient-aware overlap sampling。

模式：
    uniform:
        完全沿用原始随机 overlap pair 采样。
    gradient:
        根据当前 slice 的空间梯度图，对 overlap 区域高梯度的位置加权采样。
    mixed:
        以 overlap_gradient_alpha 的概率使用 gradient 采样，
        以 1 - overlap_gradient_alpha 的概率使用 uniform 采样。

除 overlap pair 的采样位置外，不改变：
    - EDM 主损失
    - same sigma / independent noise 设置
    - center-to-boundary teacher/student 方向
    - D_interior.detach()
    - active_patch_sizes 选择逻辑
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


def _gradient_magnitude_map(images):
    """Return a per-pixel gradient magnitude map with shape [B, 1, H, W].

    The input MRI image has multiple channels. We use mean absolute finite
    differences across channels, detach the map, and only use it to choose
    where to sample overlap pairs. It does not introduce gradients.
    """
    x = images.detach()
    b, _, h, w = x.shape

    grad_x = torch.zeros([b, 1, h, w], device=x.device, dtype=x.dtype)
    grad_y = torch.zeros([b, 1, h, w], device=x.device, dtype=x.dtype)

    grad_x[:, :, :, 1:] = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    grad_y[:, :, 1:, :] = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

    return grad_x + grad_y


def _sample_from_score(score, eps=1e-6):
    """Sample one flattened index per batch item from non-negative score."""
    b = score.shape[0]
    flat = score.reshape(b, -1).clamp_min(0)

    # If an image is nearly flat, fall back to almost uniform probabilities.
    scale = flat.detach().mean(dim=1, keepdim=True).clamp_min(eps)
    flat = flat + eps * scale

    return torch.multinomial(flat, num_samples=1).squeeze(1)


def _sample_overlap_geometry_gradient(images, patch_size, eps=1e-6):
    """Gradient-aware replacement for _sample_overlap_geometry.

    It keeps the same return format as the original helper:
        top1, left1, top2, left2, horizontal, overlap

    The only difference is that top1/left1 are sampled according to the
    average gradient magnitude in the physical overlap region.
    """
    device = images.device
    batch = images.shape[0]
    h, w = images.shape[2], images.shape[3]
    overlap = patch_size // 2
    stride = patch_size - overlap

    if overlap <= 0:
        raise ValueError('patch_size must be at least 2')

    horizontal = torch.rand([batch], device=device) < 0.5

    max_top_h = h - patch_size
    max_left_h = w - patch_size - stride
    max_top_v = h - patch_size - stride
    max_left_v = w - patch_size

    if min(max_top_h, max_left_h, max_top_v, max_left_v) < 0:
        raise ValueError('image resolution is too small for half-shift overlapping patches')

    grad = _gradient_magnitude_map(images)

    # Horizontal pair:
    # patch1: [top1:top1+p, left1:left1+p]
    # patch2: [top1:top1+p, left1+stride:left1+stride+p]
    # overlap physical region starts at (top1, left1+stride), shape [p, overlap].
    score_h_full = F.avg_pool2d(grad, kernel_size=(patch_size, overlap), stride=1)
    score_h = score_h_full[:, :, 0:max_top_h + 1, stride:stride + max_left_h + 1]
    idx_h = _sample_from_score(score_h, eps=eps)
    width_h = max_left_h + 1
    top_h = idx_h // width_h
    left_h = idx_h % width_h

    # Vertical pair:
    # overlap physical region starts at (top1+stride, left1), shape [overlap, p].
    score_v_full = F.avg_pool2d(grad, kernel_size=(overlap, patch_size), stride=1)
    score_v = score_v_full[:, :, stride:stride + max_top_v + 1, 0:max_left_v + 1]
    idx_v = _sample_from_score(score_v, eps=eps)
    width_v = max_left_v + 1
    top_v = idx_v // width_v
    left_v = idx_v % width_v

    top1 = torch.where(horizontal, top_h, top_v)
    left1 = torch.where(horizontal, left_h, left_v)
    top2 = torch.where(horizontal, top1, top1 + stride)
    left2 = torch.where(horizontal, left1 + stride, left1)

    return top1.long(), left1.long(), top2.long(), left2.long(), horizontal, overlap


def _sample_overlap_geometry_selectable(
    images,
    patch_size,
    sampling_mode='uniform',
    gradient_alpha=0.7,
    gradient_eps=1e-6,
):
    """Sample overlap geometry with a selectable strategy.

    Returns:
        top1, left1, top2, left2, horizontal, overlap, gradient_used

    gradient_used is a [B] float tensor used only for logging.
    """
    if sampling_mode not in ['uniform', 'gradient', 'mixed']:
        raise ValueError("sampling_mode must be 'uniform', 'gradient', or 'mixed'")
    if not (0.0 <= float(gradient_alpha) <= 1.0):
        raise ValueError('gradient_alpha must be in [0, 1]')

    batch = images.shape[0]
    device = images.device

    if sampling_mode == 'uniform':
        top1, left1, top2, left2, horizontal, overlap = _sample_overlap_geometry(images, int(patch_size))
        gradient_used = torch.zeros([batch], device=device, dtype=images.dtype)
        return top1, left1, top2, left2, horizontal, overlap, gradient_used

    if sampling_mode == 'gradient':
        top1, left1, top2, left2, horizontal, overlap = _sample_overlap_geometry_gradient(
            images, int(patch_size), eps=gradient_eps)
        gradient_used = torch.ones([batch], device=device, dtype=images.dtype)
        return top1, left1, top2, left2, horizontal, overlap, gradient_used

    # mixed: generate both candidates and choose per image.
    top1_u, left1_u, top2_u, left2_u, horizontal_u, overlap_u = _sample_overlap_geometry(images, int(patch_size))
    top1_g, left1_g, top2_g, left2_g, horizontal_g, overlap_g = _sample_overlap_geometry_gradient(
        images, int(patch_size), eps=gradient_eps)

    if overlap_u != overlap_g:
        raise RuntimeError('uniform and gradient samplers returned different overlap sizes')

    use_grad = torch.rand([batch], device=device) < float(gradient_alpha)

    top1 = torch.where(use_grad, top1_g, top1_u)
    left1 = torch.where(use_grad, left1_g, left1_u)
    top2 = torch.where(use_grad, top2_g, top2_u)
    left2 = torch.where(use_grad, left2_g, left2_u)
    horizontal = torch.where(use_grad, horizontal_g, horizontal_u)

    gradient_used = use_grad.to(images.dtype)
    return top1, left1, top2, left2, horizontal, overlap_u, gradient_used


@persistence.persistent_class
class IndependentNoiseOverlapPatchGradientSampling_EDMLoss(Patch_EDMLoss):
    """Selective CG-OL with optional gradient-aware overlap pair sampling."""

    def __init__(
        self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
        lambda_overlap=1.0, active_patch_size=64, active_patch_sizes=None,
        overlap_sampling_mode='uniform', overlap_gradient_alpha=0.7,
        overlap_gradient_eps=1e-6,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)

        if lambda_overlap < 0:
            raise ValueError('lambda_overlap 必须大于或等于 0')

        self.lambda_overlap = float(lambda_overlap)

        # 保留 active_patch_size 以兼容旧配置；新实验统一使用 active_patch_sizes。
        if active_patch_sizes is None:
            active_patch_sizes = [active_patch_size]
        elif isinstance(active_patch_sizes, str):
            active_patch_sizes = [int(x) for x in active_patch_sizes.split(',') if x.strip()]

        self.active_patch_sizes = tuple(sorted(set(int(x) for x in active_patch_sizes)))
        if len(self.active_patch_sizes) == 0:
            raise ValueError('active_patch_sizes 不能为空')

        if overlap_sampling_mode not in ['uniform', 'gradient', 'mixed']:
            raise ValueError("overlap_sampling_mode 必须是 'uniform'、'gradient' 或 'mixed'")
        if not (0.0 <= float(overlap_gradient_alpha) <= 1.0):
            raise ValueError('overlap_gradient_alpha 必须在 [0, 1] 内')
        if float(overlap_gradient_eps) <= 0:
            raise ValueError('overlap_gradient_eps 必须大于 0')

        self.overlap_sampling_mode = str(overlap_sampling_mode)
        self.overlap_gradient_alpha = float(overlap_gradient_alpha)
        self.overlap_gradient_eps = float(overlap_gradient_eps)

    def __call__(
        self, net, images, patch_size, resolution,
        labels=None, augment_pipe=None,
    ):
        # 未列入 active_patch_sizes 的尺度完全沿用作者原始 Patch_EDMLoss。
        if int(patch_size) not in self.active_patch_sizes:
            return super().__call__(
                net=net, images=images, patch_size=patch_size,
                resolution=resolution, labels=labels,
                augment_pipe=augment_pipe,
            )

        # 当前正式训练使用 --augment=0。若启用随机几何增强，
        # 两个 patch 的物理重叠对应关系可能被破坏，因此直接报错。
        if augment_pipe is not None:
            raise ValueError('Independent overlap 实验要求训练参数 --augment=0')

        # 使用本轮加载的全部源图像，不再随机丢弃一半样本。
        # 显存通过 --batch-gpu=1 和梯度累积控制。
        pair_batch = images.shape[0]
        clean_full = images
        labels_pair_source = labels

        if pair_batch < 1:
            raise ValueError('当前 batch 中没有可用图像')

        # 每张完整图像裁取一对半 patch 平移的重叠 patch。
        # uniform 为旧版随机采样；gradient/mixed 会提高高梯度 overlap 区域的采样概率。
        top1, left1, top2, left2, horizontal, overlap, gradient_used = _sample_overlap_geometry_selectable(
            clean_full,
            int(patch_size),
            sampling_mode=self.overlap_sampling_mode,
            gradient_alpha=self.overlap_gradient_alpha,
            gradient_eps=self.overlap_gradient_eps,
        )

        clean1 = _extract_patches(clean_full, top1, left1, int(patch_size))
        clean2 = _extract_patches(clean_full, top2, left2, int(patch_size))

        pos1 = _position_maps(top1, left1, pair_batch, int(patch_size), resolution, images.device)
        pos2 = _position_maps(top2, left2, pair_batch, int(patch_size), resolution, images.device)

        # 两个 patch 共用同一噪声水平 sigma，只让噪声实现相互独立。
        # 这样主要隔离 patch context 和随机噪声实现带来的预测差异。
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
        # 将两个分支拼成一次网络前向，随后再按 pair_batch 拆开。
        denoised_pair = net(noisy_pair, sigma_pair, x_pos=pos_pair, class_labels=labels_pair, augment_labels=None)
        D1 = denoised_pair[:pair_batch]
        D2 = denoised_pair[pair_batch:]

        # 两个视角都保留原始 clean-target EDM 监督，并对两者取平均。
        # 因此每张源图像的主损失尺度与 baseline 保持接近。
        edm_loss_1 = weight * (D1 - clean1).square()
        edm_loss_2 = weight * (D2 - clean2).square()
        edm_loss = 0.5 * (edm_loss_1 + edm_loss_2)
        training_stats.report('Loss/edm', edm_loss.detach())
        training_stats.report('Loss/overlap_gradient_sample_fraction', gradient_used.detach().mean())

        # lambda=0 即 paired-sampling control：双 patch 前向和主损失都保留，
        # 只关闭辅助一致性梯度，因此可与 lambda>0 版本公平比较。
        if self.lambda_overlap == 0:
            zero = edm_loss.detach().new_zeros([])
            training_stats.report('Loss/overlap_center_raw', zero)
            training_stats.report('Loss/overlap_center_raw_ratio', zero)
            training_stats.report('Loss/overlap_center_weighted', zero)
            training_stats.report('Loss/overlap_center_ratio', zero)
            training_stats.report('Loss/overlap_gradient_alpha', edm_loss.detach().new_tensor(self.overlap_gradient_alpha))
            return edm_loss

        # 提取两个预测中对应同一物理位置的 overlap 区域。
        D1_overlap, D2_overlap = _overlap_views(D1, D2, horizontal, overlap)

        # d1、d2 表示同一物理像素在两个 patch 中到最近边界的距离。
        # 距离更大的视角被视为内部预测，另一个被视为边界预测。
        d1, d2 = _boundary_distances(horizontal, int(patch_size), overlap, images.device, images.dtype)
        interior_is_1 = d1 >= d2
        D_interior = torch.where(interior_is_1, D1_overlap, D2_overlap)
        D_boundary = torch.where(interior_is_1, D2_overlap, D1_overlap)

        # 保持原始线性几何置信度，不在本文件里额外引入 confidence power。
        # 本实验只改变 overlap pair 的采样位置。
        confidence_raw = (d1 - d2).abs()
        confidence_mean = confidence_raw.mean(dim=(1, 2, 3), keepdim=True)
        confidence = confidence_raw / confidence_mean.clamp_min(1e-12)

        # 仅更新边界预测。D_interior.detach() 作为固定参考，
        # 避免较不可靠的边界预测反向拖动内部预测。
        aux_pixel = weight * confidence * (D_boundary - D_interior.detach()).square()

        # 每个 pair 先在通道和 overlap 空间上求平均，再扩展为 loss map。training_loop 会对返回张量执行 loss.sum()，
        # 这种写法使 ratio 可直接表示辅助项相对于平均 EDM loss 的实际强度。
        aux_per_pair = aux_pixel.mean(dim=(1, 2, 3), keepdim=True)
        auxiliary_loss = aux_per_pair.expand_as(edm_loss)
        weighted_auxiliary = self.lambda_overlap * auxiliary_loss

        edm_mean = edm_loss.detach().mean().clamp_min(1e-12)
        raw_ratio = auxiliary_loss.detach().mean() / edm_mean
        weighted_ratio = weighted_auxiliary.detach().mean() / edm_mean

        training_stats.report('Loss/overlap_center_raw', aux_per_pair.detach())
        training_stats.report('Loss/overlap_center_raw_ratio', raw_ratio)
        training_stats.report(
            'Loss/overlap_center_weighted', weighted_auxiliary.detach()
        )
        training_stats.report('Loss/overlap_center_ratio', weighted_ratio)
        training_stats.report(
            'Loss/overlap_center_lambda',
            edm_loss.detach().new_tensor(self.lambda_overlap),
        )
        training_stats.report('Loss/overlap_gradient_alpha', edm_loss.detach().new_tensor(self.overlap_gradient_alpha))

        return edm_loss + weighted_auxiliary
