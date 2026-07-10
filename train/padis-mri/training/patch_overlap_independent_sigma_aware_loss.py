"""PaDIS-MRI 独立噪声、中心引导的重叠一致性损失。

实验 1：lambda_overlap=0
    Paired-sampling control。仍使用全部源图像和双 patch 前向，
    但不加入重叠一致性梯度。

实验 2：lambda_overlap>0
    两个重叠 patch 使用相同 sigma、独立高斯噪声；
    更靠近 patch 边界的预测向更靠近 patch 内部的预测学习。

仅在 active_patch_size（默认 64）上启用上述配对训练。
16×16 和 32×32 分支保持作者原始 Patch_EDMLoss 不变。

本版本增加噪声感知 overlap 权重：
    fixed:
        lambda(sigma) = lambda_overlap
    sigmoid:
        lambda(sigma) = lambda_overlap * sigmoid(
            overlap_sigma_slope * (log(overlap_sigma_center) - log(sigma))
        )
    piecewise:
        在 log-sigma 域中，从 sigma_low 处的 lambda_overlap 线性下降到
        sigma_high 处的 0；区间外分别截断为 lambda_overlap 和 0。

权重按样本计算。同一对 overlap patch 仍共用同一个 sigma，且不修改
EDM 主损失的原始噪声权重。
"""

import math
import os
import sys

import torch

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
class IndependentNoiseOverlapPatch_EDMLoss(Patch_EDMLoss):
    """仅在指定 patch 尺度上启用独立噪声、中心引导的一致性约束。"""

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        lambda_overlap=1.0,
        active_patch_size=64,
        overlap_weight_mode='fixed',
        overlap_sigma_center=0.2,
        overlap_sigma_slope=2.0,
        overlap_sigma_low=0.1,
        overlap_sigma_high=0.5,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)

        if lambda_overlap < 0:
            raise ValueError('lambda_overlap 必须大于或等于 0')
        if overlap_weight_mode not in {'fixed', 'sigmoid', 'piecewise'}:
            raise ValueError(
                'overlap_weight_mode 必须是 fixed、sigmoid 或 piecewise'
            )
        if overlap_sigma_center <= 0:
            raise ValueError('overlap_sigma_center 必须大于 0')
        if overlap_sigma_slope <= 0:
            raise ValueError('overlap_sigma_slope 必须大于 0')
        if overlap_sigma_low <= 0:
            raise ValueError('overlap_sigma_low 必须大于 0')
        if overlap_sigma_high <= overlap_sigma_low:
            raise ValueError(
                'overlap_sigma_high 必须严格大于 overlap_sigma_low'
            )

        self.lambda_overlap = float(lambda_overlap)
        self.active_patch_size = int(active_patch_size)
        self.overlap_weight_mode = str(overlap_weight_mode)
        self.overlap_sigma_center = float(overlap_sigma_center)
        self.overlap_sigma_slope = float(overlap_sigma_slope)
        self.overlap_sigma_low = float(overlap_sigma_low)
        self.overlap_sigma_high = float(overlap_sigma_high)

    def _overlap_weight_per_sample(self, sigma):
        """根据每个样本的 sigma 返回形状为 [B] 的 overlap 权重。"""
        sigma_per_sample = sigma.reshape(-1).clamp_min(1e-12)

        if self.overlap_weight_mode == 'fixed':
            return torch.full_like(sigma_per_sample, self.lambda_overlap)

        if self.overlap_weight_mode == 'sigmoid':
            log_center = math.log(self.overlap_sigma_center)
            logits = self.overlap_sigma_slope * (
                log_center - sigma_per_sample.log()
            )
            return self.lambda_overlap * torch.sigmoid(logits)

        # piecewise：在 log-sigma 域内线性变化。
        log_low = math.log(self.overlap_sigma_low)
        log_high = math.log(self.overlap_sigma_high)
        interpolation = (
            log_high - sigma_per_sample.log()
        ) / (log_high - log_low)
        return self.lambda_overlap * interpolation.clamp(0.0, 1.0)

    @staticmethod
    def _masked_mean(values, mask):
        """计算分箱均值；空分箱返回 0，且避免 CUDA 到 CPU 同步。"""
        mask_float = mask.to(values.dtype)
        numerator = (values * mask_float).sum()
        denominator = mask_float.sum().clamp_min(1.0)
        return numerator / denominator

    def _report_sigma_binned_losses(
        self,
        sigma_per_sample,
        raw_overlap_per_sample,
        weighted_overlap_per_sample,
    ):
        """按低、中、高 sigma 分箱记录原始和加权 overlap loss。"""
        low_mask = sigma_per_sample <= self.overlap_sigma_low
        middle_mask = (
            (sigma_per_sample > self.overlap_sigma_low)
            & (sigma_per_sample < self.overlap_sigma_high)
        )
        high_mask = sigma_per_sample >= self.overlap_sigma_high

        bins = {
            'low': low_mask,
            'middle': middle_mask,
            'high': high_mask,
        }
        for bin_name, mask in bins.items():
            training_stats.report(
                f'Loss/overlap_raw_sigma_{bin_name}',
                self._masked_mean(raw_overlap_per_sample, mask),
            )
            training_stats.report(
                f'Loss/overlap_weighted_sigma_{bin_name}',
                self._masked_mean(weighted_overlap_per_sample, mask),
            )
            training_stats.report(
                f'Loss/sigma_bin_fraction_{bin_name}',
                mask.to(torch.float32).mean(),
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
        # 16×16 和 32×32 分支完全沿用作者原始 Patch_EDMLoss。
        if int(patch_size) != self.active_patch_size:
            return super().__call__(
                net=net,
                images=images,
                patch_size=patch_size,
                resolution=resolution,
                labels=labels,
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

        # 每张完整图像裁取一对水平或竖直半 patch 平移的重叠 patch。
        top1, left1, top2, left2, horizontal, overlap = _sample_overlap_geometry(
            clean_full, int(patch_size)
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

        # 两个 patch 共用同一噪声水平 sigma，只让噪声实现相互独立。
        # sigma 仍然按源图像逐样本采样，形状为 [B, 1, 1, 1]。
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
            torch.cat([labels_pair_source, labels_pair_source], dim=0)
            if labels_pair_source is not None
            else None
        )

        # 将两个分支拼成一次网络前向，随后再按 pair_batch 拆开。
        denoised_pair = net(
            noisy_pair,
            sigma_pair,
            x_pos=pos_pair,
            class_labels=labels_pair,
            augment_labels=None,
        )
        D1 = denoised_pair[:pair_batch]
        D2 = denoised_pair[pair_batch:]

        # 两个视角都保留原始 clean-target EDM 监督，并对两者取平均。
        # 不修改 EDM 原始权重，使主损失与旧 CG-OL 保持一致。
        edm_loss_1 = weight * (D1 - clean1).square()
        edm_loss_2 = weight * (D2 - clean2).square()
        edm_loss = 0.5 * (edm_loss_1 + edm_loss_2)
        edm_per_sample = edm_loss.mean(dim=(1, 2, 3))
        training_stats.report('Loss/edm', edm_loss.detach())

        sigma_per_sample = sigma.reshape(-1)
        lambda_per_sample = self._overlap_weight_per_sample(sigma)

        training_stats.report('Loss/sigma_mean', sigma_per_sample.detach().mean())
        training_stats.report(
            'Loss/overlap_weight_mean', lambda_per_sample.detach().mean()
        )
        training_stats.report(
            'Loss/overlap_weight_min', lambda_per_sample.detach().min()
        )
        training_stats.report(
            'Loss/overlap_weight_max', lambda_per_sample.detach().max()
        )

        # lambda=0 即 paired-sampling control：双 patch 前向和主损失都保留，
        # 只关闭辅助一致性梯度，因此可与 lambda>0 版本公平比较。
        if self.lambda_overlap == 0:
            zero = edm_loss.detach().new_zeros([])
            training_stats.report('Loss/overlap_center_raw', zero)
            training_stats.report('Loss/overlap_center_raw_ratio', zero)
            training_stats.report('Loss/overlap_center_weighted', zero)
            training_stats.report('Loss/overlap_center_ratio', zero)
            training_stats.report('Loss/overlap_center_lambda', zero)
            training_stats.report('Loss/overlap_loss_raw', zero)
            training_stats.report('Loss/overlap_loss_weighted', zero)
            self._report_sigma_binned_losses(
                sigma_per_sample=sigma_per_sample.detach(),
                raw_overlap_per_sample=torch.zeros_like(
                    sigma_per_sample.detach()
                ),
                weighted_overlap_per_sample=torch.zeros_like(
                    sigma_per_sample.detach()
                ),
            )
            return edm_loss

        # 提取两个预测中对应同一物理位置的 overlap 区域。
        D1_overlap, D2_overlap = _overlap_views(
            D1, D2, horizontal, overlap
        )

        # d1、d2 表示同一物理像素在两个 patch 中到最近边界的距离。
        # 距离更大的视角被视为内部预测，另一个被视为边界预测。
        d1, d2 = _boundary_distances(
            horizontal,
            int(patch_size),
            overlap,
            images.device,
            images.dtype,
        )
        interior_is_1 = d1 >= d2
        D_interior = torch.where(
            interior_is_1, D1_overlap, D2_overlap
        )
        D_boundary = torch.where(
            interior_is_1, D2_overlap, D1_overlap
        )

        # 距离差越大，内部/边界差异越明确，辅助权重越大。
        # 每个 pair 内归一化，使 confidence 的均值约为 1。
        confidence_raw = (d1 - d2).abs()
        confidence_mean = confidence_raw.mean(
            dim=(1, 2, 3), keepdim=True
        )
        confidence = confidence_raw / confidence_mean.clamp_min(1e-12)

        # 仅更新边界预测。D_interior.detach() 作为固定参考，
        # 避免较不可靠的边界预测反向拖动内部预测。
        aux_pixel = (
            weight
            * confidence
            * (D_boundary - D_interior.detach()).square()
        )

        # 每个 pair 先在通道和 overlap 空间上求平均，再扩展为 loss map。
        # training_loop 对返回张量执行 loss.sum()，保留该写法可维持
        # 原 CG-OL 的整体损失尺度和梯度归一化方式。
        aux_per_pair = aux_pixel.mean(dim=(1, 2, 3), keepdim=True)
        auxiliary_loss = aux_per_pair.expand_as(edm_loss)

        # 核心修改：每个样本使用自己的 lambda(sigma)，禁止先求 batch 均值。
        lambda_map = lambda_per_sample.view(-1, 1, 1, 1)
        weighted_auxiliary = lambda_map * auxiliary_loss

        raw_overlap_per_sample = aux_per_pair.reshape(-1)
        weighted_overlap_per_sample = (
            lambda_per_sample * raw_overlap_per_sample
        )

        edm_mean = edm_per_sample.detach().mean().clamp_min(1e-12)
        raw_ratio = raw_overlap_per_sample.detach().mean() / edm_mean
        weighted_ratio = (
            weighted_overlap_per_sample.detach().mean() / edm_mean
        )

        # 保留旧日志键，便于与固定权重 CG-OL 的历史曲线直接比较。
        training_stats.report(
            'Loss/overlap_center_raw', raw_overlap_per_sample.detach()
        )
        training_stats.report(
            'Loss/overlap_center_raw_ratio', raw_ratio
        )
        training_stats.report(
            'Loss/overlap_center_weighted',
            weighted_overlap_per_sample.detach(),
        )
        training_stats.report(
            'Loss/overlap_center_ratio', weighted_ratio
        )
        training_stats.report(
            'Loss/overlap_center_lambda',
            lambda_per_sample.detach(),
        )

        # 新增的统一日志键。
        training_stats.report(
            'Loss/overlap_loss_raw', raw_overlap_per_sample.detach()
        )
        training_stats.report(
            'Loss/overlap_loss_weighted',
            weighted_overlap_per_sample.detach(),
        )
        self._report_sigma_binned_losses(
            sigma_per_sample=sigma_per_sample.detach(),
            raw_overlap_per_sample=raw_overlap_per_sample.detach(),
            weighted_overlap_per_sample=weighted_overlap_per_sample.detach(),
        )

        return edm_loss + weighted_auxiliary
