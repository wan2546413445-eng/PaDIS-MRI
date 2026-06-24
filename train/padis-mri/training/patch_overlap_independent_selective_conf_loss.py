"""PaDIS-MRI 独立噪声、中心引导的重叠一致性损失。

实验 1：lambda_overlap=0
    Paired-sampling control。仍使用全部源图像和双 patch 前向，
    但不加入重叠一致性梯度。

实验 2：lambda_overlap>0
    两个重叠 patch 使用相同 sigma、独立高斯噪声；
    更靠近 patch 边界的预测向更靠近 patch 内部的预测学习。

仅在 active_patch_sizes 指定的尺度上启用上述配对训练。
未指定的 patch 尺度保持作者原始 Patch_EDMLoss 不变。

可选优化：几何置信度支持 linear / power 两种模式。
默认 linear + gamma=1.0 与原始实现等价；power + gamma>1 会更集中地强调
内部/边界可靠性差异更明显的 overlap 像素，同时重新归一化均值以保持
lambda_overlap 的整体尺度可比。
"""

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
class IndependentNoiseOverlapPatchSelective_EDMLoss(Patch_EDMLoss):
    """仅在指定 patch 尺度集合上启用独立噪声、中心引导的一致性约束。"""

    def __init__(
        self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
        lambda_overlap=1.0, active_patch_size=64, active_patch_sizes=None,
        confidence_mode='linear', confidence_gamma=1.0,
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

        # 几何置信度模式。linear/gamma=1.0 完全保持原始线性权重；
        # power/gamma>1 只改变 overlap 区域内部的空间权重分布，
        # 并在后面重新归一化到均值约为 1，避免改变 lambda 的整体量级。
        if confidence_mode not in ['linear', 'power']:
            raise ValueError("confidence_mode 必须是 'linear' 或 'power'")
        if confidence_gamma <= 0:
            raise ValueError('confidence_gamma 必须大于 0')
        self.confidence_mode = str(confidence_mode)
        self.confidence_gamma = float(confidence_gamma)

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

        # 每张完整图像裁取一对水平或竖直半 patch 平移的重叠 patch。
        top1, left1, top2, left2, horizontal, overlap = _sample_overlap_geometry(
            clean_full, int(patch_size))

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
        denoised_pair = net(noisy_pair, sigma_pair, x_pos=pos_pair,class_labels=labels_pair, augment_labels=None,)
        D1 = denoised_pair[:pair_batch]
        D2 = denoised_pair[pair_batch:]
        # 两个视角都保留原始 clean-target EDM 监督，并对两者取平均。
        # 因此每张源图像的主损失尺度与 baseline 保持接近。
        edm_loss_1 = weight * (D1 - clean1).square()
        edm_loss_2 = weight * (D2 - clean2).square()
        edm_loss = 0.5 * (edm_loss_1 + edm_loss_2)
        training_stats.report('Loss/edm', edm_loss.detach())
        # lambda=0 即 paired-sampling control：双 patch 前向和主损失都保留，
        # 只关闭辅助一致性梯度，因此可与 lambda>0 版本公平比较。
        if self.lambda_overlap == 0:
            zero = edm_loss.detach().new_zeros([])
            training_stats.report('Loss/overlap_center_raw', zero)
            training_stats.report('Loss/overlap_center_raw_ratio', zero)
            training_stats.report('Loss/overlap_center_weighted', zero)
            training_stats.report('Loss/overlap_center_ratio', zero)
            return edm_loss
        # 提取两个预测中对应同一物理位置的 overlap 区域。
        D1_overlap, D2_overlap = _overlap_views(D1, D2, horizontal, overlap)
        # d1、d2 表示同一物理像素在两个 patch 中到最近边界的距离。
        # 距离更大的视角被视为内部预测，另一个被视为边界预测。
        d1, d2 = _boundary_distances(horizontal, int(patch_size), overlap, images.device, images.dtype)
        interior_is_1 = d1 >= d2
        D_interior = torch.where(interior_is_1, D1_overlap, D2_overlap)
        D_boundary = torch.where(interior_is_1, D2_overlap, D1_overlap)
        # 距离差越大，内部/边界差异越明确，辅助权重越大。按每个 pair 的平均值归一化，使 confidence 的平均值约为 1，
        # 避免空间权重把辅助损失整体再次压小。
        confidence_raw = (d1 - d2).abs()
        confidence_mean = confidence_raw.mean(dim=(1, 2, 3), keepdim=True)
        confidence = confidence_raw / confidence_mean.clamp_min(1e-12)

        if self.confidence_mode == 'power':
            confidence = confidence.clamp_min(1e-12).pow(self.confidence_gamma)
            # power 会改变整体均值；重新归一化，保证 lambda_overlap 的尺度与 linear 版本可比。
            confidence = confidence / confidence.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        # linear 模式直接使用原始实现，gamma 不参与计算。

        # 仅更新边界预测。D_interior.detach() 作为固定参考，
        # 避免较不可靠的边界预测反向拖动内部预测。
        aux_pixel = weight * confidence * (D_boundary - D_interior.detach()).square()

        # 每个 pair 先在通道和 overlap 空间上求平均，再扩展为 loss map。training_loop 会对返回张量执行 loss.sum()，这种写法使 ratio
        # 可直接表示辅助项相对于平均 EDM loss 的实际强度。
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
        training_stats.report('Loss/overlap_confidence_gamma', edm_loss.detach().new_tensor(self.confidence_gamma))
        training_stats.report(
            'Loss/overlap_center_lambda',
            edm_loss.detach().new_tensor(self.lambda_overlap),
        )

        return edm_loss + weighted_auxiliary