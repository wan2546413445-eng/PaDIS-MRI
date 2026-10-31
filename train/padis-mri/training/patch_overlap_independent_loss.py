"""PaDIS-MRI 独立噪声、中心引导的重叠一致性损失。

实验 1：lambda_overlap=0
    Paired-sampling control。仍使用全部源图像和双 patch 前向，
    但不加入重叠一致性梯度。

实验 2：lambda_overlap>0
    两个重叠 patch 使用相同 sigma、独立高斯噪声；
    更靠近 patch 边界的预测向更靠近 patch 内部的预测学习。

alignment_mode='directional' 保留内部引导边界；'symmetric' 对两侧施加辅助梯度。
仅在 active_patch_size（默认 64）上启用上述配对训练。
16×16 和 32×32 分支保持作者原始 Patch_EDMLoss 不变。
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
class IndependentNoiseOverlapPatch_EDMLoss(Patch_EDMLoss):
    """仅在指定 patch 尺度上启用独立噪声、中心引导的一致性约束。"""

    def __init__(
            self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
            lambda_overlap=1.0, active_patch_size=64,
            noise_mode='independent', alignment_mode='directional',
            pair_geometry='shifted', spatial_weight_mode='distance',
    ):
        super().__init__(
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=sigma_data,
        )

        if lambda_overlap < 0:
            raise ValueError('lambda_overlap must be >= 0')

        if noise_mode not in ('same', 'independent'):
            raise ValueError(f'Invalid noise_mode: {noise_mode}')

        if alignment_mode not in ('directional', 'symmetric'):
            raise ValueError(f'Invalid alignment_mode: {alignment_mode}')

        if pair_geometry not in ('shifted', 'identical'):
            raise ValueError(f'Invalid pair_geometry: {pair_geometry}')

        if spatial_weight_mode not in ('distance', 'uniform'):
            raise ValueError(
                f'Invalid spatial_weight_mode: {spatial_weight_mode}'
            )

        if pair_geometry == 'identical':
            if spatial_weight_mode != 'uniform':
                raise ValueError(
                    'identical geometry requires spatial_weight_mode=uniform'
                )
            if lambda_overlap > 0 and alignment_mode != 'symmetric':
                raise ValueError(
                    'identical geometry with lambda_overlap>0 '
                    'requires alignment_mode=symmetric'
                )

        self.alignment_mode = alignment_mode
        self.lambda_overlap = float(lambda_overlap)
        self.active_patch_size = int(active_patch_size)
        self.noise_mode = noise_mode
        self.pair_geometry = pair_geometry
        self.spatial_weight_mode = spatial_weight_mode

        print(
            f'[OC] noise_mode={self.noise_mode}, '
            f'alignment_mode={self.alignment_mode}, '
            f'lambda={self.lambda_overlap}, '
            f'pair_geometry={self.pair_geometry}, '
            f'spatial_weight_mode={self.spatial_weight_mode}'
        )
    def __call__(
        self, net, images, patch_size, resolution,
        labels=None, augment_pipe=None,
    ):
        # 16×16 和 32×32 分支完全沿用作者原始 Patch_EDMLoss。
        if int(patch_size) != self.active_patch_size:
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

        # 默认沿用原来的错位裁剪。
        pair_geometry = getattr(self, 'pair_geometry', 'shifted')

        if pair_geometry == 'identical':
            # 第二个 Patch 使用第一个 Patch 的裁剪坐标。
            top2, left2 = top1, left1

        clean1 = _extract_patches(clean_full, top1, left1, int(patch_size))
        clean2 = _extract_patches(clean_full, top2, left2, int(patch_size))

        pos1 = _position_maps(top1, left1, pair_batch, int(patch_size), resolution, images.device)
        pos2 = _position_maps(top2, left2, pair_batch, int(patch_size), resolution, images.device)

        # 两个Patch共享相同的噪声水平 sigma
        rnd_normal = torch.randn(
            [pair_batch, 1, 1, 1],
            device=images.device,
        )

        sigma = (
                rnd_normal * self.P_std + self.P_mean
        ).exp()

        weight = (
                         sigma.square() + self.sigma_data ** 2
                 ) / (sigma * self.sigma_data).square()

        if self.noise_mode == 'same':
            # Same Noise:
            # 先在完整MRI图像上生成一次噪声
            # 再裁剪两个重叠Patch
            noisy_full = clean_full + sigma * torch.randn_like(clean_full)

            noisy1 = _extract_patches(
                noisy_full, top1, left1, int(patch_size)
            )
            noisy2 = _extract_patches(
                noisy_full, top2, left2, int(patch_size)
            )

        else:
            # Independent Noise:
            # 保持原来的实现完全不变
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
        if pair_geometry == 'identical':
            # 完全重合：整块都对应同一物理位置。
            D1_overlap, D2_overlap = D1, D2

            # 仅沿用变量名，实际没有几何上的内部/边界之分。
            D_interior, D_boundary = D1_overlap, D2_overlap

        else:
            # 原来的错位重叠分支。
            D1_overlap, D2_overlap = _overlap_views(
                D1, D2, horizontal, overlap
            )

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

        if getattr(self, 'spatial_weight_mode', 'distance') == 'uniform':
            # 只把空间权重设为1，保留 EDM 的噪声水平权重。
            confidence = torch.ones_like(D1_overlap[:, :1])

        else:
            # 原来的距离差权重。
            confidence_raw = (d1 - d2).abs()
            confidence_mean = confidence_raw.mean(
                dim=(1, 2, 3), keepdim=True
            )
            confidence = (
                confidence_raw / confidence_mean.clamp_min(1e-12)
            )
        # 新增开关不改变双 GT、噪声模式及损失归约；空间权重由配置选择。
        # getattr 兼容未保存 alignment_mode 的旧 loss 对象。
        if getattr(self, 'alignment_mode', 'directional') == 'directional':
            # 单向：Interior 作为停止梯度的参考，仅 Boundary 接受辅助梯度。
            aux_pixel = weight * confidence * (D_boundary - D_interior.detach()).square()
        else:
            # 双向：两侧均接受辅助梯度，采用 0.5 系数归一化。
            aux_pixel = 0.5 * weight * confidence * (
                D_boundary - D_interior
            ).square()

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
        training_stats.report(
            'Loss/overlap_center_lambda',
            edm_loss.detach().new_tensor(self.lambda_overlap),
        )

        return edm_loss + weighted_auxiliary
