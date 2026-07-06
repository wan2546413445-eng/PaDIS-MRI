# ---------------------------------------------------------------
# Taken from the following link as is from:
# https://github.com/jasonhu4/PaDIS/blob/main/training/patch_loss.py
#
# The license for the original version of this file can be
# found here: https://github.com/jasonhu4/PaDIS/blob/main/LICENSE.
# ---------------------------------------------------------------


"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class Patch_EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        # print("We are using PATCH EDM")
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        # Anatomy-centered radial soft sampler.
        # This sampler estimates an anatomical support center and radius
        # from each MRI slice, then softly downweights far-background and
        # external-padding regions.
        #
        # It does NOT use high-gradient selection.
        # It does NOT exclude skull or image-internal background.
        # It only changes the probability of patch-center sampling.
        #
        # Recommended:
        #   PADIS_RADIAL_ENABLE=1
        #   PADIS_RADIAL_SAMPLE_PROB=0.8
        #   PADIS_RADIAL_THR_RATIO=0.05
        #   PADIS_RADIAL_BASE=0.08
        #   PADIS_RADIAL_RADIUS_SCALE=1.20
        #   PADIS_RADIAL_TAU_SCALE=0.35
        self.radial_enable = int(os.environ.get('PADIS_RADIAL_ENABLE', '0'))
        self.radial_sample_prob = float(os.environ.get('PADIS_RADIAL_SAMPLE_PROB', '0.8'))
        self.radial_thr_ratio = float(os.environ.get('PADIS_RADIAL_THR_RATIO', '0.05'))
        self.radial_base = float(os.environ.get('PADIS_RADIAL_BASE', '0.08'))
        self.radial_radius_scale = float(os.environ.get('PADIS_RADIAL_RADIUS_SCALE', '1.20'))
        self.radial_tau_scale = float(os.environ.get('PADIS_RADIAL_TAU_SCALE', '0.35'))
        self.radial_min_pixels = int(os.environ.get('PADIS_RADIAL_MIN_PIXELS', '64'))

        if self.radial_enable:
            print(
                f"[Patch_EDMLoss] radial-soft sampler enabled: "
                f"prob={self.radial_sample_prob}, "
                f"thr={self.radial_thr_ratio}, "
                f"base={self.radial_base}, "
                f"radius_scale={self.radial_radius_scale}, "
                f"tau_scale={self.radial_tau_scale}"
            )



    def pachify(self, images, patch_size, padding=None):
        # 输入 images: [B, C, H, W]，从每张整图中随机裁取一个 patch，
        # 同时生成 patch 内每个像素对应的二维位置编码。
        device = images.device
        batch_size, resolution = images.size(0), images.size(2)
        # 在原图四周补零，使原始图像边缘也能被完整 patch 覆盖。
        if padding is not None:
            padded = torch.zeros(
                (images.size(0), images.size(1),
                 images.size(2) + padding * 2,
                 images.size(3) + padding * 2),
                dtype=images.dtype, device=device
            )
            padded[:, :, padding:-padding, padding:-padding] = images
        else:
            padded = images
        h, w = padded.size(2), padded.size(3)
        th, tw = patch_size, patch_size
        # 为 batch 中每张图独立随机选择 patch 左上角坐标：
        # i 为起始行坐标，j 为起始列坐标。
        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device).long()
            j = torch.zeros((batch_size,), device=device).long()
        else:
            # Original uniform sampler over the whole padded canvas.
            i_uniform = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j_uniform = torch.randint(0, w - tw + 1, (batch_size,), device=device)
            i = i_uniform.clone()
            j = j_uniform.clone()

            # Anatomy-centered radial soft sampler.
            # With probability radial_sample_prob, sample patch center from a
            # soft radial distribution estimated from the current slice.
            if self.radial_enable > 0:
                use_radial = torch.rand(batch_size, device=device) < self.radial_sample_prob

                # Magnitude image. For MRI complex input, first two channels are real/imag.
                use_ch = min(2, padded.size(1))
                mag = padded[:, :use_ch].float().square().sum(dim=1).sqrt()  # [B,H,W]

                yy = torch.arange(h, device=device, dtype=torch.float32).view(h, 1).expand(h, w)
                xx = torch.arange(w, device=device, dtype=torch.float32).view(1, w).expand(h, w)

                max_i = h - th
                max_j = w - tw

                # Candidate patch centers corresponding to all possible top-left positions.
                ci = torch.arange(0, max_i + 1, device=device, dtype=torch.float32) + th / 2.0
                cj = torch.arange(0, max_j + 1, device=device, dtype=torch.float32) + tw / 2.0
                grid_cy, grid_cx = torch.meshgrid(ci, cj, indexing='ij')

                for b in range(batch_size):
                    if not bool(use_radial[b].item()):
                        continue

                    mb = mag[b]
                    peak = mb.amax().clamp_min(1e-8)

                    # Estimate anatomical support from magnitude.
                    # This is only for center/radius estimation, not a hard sampling mask.
                    support = (mb > self.radial_thr_ratio * peak).float()
                    mass = support.sum()

                    if mass < self.radial_min_pixels:
                        continue

                    cy = (support * yy).sum() / mass
                    cx = (support * xx).sum() / mass

                    # Equivalent radius from support area.
                    # radius_scale > 1 expands to include skull/boundary nearby background.
                    radius = torch.sqrt(mass / 3.141592653589793) * self.radial_radius_scale
                    tau = torch.clamp(radius * self.radial_tau_scale, min=8.0)

                    dist = torch.sqrt((grid_cy - cy) ** 2 + (grid_cx - cx) ** 2)
                    outside = torch.clamp(dist - radius, min=0.0)

                    # Inside estimated anatomical envelope: high weight.
                    # Outside: gradually decays, but never zero due to radial_base.
                    score = self.radial_base + torch.exp(-0.5 * (outside / tau) ** 2)
                    score = score.flatten().clamp_min(1e-8)

                    idx = torch.multinomial(score, 1)[0].long()
                    ii = torch.div(idx, max_j + 1, rounding_mode='floor')
                    jj = idx - ii * (max_j + 1)

                    i[b] = ii
                    j[b] = jj
        # 根据左上角坐标生成每个 patch 的完整行、列索引。
        rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]
        # 使用高级索引一次性裁取整个 batch 的 patch，
        # 等价于对每张图执行 images[b, :, i:i+P, j:j+P]。
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[
            :,
            torch.arange(batch_size, device=device)[:, None, None],
            rows[:, torch.arange(th, device=device)[:, None]],
            columns[:, None]
        ]
        padded = padded.permute(1, 0, 2, 3)  # [B, C, P, P]
        # 构造 patch 内逐像素的局部横、纵坐标网格。
        x_pos = torch.arange(tw, device=device).view(1, 1, 1, tw)
        x_pos = x_pos.repeat(batch_size, 1, th, 1)
        y_pos = torch.arange(th, device=device).view(1, 1, th, 1)
        y_pos = y_pos.repeat(batch_size, 1, 1, tw)
        # 加上 patch 左上角位置，得到相对于整张 padded image 的绝对坐标。
        x_pos = x_pos + j.view(-1, 1, 1, 1)
        y_pos = y_pos + i.view(-1, 1, 1, 1)
        # 将位置坐标归一化。分母使用的是 padding 前的原图分辨率，
        # 因此存在 padding 时，部分位置编码可能超过 [-1, 1]。
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
        # 拼接为两通道位置编码：[B, 2, P, P]，
        # 第一个通道为 x 坐标，第二个通道为 y 坐标。
        images_pos = torch.cat((x_pos, y_pos), dim=1)
        return padded, images_pos



    #训练逻辑
    def __call__(self, net, images, patch_size, resolution, labels=None, augment_pipe=None):
        images, images_pos = self.pachify(images, patch_size)

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        yn = y + n

        D_yn = net(yn, sigma, x_pos=images_pos, class_labels=labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
