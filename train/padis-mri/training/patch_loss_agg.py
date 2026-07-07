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

        # Anatomy-gated gradient-aware sampler.
        #
        # This sampler first estimates an anatomical support region from each MRI slice,
        # then uses both radial anatomical proximity and local gradient strength to
        # sample patches. It is designed to avoid wasting too much probability on
        # far-background / external-padding areas while increasing the chance of
        # sampling structural and detail-rich regions.
        #
        # It is NOT a pure high-gradient sampler:
        #   - radial/anatomy gate suppresses far-background and padding;
        #   - gradient score emphasizes local structure;
        #   - clipping prevents skull / extreme boundaries from dominating;
        #   - base probability and uniform fallback keep background modeled.
        #
        # Recommended:
        #   PADIS_AGG_ENABLE=1
        #   PADIS_AGG_SAMPLE_PROB=0.8
        #   PADIS_AGG_THR_RATIO=0.05
        #   PADIS_AGG_BASE=0.05
        #   PADIS_AGG_RADIUS_SCALE=1.15
        #   PADIS_AGG_TAU_SCALE=0.30
        #   PADIS_AGG_GRAD_ALPHA=2.0
        #   PADIS_AGG_GRAD_CLIP_Q=0.95
        #   PADIS_AGG_GATE_BASE=0.20
        self.agg_enable = int(os.environ.get('PADIS_AGG_ENABLE', '0'))
        self.agg_sample_prob = float(os.environ.get('PADIS_AGG_SAMPLE_PROB', '0.8'))
        self.agg_thr_ratio = float(os.environ.get('PADIS_AGG_THR_RATIO', '0.05'))
        self.agg_base = float(os.environ.get('PADIS_AGG_BASE', '0.05'))
        self.agg_radius_scale = float(os.environ.get('PADIS_AGG_RADIUS_SCALE', '1.15'))
        self.agg_tau_scale = float(os.environ.get('PADIS_AGG_TAU_SCALE', '0.30'))
        self.agg_min_pixels = int(os.environ.get('PADIS_AGG_MIN_PIXELS', '64'))

        # Gradient-aware part.
        self.agg_grad_alpha = float(os.environ.get('PADIS_AGG_GRAD_ALPHA', '2.0'))
        self.agg_grad_clip_q = float(os.environ.get('PADIS_AGG_GRAD_CLIP_Q', '0.95'))
        self.agg_gate_base = float(os.environ.get('PADIS_AGG_GATE_BASE', '0.20'))

        if self.agg_enable:
            print(
                f"[Patch_EDMLoss] anatomy-gated gradient-aware sampler enabled: "
                f"prob={self.agg_sample_prob}, "
                f"thr={self.agg_thr_ratio}, "
                f"base={self.agg_base}, "
                f"radius_scale={self.agg_radius_scale}, "
                f"tau_scale={self.agg_tau_scale}, "
                f"grad_alpha={self.agg_grad_alpha}, "
                f"grad_clip_q={self.agg_grad_clip_q}, "
                f"gate_base={self.agg_gate_base}"
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

            # Anatomy-gated gradient-aware sampler.
            # With probability agg_sample_prob, replace the uniform top-left
            # location by a sample from:
            #   score = base + radial_score * anatomy_gate * (1 + alpha * grad_score)
            if self.agg_enable > 0:
                use_agg = torch.rand(batch_size, device=device) < self.agg_sample_prob

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
                grid_cy, grid_cx = torch.meshgrid(ci, cj, indexing='ij')  # [H-th+1, W-tw+1]

                for b in range(batch_size):
                    if not bool(use_agg[b].item()):
                        continue

                    mb = mag[b]
                    peak = mb.amax().clamp_min(1e-8)

                    # Anatomical support is used to estimate center/radius and an anatomy gate.
                    # It is not a hard foreground mask.
                    support = (mb > self.agg_thr_ratio * peak).float()
                    mass = support.sum()

                    if mass < self.agg_min_pixels:
                        continue

                    cy = (support * yy).sum() / mass
                    cx = (support * xx).sum() / mass

                    # Equivalent radius from support area.
                    radius = torch.sqrt(mass / 3.141592653589793) * self.agg_radius_scale
                    tau = torch.clamp(radius * self.agg_tau_scale, min=8.0)

                    dist = torch.sqrt((grid_cy - cy) ** 2 + (grid_cx - cx) ** 2)
                    outside = torch.clamp(dist - radius, min=0.0)

                    # Radial anatomical proximity.
                    radial_score = torch.exp(-0.5 * (outside / tau) ** 2)

                    # Local gradient magnitude by finite difference.
                    gx = torch.zeros_like(mb)
                    gy = torch.zeros_like(mb)
                    gx[:, 1:] = mb[:, 1:] - mb[:, :-1]
                    gy[1:, :] = mb[1:, :] - mb[:-1, :]
                    grad = torch.sqrt(gx.square() + gy.square())

                    # Patch-level average gradient for every candidate top-left.
                    grad_patch = torch.nn.functional.avg_pool2d(
                        grad[None, None],
                        kernel_size=(th, tw),
                        stride=1
                    )[0, 0]  # [H-th+1, W-tw+1]

                    # Patch-level support fraction.
                    support_frac = torch.nn.functional.avg_pool2d(
                        support[None, None],
                        kernel_size=(th, tw),
                        stride=1
                    )[0, 0].clamp(0.0, 1.0)

                    # Robust gradient normalization.
                    # Quantile clipping prevents skull / extreme boundary domination.
                    gp = grad_patch.flatten()
                    positive = gp[gp > 0]

                    if positive.numel() < 8:
                        grad_norm = torch.zeros_like(grad_patch)
                    else:
                        q = torch.quantile(positive, self.agg_grad_clip_q).clamp_min(1e-8)
                        grad_norm = (grad_patch / q).clamp(0.0, 1.0)

                    # Anatomy gate:
                    #   support_frac high -> keep gradient emphasis;
                    #   support_frac low  -> still keep gate_base probability.
                    anatomy_gate = self.agg_gate_base + (1.0 - self.agg_gate_base) * support_frac

                    # Final sampling score.
                    # base keeps background and transition regions modeled.
                    score = (
                        self.agg_base
                        + radial_score * anatomy_gate * (1.0 + self.agg_grad_alpha * grad_norm)
                    )

                    score = score.flatten().clamp_min(1e-8)

                    idx = torch.multinomial(score, 1)[0].long()
                    ii = torch.div(idx, max_j + 1, rounding_mode='floor')
                    jj = idx - ii * (max_j + 1)

                    i[b] = ii
                    j[b] = jj

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
