# ---------------------------------------------------------------
# Context-center patch loss for PaDIS-MRI.
#
# New file:
#   train/padis-mri/training/patch_loss_context_center.py
#
# This file does not replace training/patch_loss.py.
#
# Method:
#   - Target patch size: p
#   - Network input size: p + 2 * context_margin
#   - Loss region: center p x p only
#
# Original EDM defaults are preserved unless explicitly changed by another
# training entrypoint:
#   P_mean=-1.2, P_std=1.2, sigma_data=0.5
#
# Example:
#   patch_size=64, context_margin=16
#   network input = 96 x 96
#   supervised center = 64 x 64
# ---------------------------------------------------------------

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from torch_utils import persistence

from training.patch_loss import Patch_EDMLoss


@persistence.persistent_class
class Patch_ContextCenter_EDMLoss(Patch_EDMLoss):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, context_margin=0):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        self.context_margin = int(context_margin)

    @staticmethod
    def _center_crop(x, target_size, margin):
        if margin <= 0:
            return x
        return x[:, :, margin:margin + target_size, margin:margin + target_size]

    def pachify(self, images, patch_size, padding=None, context_margin=None):
        device = images.device
        batch_size = images.size(0)
        resolution = images.size(2)

        margin = self.context_margin if context_margin is None else int(context_margin)
        target_size = int(patch_size)
        input_size = target_size + 2 * margin

        if margin < 0:
            raise ValueError(f"context_margin must be non-negative, got {margin}")

        if padding is not None:
            padded = torch.zeros(
                (
                    images.size(0),
                    images.size(1),
                    images.size(2) + padding * 2,
                    images.size(3) + padding * 2,
                ),
                dtype=images.dtype,
                device=device,
            )
            padded[:, :, padding:-padding, padding:-padding] = images
        else:
            padded = images

        h, w = padded.size(2), padded.size(3)
        th, tw = input_size, input_size

        if th > h or tw > w:
            raise ValueError(
                f"context_margin makes input patch too large: "
                f"patch_size={target_size}, context_margin={margin}, "
                f"input_size={input_size}, image_size=({h}, {w})"
            )

        if w == tw and h == th:
            i = torch.zeros((batch_size,), device=device, dtype=torch.long)
            j = torch.zeros((batch_size,), device=device, dtype=torch.long)
        else:
            i = torch.randint(0, h - th + 1, (batch_size,), device=device)
            j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

        rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]

        padded = padded.permute(1, 0, 2, 3)
        input_patch = padded[
            :,
            torch.arange(batch_size, device=device)[:, None, None],
            rows[:, torch.arange(th, device=device)[:, None]],
            columns[:, None],
        ]
        input_patch = input_patch.permute(1, 0, 2, 3)

        x_pos = (
            torch.arange(tw, dtype=torch.long, device=device)
            .unsqueeze(0)
            .repeat(th, 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1, 1)
        )
        y_pos = (
            torch.arange(th, dtype=torch.long, device=device)
            .unsqueeze(1)
            .repeat(1, tw)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1, 1)
        )

        x_pos = x_pos + j.view(-1, 1, 1, 1)
        y_pos = y_pos + i.view(-1, 1, 1, 1)

        # Same coordinate normalization as original PaDIS:
        # absolute padded-image coordinates mapped to [-1, 1].
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.0
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.0
        input_pos = torch.cat((x_pos, y_pos), dim=1)

        return input_patch, input_pos, margin, target_size

    def __call__(
        self,
        net,
        images,
        patch_size,
        resolution,
        labels=None,
        augment_pipe=None,
        context_margin=None,
    ):
        images, images_pos, margin, target_size = self.pachify(
            images=images,
            patch_size=patch_size,
            context_margin=context_margin,
        )

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        yn = y + n

        D_yn = net(
            yn,
            sigma,
            x_pos=images_pos,
            class_labels=labels,
            augment_labels=augment_labels,
        )

        D_center = self._center_crop(D_yn, target_size=target_size, margin=margin)
        y_center = self._center_crop(y, target_size=target_size, margin=margin)

        loss = weight * ((D_center - y_center) ** 2)
        return loss
