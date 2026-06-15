# ---------------------------------------------------------------
# Aggregation-aware overlap consistency loss for PaDIS patch EDM.
# ---------------------------------------------------------------

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from torch_utils import persistence
from training.patch_loss import Patch_EDMLoss


def crop_at(images, i, j, patch_size):
    B = images.shape[0]
    rows = torch.arange(patch_size, dtype=torch.long, device=images.device) + i[:, None]
    columns = torch.arange(patch_size, dtype=torch.long, device=images.device) + j[:, None]
    patches = images.permute(1, 0, 2, 3)
    patches = patches[:, torch.arange(B, device=images.device)[:, None, None], rows[:, torch.arange(patch_size, device=images.device)[:, None]], columns[:, None]]
    return patches.permute(1, 0, 2, 3)


def make_pos(i, j, patch_size, resolution):
    device = i.device
    B = i.shape[0]
    th, tw = patch_size, patch_size

    x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

    x_pos = x_pos + j.view(-1, 1, 1, 1)
    y_pos = y_pos + i.view(-1, 1, 1, 1)

    x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
    y_pos = (y_pos / (resolution - 1) - 0.5) * 2.

    images_pos = torch.cat((x_pos, y_pos), dim=1)
    return images_pos


@persistence.persistent_class
class AggOverlapPatch_EDMLoss(Patch_EDMLoss):
    def __init__(
            self,
            P_mean=-1.2,
            P_std=1.2,
            sigma_data=0.5,
            agg_lambda=0.05,
            overlap_ratio=0.25,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        self.agg_lambda = agg_lambda
        self.overlap_ratio = overlap_ratio

    def __call__(self, net, images, patch_size, resolution, labels=None, augment_pipe=None):
        B = images.shape[0]
        device = images.device
        P = patch_size
        O = max(1, int(P * self.overlap_ratio))
        stride = P - O
        h, w = images.shape[2], images.shape[3]

        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        noise_full = torch.randn_like(y) * sigma
        yn_full = y + noise_full

        i = torch.randint(0, h - P + 1, (B,), device=device)
        j = torch.randint(0, w - P - stride + 1, (B,), device=device)

        clean1 = crop_at(y, i, j, P)
        clean2 = crop_at(y, i, j + stride, P)

        noisy1 = crop_at(yn_full, i, j, P)
        noisy2 = crop_at(yn_full, i, j + stride, P)

        pos1 = make_pos(i, j, P, resolution)
        pos2 = make_pos(i, j + stride, P, resolution)

        D1 = net(noisy1, sigma, x_pos=pos1, class_labels=labels, augment_labels=augment_labels)
        D2 = net(noisy2, sigma, x_pos=pos2, class_labels=labels, augment_labels=augment_labels)

        loss_edm = 0.5 * weight * ((D1 - clean1) ** 2 + (D2 - clean2) ** 2)

        loss_cons = torch.zeros_like(loss_edm)
        loss_cons[:, :, :, -O:] = torch.abs(
            D1[:, :, :, -O:] - D2[:, :, :, :O]
        )

        loss = loss_edm + self.agg_lambda * loss_cons

        return loss
