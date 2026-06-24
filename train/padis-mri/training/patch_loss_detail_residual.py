"""Patch EDM loss with explicit detail residual supervision for PaDIS-MRI."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from torch_utils import persistence, training_stats
from training.patch_loss import Patch_EDMLoss
from training.patch_overlap_loss import _extract_patches, _position_maps


def _sample_patch_geometry(images, patch_size):
    device = images.device
    batch = images.shape[0]
    h, w = images.shape[2], images.shape[3]
    if h < patch_size or w < patch_size:
        raise ValueError(f'image size {(h, w)} is smaller than patch_size={patch_size}')
    top = torch.randint(0, h - patch_size + 1, [batch], device=device)
    left = torch.randint(0, w - patch_size + 1, [batch], device=device)
    return top.long(), left.long()


def _spatial_gradients(x):
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    return dx, dy


def _gradient_l1(x, y):
    dx_x, dy_x = _spatial_gradients(x)
    dx_y, dy_y = _spatial_gradients(y)
    return (dx_x - dx_y).abs() + (dy_x - dy_y).abs()


def _edge_weight_from_target(target, edge_alpha=2.0):
    dx, dy = _spatial_gradients(target)
    mag = torch.sqrt(dx.square() + dy.square() + 1e-12).mean(dim=1, keepdim=True)
    mean = mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    return 1.0 + float(edge_alpha) * (mag / mean)


def _per_sample_mean(x):
    return x.mean(dim=(1, 2, 3), keepdim=True)


def _as_loss_map(per_sample, ref):
    return per_sample.expand_as(ref)


@persistence.persistent_class
class DetailResidualPatch_EDMLoss(Patch_EDMLoss):
    """EDM denoising loss plus residual, gradient, and edge-weighted detail terms."""

    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        lambda_residual=0.2,
        lambda_gradient=0.1,
        lambda_edge=0.1,
        edge_alpha=2.0,
        detail_sigma_weight=True,
    ):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        self.lambda_residual = float(lambda_residual)
        self.lambda_gradient = float(lambda_gradient)
        self.lambda_edge = float(lambda_edge)
        self.edge_alpha = float(edge_alpha)
        self.detail_sigma_weight = bool(detail_sigma_weight)

    def __call__(
        self,
        net,
        images,
        patch_size,
        resolution,
        labels=None,
        augment_pipe=None,
    ):
        if augment_pipe is not None:
            images, augment_labels = augment_pipe(images)
        else:
            augment_labels = None

        batch = images.shape[0]
        patch_size = int(patch_size)
        top, left = _sample_patch_geometry(images, patch_size)
        clean = _extract_patches(images, top, left, patch_size)
        x_pos = _position_maps(top, left, batch, patch_size, resolution, images.device)

        rnd_normal = torch.randn([batch, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma.square() + self.sigma_data ** 2) / (sigma * self.sigma_data).square()
        noisy = clean + sigma * torch.randn_like(clean)

        out = net(
            noisy,
            sigma,
            x_pos=x_pos,
            class_labels=labels,
            augment_labels=augment_labels,
            return_detail=True,
        )
        if not isinstance(out, (tuple, list)) or len(out) < 4:
            raise RuntimeError(
                'DetailResidualPatch_EDMLoss requires a network that returns '
                '(D_final, D_base, r_detail, gate) when return_detail=True.'
            )
        D_final, D_base, r_detail, gate = out[:4]

        edm_loss = weight * (D_final - clean).square()
        training_stats.report('Loss/edm', edm_loss.detach())

        detail_weight = self.sigma_data ** 2 / (sigma.square() + self.sigma_data ** 2)
        if not self.detail_sigma_weight:
            detail_weight = torch.ones_like(detail_weight)

        # Residual target is detached from the base path by construction.
        target_residual = clean - D_base.detach()
        residual_pixel = (r_detail - target_residual).abs()
        residual_per = detail_weight * _per_sample_mean(residual_pixel)
        residual_loss = _as_loss_map(residual_per, edm_loss)

        grad_pixel = _gradient_l1(D_final, clean)
        grad_per = detail_weight * _per_sample_mean(grad_pixel)
        grad_loss = _as_loss_map(grad_per, edm_loss)

        edge_weight = _edge_weight_from_target(clean.detach(), edge_alpha=self.edge_alpha)
        edge_pixel = edge_weight * (D_final - clean).abs()
        edge_per = detail_weight * _per_sample_mean(edge_pixel)
        edge_loss = _as_loss_map(edge_per, edm_loss)

        weighted_residual = self.lambda_residual * residual_loss
        weighted_gradient = self.lambda_gradient * grad_loss
        weighted_edge = self.lambda_edge * edge_loss

        edm_mean = edm_loss.detach().mean().clamp_min(1e-12)
        training_stats.report('Loss/detail_residual_raw', residual_per.detach())
        training_stats.report('Loss/detail_gradient_raw', grad_per.detach())
        training_stats.report('Loss/detail_edge_raw', edge_per.detach())
        training_stats.report('Loss/detail_residual_ratio', weighted_residual.detach().mean() / edm_mean)
        training_stats.report('Loss/detail_gradient_ratio', weighted_gradient.detach().mean() / edm_mean)
        training_stats.report('Loss/detail_edge_ratio', weighted_edge.detach().mean() / edm_mean)
        training_stats.report('Loss/detail_gate_mean', gate.detach().mean())
        training_stats.report('Loss/detail_r_abs_mean', r_detail.detach().abs().mean())
        training_stats.report('Loss/detail_sigma_weight_mean', detail_weight.detach().mean())

        return edm_loss + weighted_residual + weighted_gradient + weighted_edge
