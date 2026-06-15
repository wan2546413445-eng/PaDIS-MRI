import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from torch_utils import persistence
from torch_utils import training_stats
from training.patch_loss import Patch_EDMLoss


def _extract_patches(images, top, left, patch_size):
    batch_size = images.shape[0]
    rows = torch.arange(patch_size, device=images.device).view(1, patch_size, 1) + top.view(-1, 1, 1)
    cols = torch.arange(patch_size, device=images.device).view(1, 1, patch_size) + left.view(-1, 1, 1)
    x = images.permute(1, 0, 2, 3)
    x = x[:, torch.arange(batch_size, device=images.device).view(-1, 1, 1), rows, cols]
    return x.permute(1, 0, 2, 3)


def _position_maps(top, left, batch_size, patch_size, resolution, device):
    y = torch.arange(patch_size, dtype=torch.float32, device=device).view(1, 1, patch_size, 1) + top.view(-1, 1, 1, 1).float()
    x = torch.arange(patch_size, dtype=torch.float32, device=device).view(1, 1, 1, patch_size) + left.view(-1, 1, 1, 1).float()
    y = y.repeat(1, 1, 1, patch_size)
    x = x.repeat(1, 1, patch_size, 1)
    x = (x / (resolution - 1) - 0.5) * 2.0
    y = (y / (resolution - 1) - 0.5) * 2.0
    return torch.cat([x, y], dim=1).repeat(batch_size // x.shape[0], 1, 1, 1) if x.shape[0] != batch_size else torch.cat([x, y], dim=1)


def _sample_overlap_geometry(images, patch_size):
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
    top_h = torch.randint(0, max_top_h + 1, [batch], device=device)
    left_h = torch.randint(0, max_left_h + 1, [batch], device=device)
    top_v = torch.randint(0, max_top_v + 1, [batch], device=device)
    left_v = torch.randint(0, max_left_v + 1, [batch], device=device)
    top1 = torch.where(horizontal, top_h, top_v)
    left1 = torch.where(horizontal, left_h, left_v)
    top2 = torch.where(horizontal, top1, top1 + stride)
    left2 = torch.where(horizontal, left1 + stride, left1)
    return top1.long(), left1.long(), top2.long(), left2.long(), horizontal, overlap


def _overlap_views(x1, x2, horizontal, overlap):
    b = x1.shape[0]
    mask = horizontal.view(b, 1, 1, 1)
    a_h, b_h = x1[:, :, :, -overlap:], x2[:, :, :, :overlap]
    a_v, b_v = x1[:, :, -overlap:, :], x2[:, :, :overlap, :]
    if overlap != x1.shape[-1]:
        a_v = a_v.transpose(2, 3)
        b_v = b_v.transpose(2, 3)
    return torch.where(mask, a_h, a_v), torch.where(mask, b_h, b_v)


def _put_overlap(aux_overlap, horizontal, patch_size, overlap):
    aux = torch.zeros([aux_overlap.shape[0], aux_overlap.shape[1], patch_size, patch_size], device=aux_overlap.device, dtype=aux_overlap.dtype)
    for idx in range(aux.shape[0]):
        if bool(horizontal[idx]):
            aux[idx, :, :, -overlap:] = aux_overlap[idx]
        else:
            aux[idx, :, -overlap:, :] = aux_overlap[idx].transpose(1, 2)
    return aux


def _boundary_distances(horizontal, patch_size, overlap, device, dtype):
    coords = torch.arange(patch_size, device=device, dtype=dtype)
    dist = torch.minimum(coords, (patch_size - 1) - coords)
    d1_h = dist[-overlap:].view(1, 1, 1, overlap).repeat(horizontal.shape[0], 1, patch_size, 1)
    d2_h = dist[:overlap].view(1, 1, 1, overlap).repeat(horizontal.shape[0], 1, patch_size, 1)
    d1_v = dist[-overlap:].view(1, 1, overlap, 1).repeat(horizontal.shape[0], 1, 1, patch_size).transpose(2, 3)
    d2_v = dist[:overlap].view(1, 1, overlap, 1).repeat(horizontal.shape[0], 1, 1, patch_size).transpose(2, 3)
    mask = horizontal.view(-1, 1, 1, 1)
    return torch.where(mask, d1_h, d1_v), torch.where(mask, d2_h, d2_v)


@persistence.persistent_class
class SameNoiseOverlapPatch_EDMLoss(Patch_EDMLoss):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, lambda_overlap=0.01, overlap_mode='same'):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        self.lambda_overlap = lambda_overlap
        self.overlap_mode = overlap_mode

    def __call__(self, net, images, patch_size, resolution, labels=None, augment_pipe=None):
        if augment_pipe is not None:
            images, augment_labels = augment_pipe(images)
        else:
            augment_labels = None
        full_batch = images.shape[0]
        selected_batch = full_batch // 2
        select = torch.randperm(full_batch, device=images.device)[:selected_batch]
        clean_full = images[select]
        labels_sel = labels[select] if labels is not None else None
        top1, left1, top2, left2, horizontal, overlap = _sample_overlap_geometry(clean_full, patch_size)
        sigma = (torch.randn([selected_batch, 1, 1, 1], device=images.device) * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noisy_full = clean_full + sigma * torch.randn_like(clean_full)
        clean1, clean2 = _extract_patches(clean_full, top1, left1, patch_size), _extract_patches(clean_full, top2, left2, patch_size)
        noisy1, noisy2 = _extract_patches(noisy_full, top1, left1, patch_size), _extract_patches(noisy_full, top2, left2, patch_size)
        pos1 = _position_maps(top1, left1, selected_batch, patch_size, resolution, images.device)
        pos2 = _position_maps(top2, left2, selected_batch, patch_size, resolution, images.device)
        D = net(torch.cat([noisy1, noisy2], 0), torch.cat([sigma, sigma], 0), x_pos=torch.cat([pos1, pos2], 0), class_labels=torch.cat([labels_sel, labels_sel], 0) if labels_sel is not None else None, augment_labels=torch.cat([augment_labels, augment_labels], 0) if augment_labels is not None else None)
        D1, D2 = D[:selected_batch], D[selected_batch:]
        edm_loss = weight * (D1 - clean1).square() + weight * (D2 - clean2).square()
        D1_o, D2_o = _overlap_views(D1, D2, horizontal, overlap)
        target_o, _ = _overlap_views(clean1, clean2, horizontal, overlap)
        d1, d2 = _boundary_distances(horizontal, patch_size, overlap, images.device, images.dtype)
        interior_is_1 = d1 >= d2
        D_int = torch.where(interior_is_1, D1_o, D2_o)
        D_bnd = torch.where(interior_is_1, D2_o, D1_o)
        err_i = (D_int.detach() - target_o).square().sum(dim=1, keepdim=True)
        err_b = (D_bnd.detach() - target_o).square().sum(dim=1, keepdim=True)
        gate = (err_i < err_b).to(images.dtype)
        conf = (d1 - d2).abs(); conf = conf / conf.max().clamp_min(1e-12)
        aux_overlap = weight * conf * gate * F.smooth_l1_loss(D_bnd, D_int.detach(), reduction='none')
        aux_full = _put_overlap(aux_overlap, horizontal, patch_size, overlap) * ((full_batch / selected_batch) * (patch_size / overlap))
        training_stats.report('Loss/edm', edm_loss.detach())
        training_stats.report('Loss/overlap_same', aux_overlap.detach())
        training_stats.report('Loss/overlap_same_weighted', (self.lambda_overlap * aux_full).detach())
        training_stats.report('Loss/overlap_same_active_fraction', gate.detach().mean())
        training_stats.report('Loss/overlap_same_ratio', ((self.lambda_overlap * aux_full).detach().mean() / edm_loss.detach().mean().clamp_min(1e-12)))
        return edm_loss + self.lambda_overlap * aux_full


@persistence.persistent_class
class DifferentNoiseOverlapPatch_EDMLoss(Patch_EDMLoss):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, lambda_overlap=0.01, min_noise_ratio=1.25, max_noise_ratio=3.0, overlap_mode='diff'):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        self.lambda_overlap = lambda_overlap; self.min_noise_ratio = min_noise_ratio; self.max_noise_ratio = max_noise_ratio; self.overlap_mode = overlap_mode

    def __call__(self, net, images, patch_size, resolution, labels=None, augment_pipe=None):
        if augment_pipe is not None:
            images, augment_labels = augment_pipe(images)
        else:
            augment_labels = None
        full_batch = images.shape[0]; selected_batch = full_batch // 2
        select = torch.randperm(full_batch, device=images.device)[:selected_batch]
        clean_full = images[select]; labels_sel = labels[select] if labels is not None else None
        top1, left1, top2, left2, horizontal, overlap = _sample_overlap_geometry(clean_full, patch_size)
        sigma_a = (torch.randn([selected_batch, 1, 1, 1], device=images.device) * self.P_std + self.P_mean).exp()
        sigma_b = (torch.randn([selected_batch, 1, 1, 1], device=images.device) * self.P_std + self.P_mean).exp()
        sigma_low, sigma_high = torch.minimum(sigma_a, sigma_b), torch.maximum(sigma_a, sigma_b)
        eps = torch.randn_like(clean_full)
        noisy_low_full, noisy_high_full = clean_full + sigma_low * eps, clean_full + sigma_high * eps
        swap = (torch.rand([selected_batch, 1, 1, 1], device=images.device) < 0.5)
        nlow1, nlow2 = _extract_patches(noisy_low_full, top1, left1, patch_size), _extract_patches(noisy_low_full, top2, left2, patch_size)
        nhigh1, nhigh2 = _extract_patches(noisy_high_full, top1, left1, patch_size), _extract_patches(noisy_high_full, top2, left2, patch_size)
        noisy1, noisy2 = torch.where(swap, nhigh1, nlow1), torch.where(swap, nlow2, nhigh2)
        sigma1, sigma2 = torch.where(swap, sigma_high, sigma_low), torch.where(swap, sigma_low, sigma_high)
        clean1, clean2 = _extract_patches(clean_full, top1, left1, patch_size), _extract_patches(clean_full, top2, left2, patch_size)
        pos1 = _position_maps(top1, left1, selected_batch, patch_size, resolution, images.device); pos2 = _position_maps(top2, left2, selected_batch, patch_size, resolution, images.device)
        D = net(torch.cat([noisy1, noisy2], 0), torch.cat([sigma1, sigma2], 0), x_pos=torch.cat([pos1, pos2], 0), class_labels=torch.cat([labels_sel, labels_sel], 0) if labels_sel is not None else None, augment_labels=torch.cat([augment_labels, augment_labels], 0) if augment_labels is not None else None)
        D1, D2 = D[:selected_batch], D[selected_batch:]
        w1 = (sigma1 ** 2 + self.sigma_data ** 2) / (sigma1 * self.sigma_data) ** 2; w2 = (sigma2 ** 2 + self.sigma_data ** 2) / (sigma2 * self.sigma_data) ** 2
        edm_loss = w1 * (D1 - clean1).square() + w2 * (D2 - clean2).square()
        D1_o, D2_o = _overlap_views(D1, D2, horizontal, overlap); target_o, _ = _overlap_views(clean1, clean2, horizontal, overlap)
        low_is_1 = ~swap
        D_low = torch.where(low_is_1, D1_o, D2_o); D_high = torch.where(low_is_1, D2_o, D1_o)
        gt_gate = ((D_low.detach() - target_o).square().sum(1, keepdim=True) < (D_high.detach() - target_o).square().sum(1, keepdim=True)).to(images.dtype)
        noise_ratio = sigma_high / sigma_low
        ratio_gate = ((noise_ratio >= self.min_noise_ratio) & (noise_ratio <= self.max_noise_ratio)).to(images.dtype)
        weight_high = (sigma_high ** 2 + self.sigma_data ** 2) / (sigma_high * self.sigma_data) ** 2
        aux_overlap = weight_high * gt_gate * ratio_gate * F.smooth_l1_loss(D_high, D_low.detach(), reduction='none')
        aux_full = _put_overlap(aux_overlap, horizontal, patch_size, overlap) * ((full_batch / selected_batch) * (patch_size / overlap))
        training_stats.report('Loss/edm', edm_loss.detach()); training_stats.report('Loss/overlap_diff', aux_overlap.detach())
        training_stats.report('Loss/overlap_diff_weighted', (self.lambda_overlap * aux_full).detach())
        training_stats.report('Loss/overlap_diff_active_fraction', (gt_gate * ratio_gate).detach().mean())
        training_stats.report('Loss/noise_ratio_mean', noise_ratio.detach().mean()); training_stats.report('Loss/noise_ratio_valid_fraction', ratio_gate.detach().mean())
        training_stats.report('Loss/overlap_diff_ratio', ((self.lambda_overlap * aux_full).detach().mean() / edm_loss.detach().mean().clamp_min(1e-12)))
        return edm_loss + self.lambda_overlap * aux_full
