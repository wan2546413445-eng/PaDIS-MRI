"""AGGA patch sampling at all scales; identical-patch OPA only at 64 pixels."""

import torch

from torch_utils import persistence, training_stats
from training.patch_loss import Patch_EDMLoss
from training.agg_sampler import sample_single_locations
from training.patch_overlap_loss import _extract_patches, _position_maps


@persistence.persistent_class
class OPAAGGAPatchEDMLoss(Patch_EDMLoss):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                 opa_weight=1.0, rho=0.65, gamma=1.0, score_mode='multiply'):
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
        if not (0 <= opa_weight < float('inf') and 0 <= rho < 1
                and 0 <= gamma < float('inf')):
            raise ValueError('Require finite opa_weight/gamma >= 0 and 0 <= rho < 1')
        if score_mode not in ('multiply', 'add'):
            raise ValueError('score_mode must be multiply or add')
        self.opa_weight = float(opa_weight)
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.score_mode = score_mode

    def __call__(self, net, images, patch_size, resolution,
                 labels=None, augment_pipe=None):
        patch_size = int(patch_size)
        batch = images.shape[0]

        # AGGA acts on EVERY training patch scale (16, 32, and 64).
        top, left = sample_single_locations(images, patch_size, self.rho, self.gamma,
                                            score_mode=self.score_mode)
        clean = _extract_patches(images, top, left, patch_size)
        pos = _position_maps(top, left, batch, patch_size, resolution, images.device)
        sigma = (torch.randn([batch, 1, 1, 1], device=images.device)
                 * self.P_std + self.P_mean).exp()
        weight = (sigma.square() + self.sigma_data ** 2) / (sigma * self.sigma_data).square()

        if patch_size != 64:
            # Original one-view EDM loss; only the sampling distribution changes.
            target, augment_labels = (augment_pipe(clean) if augment_pipe is not None
                                      else (clean, None))
            noisy = target + sigma * torch.randn_like(target)
            prediction = net(noisy, sigma, x_pos=pos,
                             class_labels=labels, augment_labels=augment_labels)
            return weight * (prediction - target).square()

        if augment_pipe is not None:
            raise ValueError('64-pixel OPA branch requires --augment=0')

        # 64x64: one AGGA crop, two independent noise realizations, two GT terms.
        noisy1 = clean + sigma * torch.randn_like(clean)
        noisy2 = clean + sigma * torch.randn_like(clean)
        prediction = net(torch.cat([noisy1, noisy2], 0),
                         torch.cat([sigma, sigma], 0),
                         x_pos=torch.cat([pos, pos], 0),
                         class_labels=torch.cat([labels, labels], 0) if labels is not None else None,
                         augment_labels=None)
        D1, D2 = prediction[:batch], prediction[batch:]
        edm = 0.5 * (weight * (D1 - clean).square() + weight * (D2 - clean).square())
        aux_per_pair = (0.5 * weight * (D2 - D1).square()).mean((1, 2, 3), keepdim=True)
        auxiliary = self.opa_weight * aux_per_pair.expand_as(edm)
        training_stats.report('Loss/edm', edm.detach())
        training_stats.report('Loss/opa_raw', aux_per_pair.detach())
        training_stats.report('Loss/opa_weighted', auxiliary.detach())
        return edm + auxiliary
