"""Sparse nonlocal group-energy shrinkage and feature fusion for PaDIS-MRI.

NGS-v2: cosine matching, top-K groups, group shrinkage and direct nonlocal
feature compensation. Existing NGS-v1 pickles retain their original forward
path (they do not have the topk attribute).
Inspired by GroupCDL; this is a PaDIS-specific adaptation, not a reproduction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value):
    return math.log(math.expm1(value))


class NonlocalGroupShrinkage(nn.Module):
    def __init__(self, out_channels, dim=48, stride=8, topk=32,
                 rho_init=0.1, tau_init=0.2, proj_std=0.02):
        super().__init__()
        if topk < 1 or rho_init <= 1e-4 or tau_init <= 0:
            raise ValueError('Invalid topk/rho_init/tau_init')
        self.stride = stride
        self.topk = topk
        self.stem = nn.Conv2d(2, dim, stride, stride=stride)
        self.theta = nn.Linear(dim, dim, bias=False)
        self.phi = nn.Linear(dim, dim, bias=False)
        self.phi.load_state_dict(self.theta.state_dict())
        self.rho = nn.Linear(3, 1)
        self.tau = nn.Linear(3, dim)
        # Concatenate shrunk local feature and nonlocal compensation.
        self.proj = nn.Conv2d(2 * dim, out_channels, 1)
        nn.init.zeros_(self.rho.weight)
        nn.init.constant_(self.rho.bias, _inverse_softplus(rho_init - 1e-4))
        nn.init.zeros_(self.tau.weight)
        nn.init.constant_(self.tau.bias, _inverse_softplus(tau_init))
        nn.init.normal_(self.proj.weight, std=proj_std)
        nn.init.zeros_(self.proj.bias)

    def encode(self, x):
        # Nonoverlapping stride x stride cells; preserve existing context API.
        return self.stem(x.to(self.stem.weight.dtype))

    def valid_context(self, origins, local_hw, context_hw):
        gh, gw = context_hw
        s = self.stride
        rows = torch.arange(gh, device=origins.device).repeat_interleave(gw) * s
        cols = torch.arange(gw, device=origins.device).repeat(gh) * s
        top, left = origins.long().unbind(-1)
        return ~((rows[None] < top[:, None] + local_hw[0])
                 & (rows[None] + s > top[:, None])
                 & (cols[None] < left[:, None] + local_hw[1])
                 & (cols[None] + s > left[:, None]))

    def _legacy_features(self, q, v, qk, kk, rho, tau, valid, d):
        # Exact v1 math, only used when loading pre-v2 full-object snapshots.
        dist = (qk.square().sum(-1, keepdim=True)
                + kk.square().sum(-1).unsqueeze(1)
                - 2 * (qk @ kk.transpose(-1, -2))).clamp_min(0) / d
        logits = (-dist / rho.unsqueeze(-1)).masked_fill(~valid[:, None], -1e9)
        gamma = torch.softmax(logits, dim=-1) * valid[:, None]
        energy = torch.sqrt(gamma @ v.square() + 1e-8)
        return q * torch.relu(1 - tau / (energy + 1e-8))

    def _sparse_features(self, q, v, qk, kk, rho, tau, valid):
        b = q.shape[0]
        # Learned cosine similarity: feature magnitudes no longer set the
        # scale of the matching logits. Exclude cells overlapping the query.
        qk = F.normalize(qk, p=2, dim=-1, eps=1e-6)
        kk = F.normalize(kk, p=2, dim=-1, eps=1e-6)
        sim = qk @ kk.transpose(-1, -2)
        sim = sim.masked_fill(~valid[:, None], -1e9)

        k = min(self.topk, v.shape[1])
        scores, indices = sim.topk(k, dim=-1, largest=True, sorted=False)
        # Gather only selected values, not an expanded B x Q x N x D tensor.
        # The encoded full context may be shared by every query patch.
        if v.shape[0] == 1:
            selected_v = v[0][indices]
        else:
            batch_idx = torch.arange(b, device=v.device).view(b, 1, 1)
            selected_v = v[batch_idx, indices]

        # Handle fewer than K valid cells, including an entirely invalid set.
        selected_valid = valid.gather(1, indices.reshape(b, -1)).reshape_as(indices)
        logits = (scores / rho.unsqueeze(-1)).masked_fill(~selected_valid, -1e9)
        gamma = torch.softmax(logits, dim=-1) * selected_valid.to(scores.dtype)
        gamma = gamma / gamma.sum(-1, keepdim=True).clamp_min(1e-12)
        weights = gamma.unsqueeze(-1)

        # 1. Group-energy shrinkage (retains the v1 mechanism).
        energy = torch.sqrt((weights * selected_v.square()).sum(-2) + 1e-8)
        shrunk = q * torch.relu(1 - tau / (energy + 1e-8))
        # 2. Direct remote feature compensation, not just energy modulation.
        remote = (weights * selected_v).sum(-2)
        compensation = remote - q
        return torch.cat([shrunk, compensation], dim=-1)

    def forward(self, local, context, origins, noise_labels, out_hw):
        z = self.encode(local)
        b, d, h, w = z.shape
        if context.shape[0] not in (1, b):
            raise ValueError('context batch must be 1 or local batch size')
        if origins.shape != (b, 2):
            raise ValueError('origins must be [B,2] (top,left)')
        q = z.flatten(2).transpose(1, 2).float()
        v = context.flatten(2).transpose(1, 2).float()
        qk = self.theta(q.to(self.theta.weight.dtype)).float()
        kk = self.phi(v.to(self.phi.weight.dtype)).float()
        t = noise_labels.float().reshape(-1, 1).expand(b, 1)
        cond = torch.cat([torch.ones_like(t), t, t.square()], dim=-1)
        rho = F.softplus(self.rho(cond.to(self.rho.weight.dtype)).float()) + 1e-4
        tau = F.softplus(self.tau(cond.to(self.tau.weight.dtype)).float()).unsqueeze(1)
        valid = self.valid_context(origins, local.shape[-2:], context.shape[-2:])

        if hasattr(self, 'topk'):
            feat = self._sparse_features(q, v, qk, kk, rho, tau, valid)
        else:
            feat = self._legacy_features(q, v, qk, kk, rho, tau, valid, d)

        feat = feat.transpose(1, 2).reshape(b, -1, h, w)
        out = self.proj(feat.to(self.proj.weight.dtype))
        # No external context -> no correction, including projection bias.
        out = out * valid.any(-1).view(b, 1, 1, 1).to(out.dtype)
        return F.interpolate(out, size=out_hw, mode='bilinear', align_corners=False)

    def inactive(self, x):
        # Preserve DDP participation for inactive 16/32 patch branches.
        zero = sum(p.reshape(-1)[0] * 0 for p in self.parameters())
        return x + zero.to(x.dtype)
