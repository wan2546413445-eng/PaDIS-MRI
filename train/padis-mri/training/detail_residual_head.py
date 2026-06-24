"""Lightweight multi-scale gated residual head for PaDIS-MRI detail correction.

This module is intentionally independent from networks.py so that the detail
prior can be edited without touching the original PaDIS backbone.
"""

import torch
from torch import nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise 3x3 convolution followed by pointwise 1x1 convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=in_channels, bias=True,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MultiScaleGatedDetailResidualHead(nn.Module):
    """Predict a bounded detail residual on top of the base PaDIS denoiser output.

    Inputs are concatenated as:
        noisy patch, base denoised patch, noisy-base residual,
        gradient magnitude of base, and optionally position maps.

    Output:
        r_detail = eta * sigmoid(gate) * tanh(r_raw)
    """

    def __init__(
        self,
        data_channels=2,
        pos_channels=2,
        hidden_channels=48,
        eta=0.15,
        dilations=(1, 2, 5),
        use_pos=True,
        gate_bias=-1.0,
        init_scale=1e-3,
    ):
        super().__init__()
        if isinstance(dilations, str):
            dilations = tuple(int(x) for x in dilations.split(',') if x.strip())
        dilations = tuple(int(x) for x in dilations)
        if len(dilations) == 0:
            raise ValueError('dilations must not be empty')
        if hidden_channels <= 0:
            raise ValueError('hidden_channels must be positive')

        self.data_channels = int(data_channels)
        self.pos_channels = int(pos_channels)
        self.hidden_channels = int(hidden_channels)
        self.eta = float(eta)
        self.dilations = dilations
        self.use_pos = bool(use_pos)

        # x, base, x-base, grad(base) = 4 * data_channels, plus position maps.
        in_channels = 4 * self.data_channels + (self.pos_channels if self.use_pos else 0)
        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        self.local_branch = nn.Sequential(
            SeparableConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, dilation=1),
            nn.SiLU(inplace=True),
            SeparableConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, dilation=1),
            nn.SiLU(inplace=True),
        )

        self.dilated_branches = nn.ModuleList([
            nn.Sequential(
                SeparableConv2d(
                    hidden_channels, hidden_channels,
                    kernel_size=3, padding=d, dilation=d,
                ),
                nn.SiLU(inplace=True),
            )
            for d in self.dilations
        ])

        # A direct edge/detail branch from raw inputs keeps high-frequency cues visible.
        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            SeparableConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        fuse_channels = hidden_channels * (2 + len(self.dilated_branches))
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            SeparableConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        self.residual_head = nn.Conv2d(hidden_channels, data_channels, kernel_size=3, padding=1)
        self.gate_head = nn.Conv2d(hidden_channels, data_channels, kernel_size=3, padding=1)
        self._init_heads(init_scale=init_scale, gate_bias=gate_bias)

    def _init_heads(self, init_scale=1e-3, gate_bias=-1.0):
        for layer in [self.residual_head, self.gate_head]:
            nn.init.normal_(layer.weight, mean=0.0, std=float(init_scale))
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))

    @staticmethod
    def _gradient_magnitude(x):
        # Same spatial size as x. Per-channel gradient magnitude is retained.
        dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-12)

    def forward(self, noisy, base, x_pos=None):
        if noisy.shape != base.shape:
            raise ValueError(f'noisy and base must have the same shape, got {noisy.shape} and {base.shape}')
        if noisy.shape[1] != self.data_channels:
            raise ValueError(f'expected {self.data_channels} data channels, got {noisy.shape[1]}')

        grad_base = self._gradient_magnitude(base)
        inputs = [noisy, base, noisy - base, grad_base]

        if self.use_pos:
            if x_pos is None:
                pos = noisy.new_zeros([noisy.shape[0], self.pos_channels, noisy.shape[2], noisy.shape[3]])
            else:
                if x_pos.shape[2:] != noisy.shape[2:]:
                    raise ValueError(f'x_pos spatial shape {x_pos.shape[2:]} does not match noisy {noisy.shape[2:]}')
                pos = x_pos[:, :self.pos_channels]
                if pos.shape[1] < self.pos_channels:
                    pad = noisy.new_zeros([noisy.shape[0], self.pos_channels - pos.shape[1], noisy.shape[2], noisy.shape[3]])
                    pos = torch.cat([pos, pad], dim=1)
            inputs.append(pos)

        h_in = torch.cat(inputs, dim=1)
        h = self.stem(h_in)
        features = [self.local_branch(h)]
        features += [branch(h) for branch in self.dilated_branches]
        features.append(self.edge_branch(h_in))
        fused = self.fuse(torch.cat(features, dim=1))

        r_raw = self.residual_head(fused)
        gate = torch.sigmoid(self.gate_head(fused))
        r_detail = self.eta * gate * torch.tanh(r_raw)
        return r_detail, gate, r_raw
