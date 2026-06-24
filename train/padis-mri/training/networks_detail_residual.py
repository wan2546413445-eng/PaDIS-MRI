"""PaDIS-MRI network wrapper with a multi-scale gated detail residual head."""

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from torch_utils import persistence
from training.networks import Patch_EDMPrecond
from training.detail_residual_head import MultiScaleGatedDetailResidualHead


def _parse_int_tuple(x):
    if isinstance(x, str):
        return tuple(int(v) for v in x.split(',') if v.strip())
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    return (int(x),)


@persistence.persistent_class
class Patch_EDMPrecond_DetailResidual(Patch_EDMPrecond):
    """Patch_EDMPrecond plus an output-side bounded residual detail prior.

    By default forward() returns only D_final, so existing evaluation code can
    load and call this network in the same way as the original PaDIS model.
    During training, losses may pass return_detail=True to obtain
    (D_final, D_base, r_detail, gate).
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        out_channels,
        label_dim,
        detail_hidden=48,
        detail_eta=0.15,
        detail_dilations=(1, 2, 5),
        detail_use_pos=True,
        detail_gate_bias=-1.0,
        detail_init_scale=1e-3,
        detail_detach_base=True,
        **kwargs,
    ):
        super().__init__(
            img_resolution=img_resolution,
            img_channels=img_channels,
            out_channels=out_channels,
            label_dim=label_dim,
            **kwargs,
        )
        self.detail_detach_base = bool(detail_detach_base)
        self.detail_head = MultiScaleGatedDetailResidualHead(
            data_channels=out_channels,
            pos_channels=2,
            hidden_channels=int(detail_hidden),
            eta=float(detail_eta),
            dilations=_parse_int_tuple(detail_dilations),
            use_pos=bool(detail_use_pos),
            gate_bias=float(detail_gate_bias),
            init_scale=float(detail_init_scale),
        )

    def forward(
        self,
        x,
        sigma,
        x_pos=None,
        class_labels=None,
        force_fp32=False,
        return_detail=False,
        detail_enable=True,
        **model_kwargs,
    ):
        D_base = super().forward(
            x=x,
            sigma=sigma,
            x_pos=x_pos,
            class_labels=class_labels,
            force_fp32=force_fp32,
            **model_kwargs,
        )

        if not detail_enable:
            if return_detail:
                zero = torch.zeros_like(D_base)
                gate = torch.zeros_like(D_base)
                return D_base, D_base, zero, gate
            return D_base

        base_for_head = D_base.detach() if self.detail_detach_base else D_base
        r_detail, gate, _ = self.detail_head(x.to(torch.float32), base_for_head.to(torch.float32), x_pos=x_pos)
        D_final = D_base + r_detail.to(D_base.dtype)

        if return_detail:
            return D_final, D_base, r_detail, gate
        return D_final
