# Copyright (c) 2026.
"""A0 efficient convolutional backbone for PaDIS-MRI.

The encoder/decoder topology deliberately mirrors ``SongUNet``.  Only plain
residual computation blocks are replaced.  Resampling blocks and the forced
bottleneck attention block remain the original ``UNetBlock`` implementation.
"""

import numpy as np
import torch

from torch.nn.functional import silu
from torch_utils import persistence

from training.networks import (
    Conv2d,
    FourierEmbedding,
    GroupNorm,
    Linear,
    PositionalEmbedding,
    UNetBlock,
    weight_init,
)


@persistence.persistent_class
class DepthwiseConv2d(torch.nn.Module):
    """Depthwise convolution using the initialization conventions of SongUNet."""

    def __init__(self, channels, kernel=5, bias=True, init_mode='xavier_uniform', init_weight=1, init_bias=0):
        super().__init__()
        assert kernel % 2 == 1
        self.in_channels = channels
        self.out_channels = channels
        self.kernel = kernel
        init_kwargs = dict(mode=init_mode, fan_in=kernel * kernel, fan_out=kernel * kernel)
        self.weight = torch.nn.Parameter(
            weight_init([channels, 1, kernel, kernel], **init_kwargs) * init_weight
        )
        self.bias = torch.nn.Parameter(
            weight_init([channels], **init_kwargs) * init_bias
        ) if bias else None

    def forward(self, x):
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.conv2d(
            x,
            weight,
            bias=bias,
            padding=self.kernel // 2,
            groups=self.in_channels,
        )


@persistence.persistent_class
class EfficientDiffusionBlock(torch.nn.Module):
    """Expansion--depthwise spatial mixing block with additive sigma conditioning.

    The conditioning and residual conventions match the C96 SongUNet:
    additive embedding, two GroupNorm layers, near-zero final projection, and
    ``sqrt(0.5)`` residual scaling.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        expansion_ratio=2,
        spatial_kernel=5,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        init=dict(),
        init_zero=dict(init_weight=0),
    ):
        super().__init__()
        assert expansion_ratio >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.expansion_ratio = expansion_ratio
        self.hidden_channels = out_channels * expansion_ratio
        self.spatial_kernel = spatial_kernel
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.num_heads = 0

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.expand = Conv2d(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            kernel=1,
            **init,
        )
        self.spatial = DepthwiseConv2d(
            channels=self.hidden_channels,
            kernel=spatial_kernel,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=self.hidden_channels,
            **init,
        )
        self.norm1 = GroupNorm(num_channels=self.hidden_channels, eps=eps)
        self.project = Conv2d(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            kernel=1,
            **init_zero,
        )
        self.skip = None
        if in_channels != out_channels:
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=1,
                **init,
            )

    def forward(self, x, emb):
        residual = x
        x = self.expand(silu(self.norm0(x)))
        x = self.spatial(x)
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        x = silu(self.norm1(x.add_(params)))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.project(x)
        x = x.add_(self.skip(residual) if self.skip is not None else residual)
        return x * self.skip_scale


@persistence.persistent_class
class EfficientSongUNet(torch.nn.Module):
    """A0 backbone: SongUNet C96 topology with efficient plain blocks."""

    def __init__(
        self,
        img_resolution,
        in_channels,
        out_channels,
        label_dim=0,
        augment_dim=0,
        model_channels=128,
        channel_mult=[1, 2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=0.10,
        label_dropout=0,
        embedding_type='positional',
        channel_mult_noise=1,
        encoder_type='standard',
        decoder_type='standard',
        resample_filter=[1, 1],
        implicit_mlp=False,
        efficient_expansion=2,
        efficient_kernel=5,
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type == 'standard'
        assert decoder_type == 'standard'
        assert not implicit_mlp
        super().__init__()

        self.label_dropout = label_dropout
        self.efficient_expansion = efficient_expansion
        self.efficient_kernel = efficient_kernel
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        original_block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )
        efficient_block_kwargs = dict(
            emb_channels=emb_channels,
            expansion_ratio=efficient_expansion,
            spatial_kernel=efficient_kernel,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            init=init,
            init_zero=init_zero,
        )

        self.map_noise = PositionalEmbedding(
            num_channels=noise_channels,
            endpoint=True,
        ) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder: preserve the stem, four blocks per level, and original downsampling blocks.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(
                    in_channels=cin,
                    out_channels=cout,
                    kernel=3,
                    **init,
                )
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    down=True,
                    **original_block_kwargs,
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                if attn:
                    block = UNetBlock(
                        in_channels=cin, out_channels=cout, attention=True, **original_block_kwargs
                    )
                else:
                    block = EfficientDiffusionBlock(
                        in_channels=cin, out_channels=cout, **efficient_block_kwargs
                    )
                self.enc[f'{res}x{res}_block{idx}'] = block
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder: retain forced bottleneck attention and original upsampling blocks.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    attention=True,
                    **original_block_kwargs,
                )
                self.dec[f'{res}x{res}_in1'] = EfficientDiffusionBlock(
                    in_channels=cout,
                    out_channels=cout,
                    **efficient_block_kwargs,
                )
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    up=True,
                    **original_block_kwargs,
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                if attn:
                    block = UNetBlock(
                        in_channels=cin, out_channels=cout, attention=True, **original_block_kwargs
                    )
                else:
                    block = EfficientDiffusionBlock(
                        in_channels=cin, out_channels=cout, **efficient_block_kwargs
                    )
                self.dec[f'{res}x{res}_block{idx}'] = block
            if level == 0:
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(
                    in_channels=cout,
                    out_channels=out_channels,
                    kernel=3,
                    **init_zero,
                )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, (UNetBlock, EfficientDiffusionBlock)) else block(x)
            skips.append(x)

        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

