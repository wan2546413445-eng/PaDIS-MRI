"""Noise-Conditioned Restormer V2 for PaDIS-MRI.

V2 preserves the V1 Restormer/EDM design while reducing the full-resolution
decoder and refinement width from 160 to 96 channels.  It also replaces strict
adaLN-Zero plus a zero output head with a PaDIS-specific near-zero gated
initialization so that feature extractors receive gradients from the first
optimization step.
"""

import inspect

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torch_utils import persistence
from training.noise_conditioned_restormer import (
    Downsample,
    GatedDconvFeedForward,
    LayerNorm2d,
    MultiDconvHeadTransposedAttention,
    NoiseEmbedding,
    Upsample,
    _modulate,
)


_CHECKPOINT_SUPPORTS_USE_REENTRANT = 'use_reentrant' in inspect.signature(checkpoint).parameters


@persistence.persistent_class
class NearZeroNoiseConditionedTransformerBlock(nn.Module):
    """Restormer block with PaDIS-specific near-zero noise conditioning."""

    def __init__(
        self,
        dim,
        num_heads,
        condition_dim,
        ffn_expansion_factor=2.0,
        bias=False,
        layer_norm_type='WithBias',
        dropout=0.0,
        gate_bias=0.1,
    ):
        super().__init__()
        self.dim = int(dim)
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attention = MultiDconvHeadTransposedAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.feed_forward = GatedDconvFeedForward(dim, ffn_expansion_factor, bias)
        self.residual_dropout = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * dim),
        )

        modulation = self.adaLN_modulation[-1]
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)
        with torch.no_grad():
            modulation.bias[2 * dim:3 * dim].fill_(gate_bias)
            modulation.bias[5 * dim:6 * dim].fill_(gate_bias)

    def forward(self, x, condition):
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        attention_input = _modulate(self.norm1(x), shift_attn, scale_attn)
        x = x + gate_attn[:, :, None, None] * self.residual_dropout(
            self.attention(attention_input)
        )
        feed_forward_input = _modulate(self.norm2(x), shift_ffn, scale_ffn)
        x = x + gate_ffn[:, :, None, None] * self.residual_dropout(
            self.feed_forward(feed_forward_input)
        )
        return x


@persistence.persistent_class
class NoiseConditionedRestormerV2(nn.Module):
    """Three-scale NC-Restormer with a 96-channel full-resolution decoder."""

    def __init__(
        self,
        img_resolution,
        in_channels=4,
        out_channels=2,
        label_dim=0,
        dim=80,
        decoder_level1_dim=96,
        num_blocks=(2, 3, 4),
        num_decoder_blocks=(3, 2),
        num_refinement_blocks=2,
        heads=(2, 4, 8),
        decoder_level1_heads=2,
        ffn_expansion_factor=2.0,
        bias=False,
        layer_norm_type='WithBias',
        dropout=0.0,
        condition_dim=320,
        noise_embedding_size=256,
        gate_bias=0.1,
        output_init_std=1e-3,
        activation_checkpointing=False,
    ):
        super().__init__()
        if len(num_blocks) != 3 or len(num_decoder_blocks) != 2 or len(heads) != 3:
            raise ValueError('NoiseConditionedRestormerV2 requires three encoder scales and two decoder stages')
        if decoder_level1_dim % decoder_level1_heads != 0:
            raise ValueError('decoder_level1_dim must be divisible by decoder_level1_heads')

        self.img_resolution = int(img_resolution)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.label_dim = int(label_dim)
        self.decoder_level1_dim = int(decoder_level1_dim)
        self.activation_checkpointing = bool(activation_checkpointing)

        self.noise_embedding = NoiseEmbedding(condition_dim, noise_embedding_size)
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        def make_blocks(channels, count, num_heads):
            return nn.ModuleList([
                NearZeroNoiseConditionedTransformerBlock(
                    dim=channels,
                    num_heads=num_heads,
                    condition_dim=condition_dim,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    dropout=dropout,
                    gate_bias=gate_bias,
                )
                for _ in range(count)
            ])

        self.encoder_level1 = make_blocks(dim, num_blocks[0], heads[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = make_blocks(dim * 2, num_blocks[1], heads[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = make_blocks(dim * 4, num_blocks[2], heads[2])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_channels_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = make_blocks(dim * 2, num_decoder_blocks[0], heads[1])

        self.up2_1 = Upsample(dim * 2)
        self.reduce_channels_level1 = nn.Conv2d(
            dim * 2,
            decoder_level1_dim,
            kernel_size=1,
            bias=bias,
        )
        self.decoder_level1 = make_blocks(
            decoder_level1_dim,
            num_decoder_blocks[1],
            decoder_level1_heads,
        )
        self.refinement = make_blocks(
            decoder_level1_dim,
            num_refinement_blocks,
            decoder_level1_heads,
        )
        self.output = nn.Conv2d(
            decoder_level1_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        nn.init.normal_(self.output.weight, mean=0.0, std=output_init_std)
        nn.init.zeros_(self.output.bias)

    def _run_blocks(self, blocks, x, condition):
        for block in blocks:
            if self.activation_checkpointing and self.training and x.requires_grad:
                if _CHECKPOINT_SUPPORTS_USE_REENTRANT:
                    x = checkpoint(block, x, condition, use_reentrant=False)
                else:
                    x = checkpoint(block, x, condition)
            else:
                x = block(x, condition)
        return x

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        del class_labels, augment_labels
        if x.shape[-2] % 4 != 0 or x.shape[-1] % 4 != 0:
            raise ValueError('Patch height and width must be divisible by four')

        condition = self.noise_embedding(noise_labels)
        encoder1 = self._run_blocks(self.encoder_level1, self.patch_embed(x), condition)
        encoder2 = self._run_blocks(self.encoder_level2, self.down1_2(encoder1), condition)
        latent = self._run_blocks(self.encoder_level3, self.down2_3(encoder2), condition)

        decoder2 = self.up3_2(latent)
        decoder2 = self.reduce_channels_level2(torch.cat([decoder2, encoder2], dim=1))
        decoder2 = self._run_blocks(self.decoder_level2, decoder2, condition)

        decoder1 = torch.cat([self.up2_1(decoder2), encoder1], dim=1)
        decoder1 = self.reduce_channels_level1(decoder1)
        decoder1 = self._run_blocks(self.decoder_level1, decoder1, condition)
        decoder1 = self._run_blocks(self.refinement, decoder1, condition)
        return self.output(decoder1)
