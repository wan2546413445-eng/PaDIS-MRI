"""Noise-conditioned Restormer backbone for PaDIS-MRI.

The MDTA, GDFN, encoder-decoder, and refinement design follows the official
Restormer implementation (Zamir et al., CVPR 2022, MIT license).  The
per-block adaptive normalization and zero-initialized residual gates follow
the adaLN-Zero design in DiT (Peebles and Xie, ICCV 2023, CC BY-NC 4.0).
This implementation is adapted to PaDIS-MRI's four-channel patch input,
two-channel complex output, continuous EDM noise label, and variable patch
sizes.  It intentionally does not add Restormer's input-image residual because
Patch_EDMPrecond applies the EDM skip connection outside the backbone.
"""

import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from torch_utils import persistence


_CHECKPOINT_SUPPORTS_USE_REENTRANT = 'use_reentrant' in inspect.signature(checkpoint).parameters


def _modulate(x, shift, scale):
    return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


@persistence.persistent_class
class BiasFreeLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(variance + 1e-5) * self.weight[None, :, None, None]


@persistence.persistent_class
class WithBiasLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        normalized = (x - mean) * torch.rsqrt(variance + 1e-5)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


@persistence.persistent_class
class LayerNorm2d(nn.Module):
    def __init__(self, dim, layer_norm_type='WithBias'):
        super().__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        elif layer_norm_type == 'WithBias':
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError(f'Unsupported LayerNorm type: {layer_norm_type}')

    def forward(self, x):
        return self.body(x)


@persistence.persistent_class
class NoiseEmbedding(nn.Module):
    """Continuous sinusoidal embedding for EDM's log(sigma)/4 label."""

    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.max_period = int(max_period)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, noise_labels):
        noise_labels = noise_labels.reshape(-1).to(torch.float32)
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=noise_labels.device, dtype=torch.float32)
            / half
        )
        arguments = noise_labels[:, None] * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return self.mlp(embedding)


@persistence.persistent_class
class GatedDconvFeedForward(nn.Module):
    """Restormer GDFN."""

    def __init__(self, dim, expansion_factor=2.0, bias=False):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


@persistence.persistent_class
class MultiDconvHeadTransposedAttention(nn.Module):
    """Restormer MDTA with attention across per-head channel features."""

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'Channel dimension {dim} must be divisible by {num_heads} heads')
        self.num_heads = int(num_heads)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        channels_per_head = channels // self.num_heads
        q = q.reshape(batch, self.num_heads, channels_per_head, height * width)
        k = k.reshape(batch, self.num_heads, channels_per_head, height * width)
        v = v.reshape(batch, self.num_heads, channels_per_head, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        output = (attention @ v).reshape(batch, channels, height, width)
        return self.project_out(output)


@persistence.persistent_class
class NoiseConditionedTransformerBlock(nn.Module):
    """Restormer block modulated by DiT-style adaLN-Zero conditioning."""

    def __init__(
        self,
        dim,
        num_heads,
        condition_dim,
        ffn_expansion_factor=2.0,
        bias=False,
        layer_norm_type='WithBias',
        dropout=0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attention = MultiDconvHeadTransposedAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.feed_forward = GatedDconvFeedForward(dim, ffn_expansion_factor, bias)
        self.residual_dropout = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

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
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


@persistence.persistent_class
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


@persistence.persistent_class
class NoiseConditionedRestormer(nn.Module):
    """Three-scale Restormer denoiser conditioned in every Transformer block."""

    def __init__(
        self,
        img_resolution,
        in_channels=4,
        out_channels=2,
        label_dim=0,
        dim=80,
        num_blocks=(2, 3, 4),
        num_decoder_blocks=(3, 2),
        num_refinement_blocks=2,
        heads=(2, 4, 8),
        ffn_expansion_factor=2.0,
        bias=False,
        layer_norm_type='WithBias',
        dropout=0.0,
        condition_dim=320,
        noise_embedding_size=256,
        activation_checkpointing=False,
    ):
        super().__init__()
        if len(num_blocks) != 3 or len(num_decoder_blocks) != 2 or len(heads) != 3:
            raise ValueError('NoiseConditionedRestormer requires three encoder scales and two decoder stages')
        self.img_resolution = int(img_resolution)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.label_dim = int(label_dim)
        self.activation_checkpointing = bool(activation_checkpointing)

        self.noise_embedding = NoiseEmbedding(condition_dim, noise_embedding_size)
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        def make_blocks(channels, count, num_heads):
            return nn.ModuleList([
                NoiseConditionedTransformerBlock(
                    dim=channels,
                    num_heads=num_heads,
                    condition_dim=condition_dim,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                    dropout=dropout,
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
        # Restormer's first decoder level retains the concatenated 2*dim width.
        self.decoder_level1 = make_blocks(dim * 2, num_decoder_blocks[1], heads[0])
        self.refinement = make_blocks(dim * 2, num_refinement_blocks, heads[0])
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
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
        del class_labels, augment_labels  # PaDIS-MRI training is unconditional.
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
        decoder1 = self._run_blocks(self.decoder_level1, decoder1, condition)
        decoder1 = self._run_blocks(self.refinement, decoder1, condition)
        return self.output(decoder1)
