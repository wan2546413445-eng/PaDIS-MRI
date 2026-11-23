"""Sigma-enhanced Noise-Conditioned Restormer V2 for PaDIS-MRI.

The architecture is inherited unchanged from NoiseConditionedRestormerV2.
Sigma+ replaces its sinusoidal noise features with EDM2-style random Fourier
features and gives every adaLN modulation weight a small nonzero
initialization so that continuous log-sigma conditioning is active from the
first forward and backward pass.
"""

import math

import torch
from torch import nn

from torch_utils import persistence
from training.noise_conditioned_restormer_v2 import (
    NearZeroNoiseConditionedTransformerBlock,
    NoiseConditionedRestormerV2,
)


@persistence.persistent_class
class EDM2NoiseEmbedding(nn.Module):
    """EDM2-style magnitude-preserving random Fourier noise embedding."""

    def __init__(
        self,
        hidden_size=320,
        num_features=256,
        bandwidth=1.0,
        fourier_seed=2024,
        mlp=None,
    ):
        super().__init__()
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(fourier_seed))
        frequencies = torch.randn(num_features, generator=generator) * bandwidth
        phases = torch.rand(num_features, generator=generator) * (2 * math.pi)
        self.register_buffer('frequencies', frequencies)
        self.register_buffer('phases', phases)
        self.bandwidth = float(bandwidth)
        self.fourier_seed = int(fourier_seed)
        if mlp is None:
            self.mlp = nn.Sequential(
                nn.Linear(num_features, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            nn.init.normal_(self.mlp[0].weight, std=0.02)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.normal_(self.mlp[2].weight, std=0.02)
            nn.init.zeros_(self.mlp[2].bias)
        else:
            self.mlp = mlp

    def fourier_features(self, noise_labels):
        noise_labels = noise_labels.reshape(-1).to(torch.float32)
        arguments = (
            2 * math.pi * noise_labels[:, None] * self.frequencies[None].to(torch.float32)
            + self.phases[None].to(torch.float32)
        )
        return math.sqrt(2) * torch.cos(arguments)

    def forward(self, noise_labels):
        features = self.fourier_features(noise_labels)
        return self.mlp(features).to(noise_labels.dtype)


@persistence.persistent_class
class NoiseConditionedRestormerV2Sigma(NoiseConditionedRestormerV2):
    """V2 architecture with EDM2 Fourier sigma features and active adaLN."""

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
        modulation_init_std=0.01,
        fourier_bandwidth=1.0,
        fourier_seed=2024,
        activation_checkpointing=False,
    ):
        super().__init__(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            label_dim=label_dim,
            dim=dim,
            decoder_level1_dim=decoder_level1_dim,
            num_blocks=num_blocks,
            num_decoder_blocks=num_decoder_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            decoder_level1_heads=decoder_level1_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
            dropout=dropout,
            condition_dim=condition_dim,
            noise_embedding_size=noise_embedding_size,
            gate_bias=gate_bias,
            output_init_std=output_init_std,
            activation_checkpointing=activation_checkpointing,
        )

        inherited_mlp = self.noise_embedding.mlp
        self.noise_embedding = EDM2NoiseEmbedding(
            hidden_size=condition_dim,
            num_features=noise_embedding_size,
            bandwidth=fourier_bandwidth,
            fourier_seed=fourier_seed,
            mlp=inherited_mlp,
        )
        self.modulation_init_std = float(modulation_init_std)
        # Keep V2 and Sigma+ training RNG streams aligned after model creation.
        # Use a private CPU generator rather than torch.manual_seed/fork_rng:
        # the latter can also affect CUDA generators when CUDA is initialized.
        modulation_generator = torch.Generator(device='cpu')
        modulation_generator.manual_seed(int(fourier_seed) + 1)
        for module in self.modules():
            if not isinstance(module, NearZeroNoiseConditionedTransformerBlock):
                continue
            modulation = module.adaLN_modulation[-1]
            block_dim = module.dim
            with torch.no_grad():
                # Generate on CPU explicitly so this initialization never
                # consumes either the global CPU or CUDA RNG state.
                new_weight = torch.empty(
                    modulation.weight.shape,
                    dtype=modulation.weight.dtype,
                    device='cpu',
                ).normal_(mean=0.0, std=modulation_init_std,
                          generator=modulation_generator)
                modulation.weight.copy_(new_weight.to(modulation.weight.device))
                modulation.bias.zero_()
                modulation.bias[2 * block_dim:3 * block_dim].fill_(gate_bias)
                modulation.bias[5 * block_dim:6 * block_dim].fill_(gate_bias)
