#!/usr/bin/env python3
"""Sigma-conditioning and real PaDIS training-path tests for V2-Sigma+."""

import argparse
import json
import math
import pickle
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'train' / 'padis-mri'))

from test_nc_restormer_v2 import (  # noqa: E402
    BATCH_MULTIPLIERS,
    grouped_gradient_stats,
    make_opa_loss,
    memory_stats,
)
from training.networks import Patch_EDMPrecond  # noqa: E402
from training.noise_conditioned_restormer_v2 import NoiseConditionedRestormerV2  # noqa: E402
from training.noise_conditioned_restormer_v2_sigma import (  # noqa: E402
    EDM2NoiseEmbedding,
    NoiseConditionedRestormerV2Sigma,
)


SIGMAS = [0.003, 0.03, 0.1, 0.3, 1.0, 3.0]
MODEL_KWARGS = dict(
    dim=80,
    decoder_level1_dim=96,
    num_blocks=[2, 3, 4],
    num_decoder_blocks=[3, 2],
    num_refinement_blocks=2,
    heads=[2, 4, 8],
    decoder_level1_heads=2,
    ffn_expansion_factor=2.0,
    bias=False,
    layer_norm_type='WithBias',
    dropout=0.05,
    condition_dim=320,
    noise_embedding_size=256,
    gate_bias=0.1,
    output_init_std=1e-3,
    modulation_init_std=0.01,
    fourier_bandwidth=1.0,
    fourier_seed=2024,
)


def make_backbone(device, activation_checkpointing=False):
    return NoiseConditionedRestormerV2Sigma(
        img_resolution=512,
        in_channels=4,
        out_channels=2,
        activation_checkpointing=activation_checkpointing,
        **MODEL_KWARGS,
    ).to(device)


def make_precond(device, activation_checkpointing=False):
    return Patch_EDMPrecond(
        img_resolution=512,
        img_channels=4,
        out_channels=2,
        model_type='NoiseConditionedRestormerV2Sigma',
        activation_checkpointing=activation_checkpointing,
        **MODEL_KWARGS,
    ).to(device)


def relative_l2_matrix(tensors):
    rows = []
    for first in tensors:
        row = []
        for second in tensors:
            denominator = 0.5 * (first.norm() + second.norm()).clamp_min(1e-12)
            row.append(float((first - second).norm() / denominator))
        rows.append(row)
    return rows


def fourier_rng_and_buffer_test():
    mlp = nn.Sequential(nn.Linear(256, 320), nn.SiLU(), nn.Linear(320, 320))
    state_before = torch.random.get_rng_state().clone()
    embedding = EDM2NoiseEmbedding(mlp=mlp, fourier_seed=2024)
    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)
    buffers = dict(embedding.named_buffers())
    assert set(buffers) == {'frequencies', 'phases'}
    assert buffers['frequencies'].shape == (256,)
    assert buffers['phases'].shape == (256,)
    return {
        'fourier_generation_preserves_main_rng_state': True,
        'registered_buffers': {name: list(value.shape) for name, value in buffers.items()},
        'frequency_std': float(buffers['frequencies'].std()),
        'phase_min': float(buffers['phases'].min()),
        'phase_max': float(buffers['phases'].max()),
        'fourier_seed': embedding.fourier_seed,
        'bandwidth': embedding.bandwidth,
    }



def complete_model_initialization_rng_test():
    """Compare RNG states after otherwise matched V2 and Sigma+ creation.

    The Sigma+-specific Fourier features and modulation weights must not
    advance either the global CPU or any CUDA generator beyond V2.
    """
    v2_kwargs = {
        key: value for key, value in MODEL_KWARGS.items()
        if key not in ('modulation_init_std', 'fourier_bandwidth', 'fourier_seed')
    }

    def construct_and_capture(model_cls, kwargs):
        torch.manual_seed(123)
        model = model_cls(
            img_resolution=512,
            in_channels=4,
            out_channels=2,
            **kwargs,
        )  # Construct on CPU exactly as the training loop does.
        cpu_state = torch.random.get_rng_state().clone()
        cuda_states = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        )
        del model
        return cpu_state, cuda_states

    v2_cpu, v2_cuda = construct_and_capture(NoiseConditionedRestormerV2, v2_kwargs)
    sigma_cpu, sigma_cuda = construct_and_capture(
        NoiseConditionedRestormerV2Sigma, MODEL_KWARGS
    )
    assert torch.equal(v2_cpu, sigma_cpu), 'Sigma+ advanced the global CPU RNG'
    assert len(v2_cuda) == len(sigma_cuda)
    assert all(torch.equal(a, b) for a, b in zip(v2_cuda, sigma_cuda)), (
        'Sigma+ advanced a global CUDA RNG'
    )
    return {
        'v2_and_sigma_global_cpu_rng_equal': True,
        'v2_and_sigma_global_cuda_rng_equal': True,
        'cuda_devices_checked': len(v2_cuda),
        'seed': 123,
        'modulation_private_seed': MODEL_KWARGS['fourier_seed'] + 1,
    }


def stage_modulation_responses(model, conditions):
    stages = {
        'encoder_level1': model.encoder_level1,
        'encoder_level2': model.encoder_level2,
        'encoder_level3': model.encoder_level3,
        'decoder_level2': model.decoder_level2,
        'decoder_level1': model.decoder_level1,
        'refinement': model.refinement,
    }
    results = {}
    for stage_name, blocks in stages.items():
        per_sigma = []
        for sigma_index, sigma in enumerate(SIGMAS):
            values = {
                'attention_shift': [], 'attention_scale': [], 'attention_gate': [],
                'gdfn_shift': [], 'gdfn_scale': [], 'gdfn_gate': [],
            }
            for block in blocks:
                chunks = block.adaLN_modulation(conditions[sigma_index:sigma_index + 1]).chunk(6, dim=1)
                for key, chunk in zip(values, chunks):
                    values[key].append(chunk.detach().reshape(-1))
            amplitudes = {}
            for key, tensors in values.items():
                vector = torch.cat(tensors)
                amplitudes[f'{key}_mean_abs'] = float(vector.abs().mean())
                amplitudes[f'{key}_rms'] = float(vector.square().mean().sqrt())
            per_sigma.append({'sigma': sigma, **amplitudes})
        results[stage_name] = per_sigma
    return results


def sigma_conditioning_test(device):
    torch.manual_seed(123)
    sigma_model = make_backbone(device).eval()
    model_input = torch.randn(1, 4, 64, 64, device=device)
    noise_labels = torch.tensor(SIGMAS, device=device).log() / 4
    with torch.no_grad():
        fourier_features = [
            value.detach() for value in sigma_model.noise_embedding.fourier_features(noise_labels)
        ]
        embeddings = [value.detach() for value in sigma_model.noise_embedding(noise_labels)]
        sigma_outputs = [
            sigma_model(model_input, noise_labels[index:index + 1]).detach()
            for index in range(len(SIGMAS))
        ]
        responses = stage_modulation_responses(
            sigma_model,
            sigma_model.noise_embedding(noise_labels),
        )

    torch.manual_seed(123)
    v2_control = NoiseConditionedRestormerV2(
        img_resolution=512,
        in_channels=4,
        out_channels=2,
        dim=80,
        decoder_level1_dim=96,
        num_blocks=[2, 3, 4],
        num_decoder_blocks=[3, 2],
        num_refinement_blocks=2,
        heads=[2, 4, 8],
        decoder_level1_heads=2,
        ffn_expansion_factor=2.0,
        dropout=0.05,
        condition_dim=320,
        noise_embedding_size=256,
        gate_bias=0.1,
        output_init_std=1e-3,
    ).to(device).eval()
    with torch.no_grad():
        v2_outputs = [
            v2_control(model_input, noise_labels[index:index + 1]).detach()
            for index in range(len(SIGMAS))
        ]

    embedding_matrix = relative_l2_matrix(embeddings)
    output_matrix = relative_l2_matrix(sigma_outputs)
    v2_output_matrix = relative_l2_matrix(v2_outputs)
    assert max(max(row) for row in embedding_matrix) > 0
    assert max(max(row) for row in output_matrix) > 0
    assert max(max(row) for row in v2_output_matrix) == 0
    result = {
        'sigmas': SIGMAS,
        'c_noise': [math.log(sigma) / 4 for sigma in SIGMAS],
        'fourier_feature_relative_l2': relative_l2_matrix(fourier_features),
        'embedding_relative_l2': embedding_matrix,
        'backbone_output_relative_l2': output_matrix,
        'v2_control_backbone_output_relative_l2': v2_output_matrix,
        'stage_modulation_amplitudes': responses,
        'max_embedding_relative_l2': max(max(row) for row in embedding_matrix),
        'max_backbone_output_relative_l2': max(max(row) for row in output_matrix),
        'max_v2_control_output_relative_l2': max(max(row) for row in v2_output_matrix),
    }
    del sigma_model, v2_control, model_input
    return result


def gradient_test(device, activation_checkpointing):
    net = make_precond(device, activation_checkpointing).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    noisy = torch.randn(6, 2, 16, 16, device=device)
    position = torch.randn(6, 2, 16, 16, device=device)
    sigma = torch.tensor(SIGMAS, device=device).reshape(6, 1, 1, 1)
    target = torch.randn_like(noisy)
    fixed_backbone_input = torch.randn(1, 4, 16, 16, device=device)
    fixed_noise_labels = torch.tensor(SIGMAS, device=device).log() / 4
    steps = []
    for step_index in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = net(noisy, sigma, x_pos=position)
        loss = (output - target).square().mean()
        loss.backward()
        groups = grouped_gradient_stats(net)
        assert all(group['all_gradients_present_and_finite'] for group in groups.values())
        assert groups['noise_embedding']['nonzero_element_ratio'] > 0
        for required in ('encoder_level1', 'encoder_level2', 'encoder_level3',
                         'decoder_level2', 'decoder_level1', 'refinement', 'mdta_all', 'gdfn_all'):
            assert groups[required]['nonzero_element_ratio'] > 0
        optimizer.step()
        net.model.eval()
        with torch.no_grad():
            post_update_outputs = [
                net.model(fixed_backbone_input, fixed_noise_labels[index:index + 1]).detach()
                for index in range(len(SIGMAS))
            ]
        net.model.train()
        post_update_matrix = relative_l2_matrix(post_update_outputs)
        steps.append({
            'step': step_index + 1,
            'loss': float(loss.detach()),
            'groups': groups,
            'post_update_backbone_output_relative_l2': post_update_matrix,
            'post_update_max_backbone_output_relative_l2': max(
                max(row) for row in post_update_matrix
            ),
        })
    result = {
        'steps': steps,
        'first_step_noise_embedding_nonzero_gradient_ratio': (
            steps[0]['groups']['noise_embedding']['nonzero_element_ratio']
        ),
        'all_gradients_finite': True,
    }
    del net, optimizer, noisy, position, sigma, target, fixed_backbone_input
    return result


def real_patch_training_test(device, patch_size, activation_checkpointing, lightweight=False):
    source_batch = 1 if lightweight else BATCH_MULTIPLIERS[patch_size]
    expected_network_batch = source_batch * (2 if patch_size == 64 else 1)
    net = make_precond(device, activation_checkpointing).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = make_opa_loss()
    images = torch.randn(source_batch, 2, 512, 512, device=device)
    observed_inputs = []

    def capture_input(_module, inputs):
        observed_inputs.append(list(inputs[0].shape))

    hook = net.model.register_forward_pre_hook(capture_input)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    loss_map = loss_fn(
        net=net,
        images=images,
        patch_size=patch_size,
        resolution=512,
        labels=None,
        augment_pipe=None,
    )
    loss = loss_map.sum() / source_batch
    loss.backward()
    stats = grouped_gradient_stats(net)
    assert all(group['all_gradients_present_and_finite'] for group in stats.values())
    optimizer.step()
    hook.remove()
    assert observed_inputs[-1] == [expected_network_batch, 4, patch_size, patch_size]
    result = {
        'patch_size': patch_size,
        'source_images_shape': list(images.shape),
        'network_input_shape': observed_inputs[-1],
        'loss_map_shape': list(loss_map.shape),
        'loss': float(loss.detach()),
        'optimizer_step_completed': True,
        'all_gradients_finite': True,
        'memory': memory_stats(device),
    }
    del net, optimizer, loss_fn, images, loss_map, loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def snapshot_test():
    net = make_precond(torch.device('cpu')).eval()
    payload = {
        'ema': net,
        'loss_fn': make_opa_loss(),
        'augment_pipe': None,
        'dataset_kwargs': {},
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / 'network-snapshot-v2-sigma-test.pkl'
        with path.open('wb') as handle:
            pickle.dump(payload, handle)
        size = path.stat().st_size
        with path.open('rb') as handle:
            restored = pickle.load(handle)
    restored_net = restored['ema'].eval()
    buffers = dict(restored_net.model.noise_embedding.named_buffers())
    with torch.no_grad():
        output = restored_net(
            torch.randn(4, 2, 64, 64),
            torch.tensor(0.1),
            x_pos=torch.randn(4, 2, 64, 64),
        )
    assert tuple(output.shape) == (4, 2, 64, 64)
    assert set(buffers) == {'frequencies', 'phases'}
    return {
        'roundtrip': True,
        'restored_model_type': type(restored_net.model).__name__,
        'restored_fourier_buffers': {name: list(value.shape) for name, value in buffers.items()},
        'output_shape': list(output.shape),
        'pickle_bytes': size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--lightweight', action='store_true')
    parser.add_argument('--output', type=Path, default=THIS_DIR / 'nc_restormer_v2_sigma_test.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; GPU results were not produced')
        torch.cuda.set_device(device)
    torch.manual_seed(123)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(123)

    results = {
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch_version': torch.__version__,
        'dtype': 'FP32',
        'real_training_batches': not args.lightweight,
        'batch_multipliers': BATCH_MULTIPLIERS,
        'fourier_rng_and_buffers': fourier_rng_and_buffer_test(),
        'complete_model_initialization_rng': complete_model_initialization_rng_test(),
        'sigma_conditioning': sigma_conditioning_test(device),
        'gradient_test': gradient_test(device, args.activation_checkpointing),
        'patch_training_paths': {
            str(patch): real_patch_training_test(
                device, patch, args.activation_checkpointing, args.lightweight,
            )
            for patch in (16, 32, 64)
        },
        'snapshot': snapshot_test(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
