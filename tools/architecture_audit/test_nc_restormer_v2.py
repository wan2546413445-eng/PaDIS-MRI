#!/usr/bin/env python3
"""Correctness and real-batch training-path tests for NC-Restormer V2."""

import argparse
import json
import pickle
import sys
import tempfile
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'train' / 'padis-mri'))

from training.networks import Patch_EDMPrecond  # noqa: E402
from training.noise_conditioned_restormer_v2 import (  # noqa: E402
    NearZeroNoiseConditionedTransformerBlock,
    NoiseConditionedRestormerV2,
)
from training.patch_overlap_independent_loss import IndependentNoiseOverlapPatch_EDMLoss  # noqa: E402


BATCH_MULTIPLIERS = {16: 64, 32: 32, 64: 16}
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
)


def make_backbone(device, activation_checkpointing=False):
    return NoiseConditionedRestormerV2(
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
        model_type='NoiseConditionedRestormerV2',
        activation_checkpointing=activation_checkpointing,
        **MODEL_KWARGS,
    ).to(device)


def make_opa_loss():
    return IndependentNoiseOverlapPatch_EDMLoss(
        lambda_overlap=1.25,
        active_patch_size=64,
        noise_mode='independent',
        alignment_mode='symmetric',
        pair_geometry='identical',
        spatial_weight_mode='uniform',
    )


def memory_stats(device):
    if device.type != 'cuda':
        return {'allocated_mib': None, 'reserved_mib': None, 'peak_allocated_mib': None}
    mib = 1024 ** 2
    return {
        'allocated_mib': torch.cuda.memory_allocated(device) / mib,
        'reserved_mib': torch.cuda.memory_reserved(device) / mib,
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / mib,
    }


def parameter_gradient_stats(named_parameters):
    total_elements = 0
    nonzero_elements = 0
    finite = True
    squared_norm = 0.0
    tensors = 0
    nonzero_tensors = 0
    for _, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        tensors += 1
        total_elements += parameter.numel()
        if parameter.grad is None:
            finite = False
            continue
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all())
        count = int(torch.count_nonzero(gradient).item())
        nonzero_elements += count
        nonzero_tensors += int(count > 0)
        squared_norm += float(gradient.to(torch.float64).square().sum())
    return {
        'parameter_tensors': tensors,
        'nonzero_gradient_tensors': nonzero_tensors,
        'nonzero_tensor_ratio': nonzero_tensors / tensors if tensors else 0.0,
        'parameter_elements': total_elements,
        'nonzero_gradient_elements': nonzero_elements,
        'nonzero_element_ratio': nonzero_elements / total_elements if total_elements else 0.0,
        'gradient_l2_norm': squared_norm ** 0.5,
        'all_gradients_present_and_finite': finite,
    }


def grouped_gradient_stats(net):
    named = list(net.named_parameters())
    selectors = {
        'noise_embedding': lambda name: name.startswith('model.noise_embedding'),
        'encoder_level1': lambda name: name.startswith('model.encoder_level1'),
        'encoder_level2': lambda name: name.startswith('model.encoder_level2'),
        'encoder_level3': lambda name: name.startswith('model.encoder_level3'),
        'decoder_level2': lambda name: name.startswith('model.decoder_level2'),
        'decoder_level1': lambda name: name.startswith('model.decoder_level1'),
        'refinement': lambda name: name.startswith('model.refinement'),
        'mdta_all': lambda name: '.attention.' in name,
        'gdfn_all': lambda name: '.feed_forward.' in name,
        'output': lambda name: name.startswith('model.output'),
        'all_trainable': lambda name: True,
    }
    return {
        group: parameter_gradient_stats((name, parameter) for name, parameter in named if selector(name))
        for group, selector in selectors.items()
    }


def architecture_and_initialization_test(device):
    model = make_backbone(device)
    blocks = [module for module in model.modules() if isinstance(module, NearZeroNoiseConditionedTransformerBlock)]
    gate_checks = []
    for block in blocks:
        dim = block.dim
        modulation = block.adaLN_modulation[-1]
        bias = modulation.bias.detach()
        gate_checks.append({
            'dim': dim,
            'weight_is_zero': bool(torch.count_nonzero(modulation.weight).item() == 0),
            'attention_shift_scale_bias_is_zero': bool(torch.count_nonzero(bias[:2 * dim]).item() == 0),
            'attention_gate_bias_mean': float(bias[2 * dim:3 * dim].mean()),
            'ffn_shift_scale_bias_is_zero': bool(torch.count_nonzero(bias[3 * dim:5 * dim]).item() == 0),
            'ffn_gate_bias_mean': float(bias[5 * dim:6 * dim].mean()),
        })
    assert len(blocks) == 16
    assert all(item['weight_is_zero'] for item in gate_checks)
    assert all(item['attention_shift_scale_bias_is_zero'] for item in gate_checks)
    assert all(item['ffn_shift_scale_bias_is_zero'] for item in gate_checks)
    assert all(abs(item['attention_gate_bias_mean'] - 0.1) < 1e-6 for item in gate_checks)
    assert all(abs(item['ffn_gate_bias_mean'] - 0.1) < 1e-6 for item in gate_checks)
    assert torch.count_nonzero(model.output.weight).item() > 0
    assert torch.count_nonzero(model.output.bias).item() == 0

    output_shapes = {}
    with torch.no_grad():
        for patch in (16, 32, 64):
            output = model(
                torch.randn(1, 4, patch, patch, device=device),
                torch.zeros(1, device=device),
            )
            assert tuple(output.shape) == (1, 2, patch, patch)
            assert torch.isfinite(output).all()
            output_shapes[str(patch)] = list(output.shape)

    result = {
        'parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'conditioned_blocks': len(blocks),
        'stage_widths': {
            'encoder1': 80,
            'encoder2': 160,
            'latent': 320,
            'decoder2': 160,
            'decoder1': 96,
            'refinement': 96,
        },
        'reduce_channels_level1_weight_shape': list(model.reduce_channels_level1.weight.shape),
        'output_weight_std': float(model.output.weight.detach().std()),
        'output_shapes': output_shapes,
        'near_zero_gate_checks': gate_checks,
    }
    del model
    return result


def gradient_and_sigma_test(device, activation_checkpointing):
    net = make_precond(device, activation_checkpointing).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    noisy = torch.randn(2, 2, 16, 16, device=device)
    position = torch.randn(2, 2, 16, 16, device=device)
    sigma = torch.tensor([0.01, 1.0], device=device).reshape(2, 1, 1, 1)
    target = torch.randn_like(noisy)
    steps = []
    for step_index in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = net(noisy, sigma, x_pos=position)
        loss = (output - target).square().mean()
        loss.backward()
        groups = grouped_gradient_stats(net)
        assert all(group['all_gradients_present_and_finite'] for group in groups.values())
        for required in ('encoder_level1', 'encoder_level2', 'encoder_level3',
                         'decoder_level2', 'decoder_level1', 'refinement', 'mdta_all', 'gdfn_all'):
            assert groups[required]['nonzero_element_ratio'] > 0
        steps.append({'step': step_index + 1, 'loss': float(loss.detach()), 'groups': groups})
        optimizer.step()

    net.model.eval()
    model_input = torch.cat([noisy[:1], position[:1]], dim=1)
    with torch.no_grad():
        output_low_sigma = net.model(model_input, torch.tensor([0.01], device=device).log() / 4)
        output_high_sigma = net.model(model_input, torch.tensor([1.0], device=device).log() / 4)
    sigma_difference = float((output_low_sigma - output_high_sigma).abs().mean())
    assert sigma_difference > 0
    assert torch.isfinite(output_low_sigma).all() and torch.isfinite(output_high_sigma).all()
    result = {
        'optimization_steps': steps,
        'post_update_backbone_sigma_output_mean_abs_difference': sigma_difference,
        'all_gradients_finite': True,
    }
    del net, optimizer, noisy, position, sigma, target
    return result


def real_patch_training_test(device, patch_size, activation_checkpointing, lightweight=False):
    source_batch = BATCH_MULTIPLIERS[patch_size]
    if lightweight:
        source_batch = 1
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


def snapshot_test(device):
    net = make_precond(device).eval()
    payload = {
        'ema': net.cpu(),
        'loss_fn': make_opa_loss(),
        'augment_pipe': None,
        'dataset_kwargs': {},
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / 'network-snapshot-v2-test.pkl'
        with path.open('wb') as handle:
            pickle.dump(payload, handle)
        size = path.stat().st_size
        with path.open('rb') as handle:
            restored = pickle.load(handle)
    restored_net = restored['ema'].eval()
    with torch.no_grad():
        output = restored_net(
            torch.randn(4, 2, 64, 64),
            torch.tensor(0.1),
            x_pos=torch.randn(4, 2, 64, 64),
        )
    assert tuple(output.shape) == (4, 2, 64, 64)
    assert torch.isfinite(output).all()
    return {
        'roundtrip': True,
        'restored_model_type': type(restored_net.model).__name__,
        'output_shape': list(output.shape),
        'pickle_bytes': size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument(
        '--lightweight',
        action='store_true',
        help='Use one source image per patch only for CPU structural validation',
    )
    parser.add_argument('--output', type=Path, default=THIS_DIR / 'nc_restormer_v2_test_results.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; real GPU training-path results were not produced')
        torch.cuda.set_device(device)
    torch.manual_seed(123)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(123)

    results = {
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch_version': torch.__version__,
        'dtype': 'FP32',
        'activation_checkpointing': args.activation_checkpointing,
        'real_training_batches': not args.lightweight,
        'batch_multipliers': BATCH_MULTIPLIERS,
        'architecture_and_initialization': architecture_and_initialization_test(device),
        'gradient_and_sigma': gradient_and_sigma_test(device, args.activation_checkpointing),
        'patch_training_paths': {
            str(patch): real_patch_training_test(
                device,
                patch,
                args.activation_checkpointing,
                args.lightweight,
            )
            for patch in (16, 32, 64)
        },
        'snapshot': snapshot_test(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
