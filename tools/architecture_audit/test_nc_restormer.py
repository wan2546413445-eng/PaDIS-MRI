#!/usr/bin/env python3
"""Validate and benchmark the PaDIS-MRI Noise-Conditioned Restormer."""

import argparse
import json
import pickle
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'train' / 'padis-mri'))

from training.networks import Patch_EDMPrecond  # noqa: E402
from training.noise_conditioned_restormer import NoiseConditionedRestormer  # noqa: E402
from training.patch_overlap_independent_loss import IndependentNoiseOverlapPatch_EDMLoss  # noqa: E402


MODEL_KWARGS = dict(
    dim=80,
    num_blocks=[2, 3, 4],
    num_decoder_blocks=[3, 2],
    num_refinement_blocks=2,
    heads=[2, 4, 8],
    ffn_expansion_factor=2.0,
    bias=False,
    layer_norm_type='WithBias',
    dropout=0.05,
    condition_dim=320,
    noise_embedding_size=256,
)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def reset_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def memory_stats(device):
    if device.type != 'cuda':
        return {'allocated_mib': None, 'reserved_mib': None, 'peak_allocated_mib': None}
    mib = 1024 ** 2
    return {
        'allocated_mib': torch.cuda.memory_allocated(device) / mib,
        'reserved_mib': torch.cuda.memory_reserved(device) / mib,
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / mib,
    }


def make_backbone(device, activation_checkpointing=False):
    return NoiseConditionedRestormer(
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
        model_type='NoiseConditionedRestormer',
        activation_checkpointing=activation_checkpointing,
        **MODEL_KWARGS,
    ).to(device)


def model_summary(device):
    model = make_backbone(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    output_is_zero = bool(torch.count_nonzero(model.output.weight).item() == 0)
    modulation_is_zero = all(
        torch.count_nonzero(module.adaLN_modulation[-1].weight).item() == 0
        and torch.count_nonzero(module.adaLN_modulation[-1].bias).item() == 0
        for module in model.modules()
        if module.__class__.__name__ == 'NoiseConditionedTransformerBlock'
    )
    conditioned_block_count = sum(
        module.__class__.__name__ == 'NoiseConditionedTransformerBlock'
        for module in model.modules()
    )
    del model
    return {
        'total_parameters': total,
        'trainable_parameters': trainable,
        'output_zero_initialized': output_is_zero,
        'all_adaln_modulations_zero_initialized': modulation_is_zero,
        'conditioned_transformer_blocks': conditioned_block_count,
        'channels': [80, 160, 320],
        'encoder_blocks': [2, 3, 4],
        'decoder_blocks': [3, 2],
        'refinement_blocks': 2,
        'heads': [2, 4, 8],
    }


def forward_and_sigma_tests(device):
    model = make_backbone(device).eval()
    shapes = {}
    sigma_outputs = {}
    with torch.no_grad():
        for patch in (16, 32, 64):
            output = model(
                torch.randn(1, 4, patch, patch, device=device),
                torch.zeros(1, device=device),
            )
            assert tuple(output.shape) == (1, 2, patch, patch)
            assert torch.isfinite(output).all()
            shapes[str(patch)] = list(output.shape)

        embedded_conditions = []
        for sigma in (0.003, 0.1, 1.0, 10.0):
            noise_label = torch.full((1,), sigma, device=device).log() / 4
            condition = model.noise_embedding(noise_label)
            assert torch.isfinite(condition).all()
            embedded_conditions.append(condition)
            sigma_outputs[str(sigma)] = {
                'finite': True,
                'condition_l2_norm': float(condition.norm()),
            }
        minimum_pairwise_distance = min(
            float((embedded_conditions[i] - embedded_conditions[j]).norm())
            for i in range(len(embedded_conditions))
            for j in range(i + 1, len(embedded_conditions))
        )
        assert minimum_pairwise_distance > 0
    del model
    return {
        'patch_output_shapes': shapes,
        'different_sigma_embeddings': sigma_outputs,
        'minimum_pairwise_condition_distance': minimum_pairwise_distance,
    }


def backward_tests(device, activation_checkpointing):
    results = {}
    for patch in (16, 32, 64):
        reset_peak_memory(device)
        net = make_precond(device, activation_checkpointing).train()
        noisy = torch.randn(1, 2, patch, patch, device=device)
        position = torch.randn(1, 2, patch, patch, device=device)
        sigma = torch.full((1, 1, 1, 1), 0.1, device=device)
        output = net(noisy, sigma, x_pos=position)
        loss = output.square().mean()
        loss.backward()
        missing = [
            name for name, parameter in net.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        nonfinite = [
            name for name, parameter in net.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        assert not missing, missing
        assert not nonfinite, nonfinite
        results[str(patch)] = {
            'loss': float(loss.detach()),
            'all_trainable_parameters_have_gradients': True,
            'all_gradients_finite': True,
            'memory': memory_stats(device),
        }
        del net, noisy, position, sigma, output, loss
    return results


def make_opa_loss():
    return IndependentNoiseOverlapPatch_EDMLoss(
        lambda_overlap=1.25,
        active_patch_size=64,
        noise_mode='independent',
        alignment_mode='symmetric',
        pair_geometry='identical',
        spatial_weight_mode='uniform',
    )


def opa_test(device, activation_checkpointing):
    reset_peak_memory(device)
    net = make_precond(device, activation_checkpointing).train()
    loss_fn = make_opa_loss()
    images = torch.randn(1, 2, 96, 96, device=device)
    loss_map = loss_fn(
        net=net,
        images=images,
        patch_size=64,
        resolution=96,
        labels=None,
        augment_pipe=None,
    )
    loss = loss_map.mean()
    loss.backward()
    missing = [
        name for name, parameter in net.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite = [
        name for name, parameter in net.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert tuple(loss_map.shape) == (1, 2, 64, 64)
    assert torch.isfinite(loss_map).all()
    assert not missing, missing
    assert not nonfinite, nonfinite
    result = {
        'source_image_shape': list(images.shape),
        'paired_network_batch_shape': [2, 2, 64, 64],
        'loss_map_shape': list(loss_map.shape),
        'loss': float(loss.detach()),
        'all_trainable_parameters_have_gradients': True,
        'all_gradients_finite': True,
        'memory': memory_stats(device),
    }
    del net, loss_fn, images, loss_map, loss
    return result


def snapshot_test(device):
    net = make_precond(device).eval()
    loss_fn = make_opa_loss()
    with torch.no_grad():
        batched_output = net(
            torch.randn(4, 2, 64, 64, device=device),
            torch.tensor(0.1, device=device),
            x_pos=torch.randn(4, 2, 64, 64, device=device),
        )
    assert tuple(batched_output.shape) == (4, 2, 64, 64)
    assert torch.isfinite(batched_output).all()
    snapshot_data = {
        'ema': net.cpu(),
        'loss_fn': loss_fn,
        'augment_pipe': None,
        'dataset_kwargs': {},
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot_path = Path(temp_dir) / 'network-snapshot-test.pkl'
        with snapshot_path.open('wb') as handle:
            pickle.dump(snapshot_data, handle)
        pickle_bytes = snapshot_path.stat().st_size
        with snapshot_path.open('rb') as handle:
            restored = pickle.load(handle)

    restored_net = restored['ema'].eval()
    with torch.no_grad():
        output = restored_net(
            torch.randn(1, 2, 16, 16),
            torch.tensor(0.1),
            x_pos=torch.randn(1, 2, 16, 16),
        )
    assert tuple(output.shape) == (1, 2, 16, 16)
    assert torch.isfinite(output).all()
    return {
        'full_snapshot_dictionary_roundtrip': True,
        'restored_model_type': type(restored_net.model).__name__,
        'batched_scalar_sigma_output_shape': list(batched_output.shape),
        'restored_output_shape': list(output.shape),
        'pickle_bytes': pickle_bytes,
    }


def timed_forward(device, batch, warmup, iterations):
    reset_peak_memory(device)
    net = make_precond(device).eval()
    noisy = torch.randn(batch, 2, 64, 64, device=device)
    position = torch.randn(batch, 2, 64, 64, device=device)
    sigma = torch.full((batch, 1, 1, 1), 0.1, device=device)
    samples = []
    with torch.no_grad():
        for _ in range(warmup):
            net(noisy, sigma, x_pos=position)
        synchronize(device)
        for _ in range(iterations):
            start = time.perf_counter()
            net(noisy, sigma, x_pos=position)
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
    result = {
        'mean_ms': statistics.mean(samples),
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'iterations': iterations,
        'memory': memory_stats(device),
    }
    del net, noisy, position, sigma
    return result


def timed_training_step(device, activation_checkpointing, warmup, iterations):
    reset_peak_memory(device)
    net = make_precond(device, activation_checkpointing).train()
    loss_fn = make_opa_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

    def step():
        optimizer.zero_grad(set_to_none=True)
        images = torch.randn(1, 2, 96, 96, device=device)
        loss = loss_fn(
            net=net,
            images=images,
            patch_size=64,
            resolution=96,
            labels=None,
            augment_pipe=None,
        ).mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(warmup):
        step()
    synchronize(device)
    samples = []
    losses = []
    for _ in range(iterations):
        start = time.perf_counter()
        losses.append(step())
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    result = {
        'patch_size': 64,
        'source_batch': 1,
        'effective_paired_forward_batch': 2,
        'mean_ms': statistics.mean(samples),
        'median_ms': statistics.median(samples),
        'iterations': iterations,
        'last_loss': losses[-1],
        'memory': memory_stats(device),
    }
    del net, loss_fn, optimizer
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:2', help='Use cuda:2 for the requested GPU 2 audit')
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--train-warmup', type=int, default=2)
    parser.add_argument('--train-iterations', type=int, default=10)
    parser.add_argument('--skip-performance', action='store_true')
    parser.add_argument('--skip-opa', action='store_true')
    parser.add_argument('--output', type=Path, default=THIS_DIR / 'nc_restormer_test_results.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; GPU results were not produced')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f'{device} is unavailable; visible CUDA device count is {torch.cuda.device_count()}')
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
        'model_summary': model_summary(device),
        'forward_and_sigma_tests': forward_and_sigma_tests(device),
        'backward_tests': backward_tests(device, args.activation_checkpointing),
        'snapshot_test': snapshot_test(device),
    }
    if not args.skip_opa:
        results['opa_test'] = opa_test(device, args.activation_checkpointing)
    if not args.skip_performance:
        results['forward_latency'] = {
            'batch_1': timed_forward(device, 1, args.warmup, args.iterations),
            'batch_32': timed_forward(device, 32, args.warmup, args.iterations),
        }
        results['training_step_latency'] = timed_training_step(
            device,
            args.activation_checkpointing,
            args.train_warmup,
            args.train_iterations,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
