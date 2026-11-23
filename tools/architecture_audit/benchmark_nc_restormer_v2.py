#!/usr/bin/env python3
"""Matched static and GPU benchmark for SongUNet C96, NC-Restormer V1/V2."""

import argparse
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'train' / 'padis-mri'))

from profile_songunet import audit as audit_songunet  # noqa: E402
from test_nc_restormer_v2 import BATCH_MULTIPLIERS, MODEL_KWARGS, make_opa_loss  # noqa: E402
from training.networks import Patch_EDMPrecond  # noqa: E402


SONG_KWARGS = dict(
    model_channels=96,
    channel_mult=[2, 2, 2],
    num_blocks=4,
    dropout=0.05,
    attn_resolutions=[16],
    channel_mult_noise=1,
    embedding_type='positional',
    encoder_type='standard',
    decoder_type='standard',
    resample_filter=[1, 1],
)


def make_model(kind, device):
    if kind == 'songunet_c96':
        kwargs = dict(model_type='SongUNet', **SONG_KWARGS)
    elif kind == 'nc_restormer_v1':
        kwargs = dict(model_type='NoiseConditionedRestormer', **{
            key: value for key, value in MODEL_KWARGS.items()
            if key not in ('decoder_level1_dim', 'decoder_level1_heads', 'gate_bias', 'output_init_std')
        })
    elif kind in ('nc_restormer_v2', 'nc_restormer_v2_checkpointed'):
        kwargs = dict(
            model_type='NoiseConditionedRestormerV2',
            activation_checkpointing=(kind == 'nc_restormer_v2_checkpointed'),
            **MODEL_KWARGS,
        )
    else:
        raise ValueError(kind)
    return Patch_EDMPrecond(
        img_resolution=512,
        img_channels=4,
        out_channels=2,
        **kwargs,
    ).to(device)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def memory_stats(device):
    if device.type != 'cuda':
        return {'allocated_mib': None, 'reserved_mib': None, 'peak_allocated_mib': None,
                'peak_reserved_mib': None}
    mib = 1024 ** 2
    return {
        'allocated_mib': torch.cuda.memory_allocated(device) / mib,
        'reserved_mib': torch.cuda.memory_reserved(device) / mib,
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / mib,
        'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / mib,
    }


def reset_stable_peak(device):
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def cleanup(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def count_restormer_macs(backbone, patch=64):
    total_macs = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal total_macs
        batch, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        input_channels = module.in_channels // module.groups
        total_macs += batch * out_channels * out_height * out_width * input_channels * kernel_height * kernel_width

    def linear_hook(module, inputs, output):
        nonlocal total_macs
        vectors = inputs[0].numel() // module.in_features
        total_macs += vectors * module.in_features * module.out_features

    def attention_hook(module, inputs, _output):
        nonlocal total_macs
        x = inputs[0]
        batch, channels, height, width = x.shape
        channels_per_head = channels // module.num_heads
        total_macs += 2 * batch * module.num_heads * channels_per_head * channels_per_head * height * width

    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif module.__class__.__name__ == 'MultiDconvHeadTransposedAttention':
            hooks.append(module.register_forward_hook(attention_hook))
    backbone.eval()
    with torch.no_grad():
        backbone(torch.zeros(1, 4, patch, patch), torch.zeros(1))
    for hook in hooks:
        hook.remove()
    return int(total_macs)


def static_metrics(kind):
    model = make_model(kind, torch.device('cpu'))
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if kind == 'songunet_c96':
        audit = audit_songunet(96, 64)
        macs = int(audit['total_macs'])
    else:
        macs = count_restormer_macs(model.model, patch=64)
    result = {
        'trainable_parameters': parameters,
        'p64_batch1_macs': macs,
        'p64_batch1_flops': 2 * macs,
        'flop_convention': (
            '1 MAC = 2 FLOPs; learned Conv2d/Linear and MDTA QK/AV only; '
            'normalization, activations, softmax, concatenation and EDM scalar arithmetic excluded'
        ),
    }
    del model
    cleanup(torch.device('cpu'))
    return result


def timed_samples(step, device, warmup, iterations):
    for _ in range(warmup):
        step()
    reset_stable_peak(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        step()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    return {
        'mean_ms': statistics.mean(samples),
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'iterations': iterations,
        'memory': memory_stats(device),
    }


def measure_forward(model, device, batch, warmup, iterations):
    model.eval()
    noisy = torch.randn(batch, 2, 64, 64, device=device)
    position = torch.randn(batch, 2, 64, 64, device=device)
    sigma = torch.full((batch, 1, 1, 1), 0.1, device=device)

    def step():
        with torch.no_grad():
            model(noisy, sigma, x_pos=position)

    result = timed_samples(step, device, warmup, iterations)
    result.update({'patch_size': 64, 'network_batch': batch, 'dtype': 'FP32'})
    return result


def measure_synthetic_training(model, device, warmup, iterations):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    noisy = torch.randn(32, 2, 64, 64, device=device)
    position = torch.randn(32, 2, 64, 64, device=device)
    sigma = torch.full((32, 1, 1, 1), 0.1, device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(noisy, sigma, x_pos=position).square().mean()
        loss.backward()
        optimizer.step()

    result = timed_samples(step, device, warmup, iterations)
    result.update({
        'patch_size': 64,
        'network_batch': 32,
        'operation': 'forward + backward + Adam, synthetic denoised-output objective',
    })
    return result


def loss_round(model, loss_fn, images, patch_size):
    loss_map = loss_fn(
        net=model,
        images=images,
        patch_size=patch_size,
        resolution=512,
        labels=None,
        augment_pipe=None,
    )
    return loss_map.sum() / images.shape[0]


def measure_patch_microstep(model, device, patch_size, warmup, iterations):
    model.train()
    source_batch = BATCH_MULTIPLIERS[patch_size]
    network_batch = source_batch * (2 if patch_size == 64 else 1)
    images = torch.randn(source_batch, 2, 512, 512, device=device)
    loss_fn = make_opa_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_round(model, loss_fn, images, patch_size)
        loss.backward()
        optimizer.step()

    result = timed_samples(step, device, warmup, iterations)
    result.update({
        'patch_size': patch_size,
        'source_batch': source_batch,
        'network_batch': network_batch,
        'operation': 'one real PaDIS loss round + backward + Adam',
    })
    return result


def measure_fixed_p64_formal_update(model, device, warmup, iterations):
    model.train()
    source_batch = BATCH_MULTIPLIERS[64]
    images = torch.randn(source_batch, 2, 512, 512, device=device)
    loss_fn = make_opa_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

    def step():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(4):
            (loss_round(model, loss_fn, images, 64) / 4).backward()
        optimizer.step()

    result = timed_samples(step, device, warmup, iterations)
    result.update({
        'patch_size': 64,
        'source_batch_per_accumulation_round': source_batch,
        'network_batch_per_accumulation_round': 32,
        'gradient_accumulation_rounds': 4,
        'operation': 'four fixed-P64 PaDIS loss rounds + one Adam step',
    })
    return result


def measure_mixed_formal_update(model, device, warmup, iterations):
    model.train()
    images = {
        patch: torch.randn(BATCH_MULTIPLIERS[patch], 2, 512, 512, device=device)
        for patch in (16, 32, 64)
    }
    loss_fn = make_opa_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    rng = random.Random(123)
    measured_sequences = []

    def step(record=False):
        sequence = rng.choices((16, 32, 64), weights=(0.2, 0.3, 0.5), k=4)
        optimizer.zero_grad(set_to_none=True)
        for patch in sequence:
            (loss_round(model, loss_fn, images[patch], patch) / 4).backward()
        optimizer.step()
        if record:
            measured_sequences.append(sequence)

    for _ in range(warmup):
        step()
    reset_stable_peak(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        step(record=True)
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    mean_ms = statistics.mean(samples)
    expected_kimg_per_update = 4 * (0.2 * 64 + 0.3 * 32 + 0.5 * 16) / 1000
    return {
        'mean_ms': mean_ms,
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'iterations': iterations,
        'gradient_accumulation_rounds': 4,
        'patch_probabilities': {'16': 0.2, '32': 0.3, '64': 0.5},
        'measured_patch_sequences': measured_sequences,
        'expected_kimg_per_optimizer_update': expected_kimg_per_update,
        'model_only_seconds_per_kimg': mean_ms / 1000 / expected_kimg_per_update,
        'memory': memory_stats(device),
        'scope': 'model/loss/optimizer only; excludes data loading, EMA, snapshots and logging',
    }


def safe_gpu_measure(kind, device, function):
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    model = make_model(kind, device)
    try:
        return {'status': 'ok', **function(model)}
    except RuntimeError as error:
        if 'out of memory' not in str(error).lower():
            raise
        return {'status': 'oom', 'error': str(error)}
    finally:
        del model
        cleanup(device)


def benchmark_model(kind, device, args):
    result = {'static': static_metrics(kind)}
    if args.static_only:
        return result
    result['p64_forward_batch1'] = safe_gpu_measure(
        kind, device, lambda model: measure_forward(model, device, 1, args.warmup, args.iterations))
    result['p64_forward_batch32'] = safe_gpu_measure(
        kind, device, lambda model: measure_forward(model, device, 32, args.warmup, args.iterations))
    result['p64_batch32_synthetic_train'] = safe_gpu_measure(
        kind, device, lambda model: measure_synthetic_training(
            model, device, args.train_warmup, args.train_iterations))
    result['real_patch_microsteps'] = {
        str(patch): safe_gpu_measure(
            kind,
            device,
            lambda model, patch=patch: measure_patch_microstep(
                model, device, patch, args.train_warmup, args.train_iterations),
        )
        for patch in (16, 32, 64)
    }
    result['fixed_p64_formal_optimizer_update'] = safe_gpu_measure(
        kind, device, lambda model: measure_fixed_p64_formal_update(
            model, device, args.formal_warmup, args.formal_iterations))
    result['mixed_patch_formal_optimizer_update'] = safe_gpu_measure(
        kind, device, lambda model: measure_mixed_formal_update(
            model, device, args.formal_warmup, args.formal_iterations))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--train-warmup', type=int, default=2)
    parser.add_argument('--train-iterations', type=int, default=10)
    parser.add_argument('--formal-warmup', type=int, default=1)
    parser.add_argument('--formal-iterations', type=int, default=5)
    parser.add_argument('--static-only', action='store_true')
    parser.add_argument('--output', type=Path, default=THIS_DIR / 'nc_restormer_v2_benchmark.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    if not args.static_only:
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('The runtime benchmark requires CUDA; use --static-only for CPU architecture audit')
        torch.cuda.set_device(device)

    kinds = (
        'songunet_c96',
        'nc_restormer_v1',
        'nc_restormer_v2',
        'nc_restormer_v2_checkpointed',
    )
    results = {
        'device': str(device),
        'device_name': (
            torch.cuda.get_device_name(device)
            if not args.static_only else 'not measured (static-only)'
        ),
        'torch_version': torch.__version__,
        'dtype': 'FP32',
        'physical_gpu_selection': 'Run with CUDA_VISIBLE_DEVICES=2 and --device cuda:0',
        'gpu_runtime_status': 'pending' if args.static_only else 'measured',
        'models': {kind: benchmark_model(kind, device, args) for kind in kinds},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
