#!/usr/bin/env python3
"""Matched FP32 benchmark for NC-Restormer V2 and V2-Sigma+.

GPU selection is intentionally external.  For example, setting
CUDA_VISIBLE_DEVICES=3 makes the physical GPU 3 appear as cuda:0 here.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'train' / 'padis-mri'))

import benchmark_nc_restormer_v2 as base  # noqa: E402
from test_nc_restormer_v2_sigma import MODEL_KWARGS  # noqa: E402
from training.networks import Patch_EDMPrecond  # noqa: E402


def make_model(kind, device):
    if kind not in (
        'nc_restormer_v2',
        'nc_restormer_v2_sigma',
        'nc_restormer_v2_sigma_checkpointed',
    ):
        raise ValueError(kind)
    sigma_variant = kind.startswith('nc_restormer_v2_sigma')
    kwargs = dict(MODEL_KWARGS)
    if not sigma_variant:
        for key in ('modulation_init_std', 'fourier_bandwidth', 'fourier_seed'):
            kwargs.pop(key)
    return Patch_EDMPrecond(
        img_resolution=512,
        img_channels=4,
        out_channels=2,
        model_type=(
            'NoiseConditionedRestormerV2Sigma'
            if sigma_variant else 'NoiseConditionedRestormerV2'
        ),
        activation_checkpointing=(kind == 'nc_restormer_v2_sigma_checkpointed'),
        **kwargs,
    ).to(device)


def static_metrics(kind):
    model = make_model(kind, torch.device('cpu'))
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    macs = base.count_restormer_macs(model.model, patch=64)
    result = {
        'trainable_parameters': parameters,
        'p64_batch1_macs': macs,
        'p64_batch1_flops': 2 * macs,
        'flop_convention': (
            '1 MAC = 2 FLOPs; learned Conv2d/Linear and MDTA QK/AV only; '
            'normalization, activations, cosine, softmax, concatenation and EDM scalar arithmetic excluded'
        ),
    }
    del model
    base.cleanup(torch.device('cpu'))
    return result


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
        base.cleanup(device)


def benchmark_model(kind, device, args):
    result = {'static': static_metrics(kind)}
    if args.static_only:
        return result
    result['p64_forward_batch1'] = safe_gpu_measure(
        kind, device,
        lambda model: base.measure_forward(model, device, 1, args.warmup, args.iterations),
    )
    result['p64_forward_batch32'] = safe_gpu_measure(
        kind, device,
        lambda model: base.measure_forward(model, device, 32, args.warmup, args.iterations),
    )
    result['p64_batch32_synthetic_train'] = safe_gpu_measure(
        kind, device,
        lambda model: base.measure_synthetic_training(
            model, device, args.train_warmup, args.train_iterations,
        ),
    )
    result['real_patch_microsteps'] = {
        str(patch): safe_gpu_measure(
            kind,
            device,
            lambda model, patch=patch: base.measure_patch_microstep(
                model, device, patch, args.train_warmup, args.train_iterations,
            ),
        )
        for patch in (16, 32, 64)
    }
    result['fixed_p64_formal_optimizer_update'] = safe_gpu_measure(
        kind, device,
        lambda model: base.measure_fixed_p64_formal_update(
            model, device, args.formal_warmup, args.formal_iterations,
        ),
    )
    result['mixed_patch_formal_optimizer_update'] = safe_gpu_measure(
        kind, device,
        lambda model: base.measure_mixed_formal_update(
            model, device, args.formal_warmup, args.formal_iterations,
        ),
    )
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
    parser.add_argument(
        '--output', type=Path,
        default=THIS_DIR / 'nc_restormer_v2_sigma_benchmark.json',
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if not args.static_only:
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError(
                'The runtime benchmark requires CUDA; use --static-only for the architecture audit'
            )
        torch.cuda.set_device(device)

    kinds = (
        'nc_restormer_v2',
        'nc_restormer_v2_sigma',
        'nc_restormer_v2_sigma_checkpointed',
    )
    results = {
        'device': str(device),
        'device_name': (
            torch.cuda.get_device_name(device)
            if not args.static_only else 'not measured (static-only)'
        ),
        'torch_version': torch.__version__,
        'dtype': 'FP32',
        'physical_gpu_selection': (
            'Set CUDA_VISIBLE_DEVICES to a free physical GPU; the program uses --device cuda:0'
        ),
        'gpu_runtime_status': 'pending' if args.static_only else 'measured',
        'isolation': (
            'Each metric constructs and releases its model independently; stable peaks are reset after warmup'
        ),
        'models': {kind: benchmark_model(kind, device, args) for kind in kinds},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
