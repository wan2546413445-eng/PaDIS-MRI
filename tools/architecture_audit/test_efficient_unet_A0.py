#!/usr/bin/env python3
"""Lightweight shape, EDM, backward, OPA, and optional CUDA latency tests for A0."""

import argparse
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import psutil
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'train' / 'padis-mri'))

from training.efficient_unet import EfficientSongUNet  # noqa: E402
from training.networks import Patch_EDMPrecond, SongUNet  # noqa: E402
from training.patch_overlap_independent_loss import IndependentNoiseOverlapPatch_EDMLoss  # noqa: E402


MODEL_KWARGS = dict(
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


def make_backbone(model_class, device):
    return model_class(
        img_resolution=512,
        in_channels=4,
        out_channels=2,
        **MODEL_KWARGS,
    ).to(device)


def make_precond(device):
    return Patch_EDMPrecond(
        img_resolution=512,
        img_channels=4,
        out_channels=2,
        model_type='EfficientSongUNet',
        **MODEL_KWARGS,
    ).to(device)


def shape_test(device):
    model = make_backbone(EfficientSongUNet, device).eval()
    results = {}
    with torch.no_grad():
        for patch in (16, 32, 64):
            output = model(
                torch.randn(1, 4, patch, patch, device=device),
                torch.zeros(1, device=device),
                None,
            )
            expected = (1, 2, patch, patch)
            assert tuple(output.shape) == expected
            assert torch.isfinite(output).all()
            results[str(patch)] = list(output.shape)
    return results


def edm_test(device):
    net = make_precond(device).eval()
    results = {}
    noisy = torch.randn(1, 2, 16, 16, device=device)
    position = torch.randn(1, 2, 16, 16, device=device)
    with torch.no_grad():
        for sigma_value in (0.003, 0.1, 1.0, 10.0):
            sigma = torch.full((1, 1, 1, 1), sigma_value, device=device)
            output = net(noisy, sigma, x_pos=position)
            finite = bool(torch.isfinite(output).all())
            assert finite
            results[str(sigma_value)] = {'shape': list(output.shape), 'finite': finite}
    return results


def backward_test(device):
    process = psutil.Process()
    results = {}
    for patch in (16, 32, 64):
        net = make_precond(device).train()
        before_rss = process.memory_info().rss
        noisy = torch.randn(1, 2, patch, patch, device=device)
        position = torch.randn(1, 2, patch, patch, device=device)
        sigma = torch.full((1, 1, 1, 1), 0.1, device=device)
        loss = net(noisy, sigma, x_pos=position).square().mean()
        loss.backward()
        trainable = [parameter for parameter in net.parameters() if parameter.requires_grad]
        missing = [name for name, parameter in net.named_parameters() if parameter.requires_grad and parameter.grad is None]
        nonfinite = [name for name, parameter in net.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        assert not missing, missing
        assert not nonfinite, nonfinite
        after_rss = process.memory_info().rss
        results[str(patch)] = {
            'loss': float(loss.detach()),
            'trainable_tensors': len(trainable),
            'missing_gradient_tensors': missing,
            'nonfinite_gradient_tensors': nonfinite,
            'process_rss_delta_mib': (after_rss - before_rss) / (1024 ** 2),
        }
        del net, noisy, position, sigma, loss
    return results


def opa_test(device):
    net = make_precond(device).train()
    loss_fn = IndependentNoiseOverlapPatch_EDMLoss(
        lambda_overlap=1.25,
        active_patch_size=64,
        noise_mode='independent',
        alignment_mode='symmetric',
        pair_geometry='identical',
        spatial_weight_mode='uniform',
    )
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
    missing = [name for name, parameter in net.named_parameters() if parameter.requires_grad and parameter.grad is None]
    nonfinite = [name for name, parameter in net.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
    assert tuple(loss_map.shape) == (1, 2, 64, 64)
    assert torch.isfinite(loss_map).all()
    assert not missing, missing
    assert not nonfinite, nonfinite
    return {
        'input_full_image_shape': list(images.shape),
        'paired_network_batch_shape': [2, 2, 64, 64],
        'loss_map_shape': list(loss_map.shape),
        'loss': float(loss.detach()),
        'missing_gradient_tensors': missing,
        'nonfinite_gradient_tensors': nonfinite,
    }


def batched_edm_and_pickle_test(device):
    """Exercise the scalar-sigma batched call used by patch DPS and snapshots."""
    net = make_precond(device).eval()
    with torch.no_grad():
        output = net(
            torch.randn(4, 2, 64, 64, device=device),
            torch.tensor(0.1, device=device),
            x_pos=torch.randn(4, 2, 64, 64, device=device),
        )
    assert tuple(output.shape) == (4, 2, 64, 64)
    assert torch.isfinite(output).all()

    # Persistent classes are the repository's checkpoint format.  Perform the
    # round-trip on CPU to avoid serializing device-specific state.
    snapshot = pickle.dumps(net.cpu())
    restored = pickle.loads(snapshot).eval()
    with torch.no_grad():
        restored_output = restored(
            torch.randn(1, 2, 16, 16),
            torch.tensor(0.1),
            x_pos=torch.randn(1, 2, 16, 16),
        )
    assert tuple(restored_output.shape) == (1, 2, 16, 16)
    assert torch.isfinite(restored_output).all()
    return {
        'batched_output_shape': list(output.shape),
        'scalar_sigma_broadcast': True,
        'persistent_pickle_roundtrip': True,
        'pickle_bytes': len(snapshot),
    }


def original_songunet_regression_test(device):
    net = Patch_EDMPrecond(
        img_resolution=512,
        img_channels=4,
        out_channels=2,
        model_type='SongUNet',
        **MODEL_KWARGS,
    ).to(device).eval()
    assert type(net.model) is SongUNet
    params = sum(parameter.numel() for parameter in net.parameters())
    assert params == 30623138
    with torch.no_grad():
        output = net(
            torch.randn(1, 2, 16, 16, device=device),
            torch.tensor(0.1, device=device),
            x_pos=torch.randn(1, 2, 16, 16, device=device),
        )
    assert tuple(output.shape) == (1, 2, 16, 16)
    assert torch.isfinite(output).all()
    return {'model_type': type(net.model).__name__, 'params': params, 'output_shape': list(output.shape)}


def latency_one(model, batch, warmup=10, iterations=30):
    noisy = torch.randn(batch, 4, 64, 64, device='cuda')
    noise_labels = torch.zeros(batch, device='cuda')
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(noisy, noise_labels, None)
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(noisy, noise_labels, None)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    return {'mean_ms': statistics.mean(samples), 'median_ms': statistics.median(samples), 'iterations': iterations}


def latency_test():
    if not torch.cuda.is_available():
        return {'status': 'skipped', 'reason': 'CUDA device unavailable'}
    results = {'device': torch.cuda.get_device_name(0), 'dtype': 'FP32'}
    for name, model_class in [('SongUNet C96', SongUNet), ('A0 EfficientSongUNet', EfficientSongUNet)]:
        model = make_backbone(model_class, 'cuda')
        results[name] = {str(batch): latency_one(model, batch) for batch in (1, 49)}
        del model
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=THIS_DIR / 'efficient_unet_A0_test_results.json')
    args = parser.parse_args()
    torch.manual_seed(123)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {
        'device': device,
        'torch_version': torch.__version__,
        'shape_test': shape_test(device),
        'edm_test': edm_test(device),
        'backward_test': backward_test(device),
        'opa_test': opa_test(device),
        'batched_edm_and_pickle_test': batched_edm_and_pickle_test(device),
        'original_songunet_regression_test': original_songunet_regression_test(device),
        'latency_test': latency_test(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

