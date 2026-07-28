"""Smoke tests for the isolated low-noise magnitude-SSIM loss."""

import argparse
import json
import math
import os
import pickle
import sys
import tempfile
import time
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from training.patch_loss import Patch_EDMLoss
from training.patch_ssim_loss import (
    MagnitudeSSIMPatchEDMLoss,
    gaussian_ssim_2d,
    magnitude_from_two_channels,
)


class CountingAffineNet(torch.nn.Module):
    """Small differentiable denoiser used only by this smoke test."""

    def __init__(self, scale=0.85):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
        self.calls = 0

    def forward(
        self,
        value,
        sigma,
        x_pos=None,
        class_labels=None,
        augment_labels=None,
    ):
        del sigma, x_pos, class_labels, augment_labels
        self.calls += 1
        return self.scale * value


def _two_channel_from_magnitude(magnitude, phase):
    return torch.cat(
        [
            magnitude * torch.cos(phase),
            magnitude * torch.sin(phase),
        ],
        dim=1,
    )


def _ssim_for_two_channel(prediction, target):
    return gaussian_ssim_2d(
        magnitude_from_two_channels(prediction),
        magnitude_from_two_channels(target),
    )


def _baseline_bypass_check():
    """Run the strict equivalence test before all other smoke tests."""
    torch.manual_seed(20260728)
    np.random.seed(20260728)
    images = torch.randn(3, 2, 40, 40)
    labels = torch.empty(3, 0)

    baseline_net = CountingAffineNet()
    bypass_net = CountingAffineNet()
    bypass_net.load_state_dict(baseline_net.state_dict())
    baseline = Patch_EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=0.5)
    bypass = MagnitudeSSIMPatchEDMLoss(
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        ssim_weight=0.0,
    )

    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    baseline_loss = baseline(
        net=baseline_net,
        images=images.clone(),
        patch_size=16,
        resolution=40,
        labels=labels.clone(),
    )

    torch.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    bypass_loss = bypass(
        net=bypass_net,
        images=images.clone(),
        patch_size=16,
        resolution=40,
        labels=labels.clone(),
    )

    if not torch.equal(baseline_loss, bypass_loss):
        maximum = (baseline_loss - bypass_loss).abs().max().item()
        raise AssertionError(
            f'baseline bypass is not torch.equal; max difference={maximum}'
        )
    baseline_loss.sum().backward()
    bypass_loss.sum().backward()
    if not torch.equal(baseline_net.scale.grad, bypass_net.scale.grad):
        difference = (
            baseline_net.scale.grad - bypass_net.scale.grad
        ).abs().max().item()
        raise AssertionError(
            'baseline bypass changed gradients; '
            f'max difference={difference}'
        )
    if baseline_net.calls != 1 or bypass_net.calls != 1:
        raise AssertionError('baseline bypass changed the network call count')
    return {
        'torch_equal': True,
        'maximum_absolute_difference': 0.0,
        'gradient_torch_equal': True,
        'maximum_gradient_difference': 0.0,
        'baseline_network_calls': baseline_net.calls,
        'bypass_network_calls': bypass_net.calls,
    }


class MagnitudeSSIMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_invalid_parameters(self):
        invalid = [
            {'ssim_weight': -0.1},
            {'ssim_sigma_max': 0.0},
            {'ssim_window_size': 0},
            {'ssim_window_size': 10},
            {'ssim_window_size': 17},
            {'ssim_window_sigma': 0.0},
            {'ssim_k1': 0.0},
            {'ssim_k2': 0.0},
            {'ssim_data_range': 0.0},
            {'magnitude_eps': 0.0},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    MagnitudeSSIMPatchEDMLoss(**kwargs)

    def test_identical_and_constant_images(self):
        random_image = torch.rand(2, 1, 32, 32)
        identical = gaussian_ssim_2d(random_image, random_image)
        self.assertTrue(torch.allclose(identical, torch.ones_like(identical), atol=1e-6))

        constant = torch.full((2, 1, 32, 32), 0.4)
        constant_ssim = gaussian_ssim_2d(constant, constant)
        self.assertTrue(
            torch.allclose(
                constant_ssim,
                torch.ones_like(constant_ssim),
                atol=1e-6,
            )
        )

    def test_zero_image_is_finite(self):
        zeros = torch.zeros(2, 1, 32, 32)
        value = gaussian_ssim_2d(zeros, zeros)
        self.assertTrue(torch.isfinite(value).all())

    def test_blur_shift_and_local_damage_reduce_ssim(self):
        image = torch.rand(1, 1, 32, 32)
        identical = gaussian_ssim_2d(image, image).item()
        blurred = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        shifted = torch.roll(image, shifts=(2, 3), dims=(-2, -1))
        damaged = image.clone()
        damaged[:, :, 10:22, 10:22] = 0
        self.assertLess(gaussian_ssim_2d(blurred, image).item(), identical)
        self.assertLess(gaussian_ssim_2d(shifted, image).item(), identical)
        self.assertLess(gaussian_ssim_2d(damaged, image).item(), identical)

    def test_patch_sizes(self):
        for patch_size in (16, 32, 64):
            with self.subTest(patch_size=patch_size):
                image = torch.rand(2, 1, patch_size, patch_size)
                value = gaussian_ssim_2d(image, image)
                self.assertEqual(tuple(value.shape), (2, 1, 1, 1))
                self.assertTrue(torch.isfinite(value).all())

    def test_global_phase_and_magnitude_change(self):
        magnitude = torch.rand(2, 1, 32, 32)
        phase_a = torch.zeros_like(magnitude)
        phase_b = torch.full_like(magnitude, math.pi / 3)
        image_a = _two_channel_from_magnitude(magnitude, phase_a)
        image_b = _two_channel_from_magnitude(magnitude, phase_b)

        magnitude_ssim = _ssim_for_two_channel(image_a, image_b)
        complex_edm = (image_a - image_b).square().mean()
        self.assertTrue(
            torch.allclose(
                magnitude_ssim,
                torch.ones_like(magnitude_ssim),
                atol=1e-5,
            )
        )
        self.assertGreater(complex_edm.item(), 0)

        changed = image_b.clone()
        changed[:, :, 8:24, 8:24] *= 0.5
        self.assertLess(
            _ssim_for_two_channel(changed, image_a).mean().item(),
            magnitude_ssim.mean().item(),
        )

    def test_backward_and_nonzero_finite_gradient(self):
        prediction = torch.randn(2, 2, 32, 32, requires_grad=True)
        target = torch.randn(2, 2, 32, 32)
        value = _ssim_for_two_channel(prediction, target)
        loss = torch.clamp_min(1.0 - value, 0.0).mean()
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.norm().item(), 0)

    def test_active_inactive_and_mixed_gate(self):
        raw = torch.tensor([0.2, 0.4, 0.6]).reshape(3, 1, 1, 1)
        sigma = torch.tensor([0.2, 0.7, 0.5]).reshape(3, 1, 1, 1)
        active = (sigma <= 0.5).float()
        gated = active * raw
        expected = torch.tensor([0.2, 0.0, 0.6]).reshape(3, 1, 1, 1)
        self.assertTrue(torch.equal(gated, expected))
        self.assertAlmostEqual(float(active.mean()), 2.0 / 3.0, places=6)

        prediction = torch.randn(3, 2, 16, 16, requires_grad=True)
        target = torch.randn(3, 2, 16, 16)
        ssim_value = _ssim_for_two_channel(prediction, target)
        ssim_loss = torch.clamp_min(1.0 - ssim_value, 0.0)
        (active * ssim_loss).sum().backward()
        per_sample_grad = prediction.grad.flatten(1).norm(dim=1)
        self.assertGreater(per_sample_grad[0].item(), 0.0)
        self.assertEqual(per_sample_grad[1].item(), 0.0)
        self.assertGreater(per_sample_grad[2].item(), 0.0)

    def test_one_network_call_and_output_shape(self):
        loss_fn = MagnitudeSSIMPatchEDMLoss()
        net = CountingAffineNet()
        images = torch.randn(2, 2, 64, 64)
        loss = loss_fn(
            net=net,
            images=images,
            patch_size=32,
            resolution=64,
            labels=torch.empty(2, 0),
        )
        self.assertEqual(net.calls, 1)
        self.assertEqual(tuple(loss.shape), (2, 2, 32, 32))
        self.assertTrue(torch.isfinite(loss).all())

    def test_two_channels_are_required(self):
        loss_fn = MagnitudeSSIMPatchEDMLoss()
        net = CountingAffineNet()
        with self.assertRaises(ValueError):
            loss_fn(
                net=net,
                images=torch.randn(2, 1, 32, 32),
                patch_size=16,
                resolution=32,
                labels=torch.empty(2, 0),
            )

    def test_pickle_roundtrip_and_no_persistent_window(self):
        loss_fn = MagnitudeSSIMPatchEDMLoss()
        tensor_fields = [
            key
            for key, value in loss_fn.__dict__.items()
            if torch.is_tensor(value)
        ]
        self.assertEqual(tensor_fields, [])
        restored = pickle.loads(pickle.dumps(loss_fn))
        self.assertEqual(restored.ssim_weight, 0.50)
        self.assertEqual(restored.ssim_window_size, 11)

    def test_optimizer_and_checkpoint_roundtrip(self):
        loss_fn = MagnitudeSSIMPatchEDMLoss()
        net = CountingAffineNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        images = torch.randn(2, 2, 32, 32)
        loss = loss_fn(
            net=net,
            images=images,
            patch_size=16,
            resolution=32,
            labels=torch.empty(2, 0),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'network-snapshot-000001.pkl')
            with open(path, 'wb') as handle:
                pickle.dump(
                    {'ema': net, 'loss_fn': loss_fn},
                    handle,
                )
            with open(path, 'rb') as handle:
                restored = pickle.load(handle)
        self.assertIn('ema', restored)
        self.assertIn('loss_fn', restored)
        self.assertTrue(
            torch.equal(
                restored['ema'].scale.detach(),
                net.scale.detach(),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_memory_and_fp16(self):
        device = torch.device('cuda')
        torch.cuda.reset_peak_memory_stats(device)
        prediction = torch.randn(
            2,
            2,
            32,
            32,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        target = torch.randn(
            2,
            2,
            32,
            32,
            device=device,
            dtype=torch.float16,
        )
        value = _ssim_for_two_channel(prediction, target)
        self.assertEqual(value.dtype, torch.float32)
        (1.0 - value).mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        print(
            'CUDA peak_memory_allocated_mb=',
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        )
        print(
            'CUDA peak_memory_reserved_mb=',
            torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
        )


def _diagnostic_ratios():
    """Measure auxiliary strength on fixed active synthetic samples."""
    torch.manual_seed(20260728)
    target = torch.randn(2, 2, 32, 32)
    prediction = (
        0.82 * target + 0.08 * torch.randn_like(target)
    ).requires_grad_(True)
    sigma = torch.full((2, 1, 1, 1), 0.25)
    sigma_data = 0.5
    edm_weight = (
        sigma.square() + sigma_data ** 2
    ) / (sigma * sigma_data).square()
    edm_loss = edm_weight * (prediction - target).square()

    ssim_value = _ssim_for_two_channel(prediction, target)
    ssim_loss = torch.clamp_min(1.0 - ssim_value, 0.0)
    active = (sigma <= 0.5).float()
    weighted_ssim = 0.50 * (active * ssim_loss).expand_as(edm_loss)

    grad_edm = torch.autograd.grad(
        edm_loss.mean(),
        prediction,
        retain_graph=True,
    )[0]
    grad_ssim = torch.autograd.grad(
        weighted_ssim.mean(),
        prediction,
    )[0]
    loss_ratio = (
        weighted_ssim.detach().mean()
        / edm_loss.detach().mean().clamp_min(1e-12)
    ).item()
    gradient_ratio = (
        grad_ssim.detach().norm()
        / grad_edm.detach().norm().clamp_min(1e-12)
    ).item()

    warnings = []
    if loss_ratio < 0.01:
        warnings.append('weighted SSIM loss ratio is below 0.01')
    if loss_ratio > 0.30:
        warnings.append('weighted SSIM loss ratio is above 0.30')
    if gradient_ratio < 0.03:
        warnings.append('SSIM gradient ratio is below 0.03')
    if gradient_ratio > 0.50:
        warnings.append('SSIM gradient ratio is above 0.50')
    return {
        'weighted_ssim_to_edm_loss_ratio': loss_ratio,
        'ssim_to_edm_gradient_norm_ratio': gradient_ratio,
        'warnings': warnings,
    }


def _training_data_statistics(path):
    if path is None:
        return {'status': 'not_provided'}
    if not os.path.isfile(path):
        return {'status': 'missing', 'path': os.path.abspath(path)}
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if 'x_est_gt' not in payload:
        raise KeyError(f"{path} does not contain 'x_est_gt'")
    value = payload['x_est_gt']
    magnitude = value.abs().float().reshape(-1)
    quantiles = torch.quantile(
        magnitude,
        torch.tensor([0.95, 0.99, 0.995, 0.999]),
    )
    return {
        'status': 'ok',
        'path': os.path.abspath(path),
        'shape': list(value.shape),
        'dtype': str(value.dtype),
        'is_complex': bool(torch.is_complex(value)),
        'complex_dtype': str(value.dtype) if torch.is_complex(value) else None,
        'magnitude_minimum': float(magnitude.min()),
        'magnitude_maximum': float(magnitude.max()),
        'magnitude_mean': float(magnitude.mean()),
        'magnitude_standard_deviation': float(magnitude.std()),
        'magnitude_p95': float(quantiles[0]),
        'magnitude_p99': float(quantiles[1]),
        'magnitude_p99_5': float(quantiles[2]),
        'magnitude_p99_9': float(quantiles[3]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--json-out', type=str)
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        bypass = _baseline_bypass_check()
    except Exception as error:
        print(f'BASELINE BYPASS FAILED: {error}', file=sys.stderr)
        return 1

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        MagnitudeSSIMTests
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed = time.perf_counter() - started
    summary = {
        'baseline_bypass': bypass,
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'success': result.wasSuccessful(),
        'diagnostics': _diagnostic_ratios(),
        'training_data': _training_data_statistics(args.data),
        'wall_time_seconds': elapsed,
        'cuda_available': torch.cuda.is_available(),
        'cuda_peak_memory_allocated_mb': (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            if torch.cuda.is_available()
            else None
        ),
        'cuda_peak_memory_reserved_mb': (
            torch.cuda.max_memory_reserved() / (1024.0 ** 2)
            if torch.cuda.is_available()
            else None
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        output = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
