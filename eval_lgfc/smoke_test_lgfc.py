"""Executable smoke tests for LGFC stage 1 v2."""

import inspect
import os
import sys
import types
import unittest
from unittest import mock

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EVAL_DIR = os.path.join(_REPO_ROOT, "eval")
for _path in (_REPO_ROOT, _EVAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from eval_lgfc import recon_lgfc
from eval_lgfc.frequency_correction import (
    LGFCConfig,
    apply_lgfc_frequency_correction,
    canonicalize_mask,
    compute_dc_residual,
    compute_full_multicoil_kspace,
    update_reference,
    validate_lgfc_config,
)


def _complex_ones(shape, device="cpu"):
    return torch.ones(shape, dtype=torch.complex64, device=device)


class FrequencyCorrectionTests(unittest.TestCase):
    def test_invalid_configs_raise(self):
        invalid = [
            LGFCConfig(reference_mode="bad"),
            LGFCConfig(ema_beta=1.0),
            LGFCConfig(start_outer=0),
            LGFCConfig(correction_every=0),
            LGFCConfig(unsampled_weight=-0.1),
            LGFCConfig(max_relative_update=0.0),
            LGFCConfig(dc_growth_limit=0.99),
            LGFCConfig(backtrack_factor=0.0),
            LGFCConfig(backtrack_factor=1.0),
            LGFCConfig(max_backtracks=-1),
            LGFCConfig(eps=0.0),
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_lgfc_config(config)

    def test_baseline_bypass_calls_original(self):
        sentinel = (torch.tensor([123]), 0, 0, 0, 0, 0, 0)
        baseline = mock.Mock(return_value=sentinel)
        with mock.patch.object(recon_lgfc, "_load_baseline_dps2", return_value=baseline):
            result = recon_lgfc.dps2_lgfc(
                net=object(),
                latents=torch.empty(0),
                latents_pos=torch.empty(0),
                inverseop=object(),
                measurement=None,
                lgfc_config=LGFCConfig(enabled=False),
            )
        self.assertIs(result, sentinel)
        baseline.assert_called_once()

    def test_reference_last_ema_detach_and_outer_reset(self):
        first = _complex_ones((1, 1, 2, 2)).requires_grad_(True)
        second = (3.0 * _complex_ones((1, 1, 2, 2))).requires_grad_(True)
        last = update_reference(first.detach(), second, "last", 0.8)
        ema = update_reference(first.detach(), second, "ema", 0.75)
        reset = update_reference(None, second, "ema", 0.75)
        self.assertTrue(torch.equal(last, second.detach()))
        self.assertTrue(torch.allclose(ema, 1.5 * _complex_ones((1, 1, 2, 2))))
        self.assertTrue(torch.equal(reset, second.detach()))
        self.assertFalse(last.requires_grad)
        self.assertFalse(ema.requires_grad)
        self.assertFalse(reset.requires_grad)

    def test_mask_multicoil_fft_and_complex_dtype(self):
        batch, coils, height, width = 2, 3, 4, 4
        x = _complex_ones((batch, 1, height, width))
        maps = _complex_ones((batch, coils, height, width))
        mask = torch.zeros(height, width)
        mask[:, ::2] = 1
        canonical = canonicalize_mask(
            mask,
            batch,
            height,
            width,
            device=x.device,
            real_dtype=x.real.dtype,
        )
        kspace = compute_full_multicoil_kspace(x, maps)
        self.assertEqual(tuple(canonical.shape), (batch, 1, height, width))
        self.assertEqual(tuple(kspace.shape), (batch, coils, height, width))
        self.assertTrue(torch.is_complex(kspace))

    def test_zero_weight_and_full_mask_are_identity(self):
        x = _complex_ones((1, 1, 4, 4))
        reference = (2.0 + 1.0j) * x
        maps = _complex_ones((1, 2, 4, 4))
        measurement = torch.zeros((1, 2, 4, 4), dtype=torch.complex64)

        zero_weight, _ = apply_lgfc_frequency_correction(
            x,
            reference,
            maps,
            torch.zeros(4, 4),
            measurement,
            unsampled_weight=0.0,
        )
        full_kspace = compute_full_multicoil_kspace(x, maps)
        full_mask, stats = apply_lgfc_frequency_correction(
            x, reference, maps, torch.ones(4, 4), full_kspace
        )
        self.assertTrue(torch.equal(zero_weight, x))
        self.assertTrue(torch.equal(full_mask, x))
        self.assertEqual(float(stats["unsampled_k_difference_norm"]), 0.0)

    def test_output_is_finite_and_dc_residual_is_finite(self):
        torch.manual_seed(5)
        x = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
        reference = 1.2 * x
        maps = _complex_ones((1, 2, 4, 4))
        mask = torch.zeros(4, 4)
        measurement = torch.zeros((1, 2, 4, 4), dtype=torch.complex64)
        output, stats = apply_lgfc_frequency_correction(
            x, reference, maps, mask, measurement
        )
        residual = compute_dc_residual(output, maps, mask, measurement)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(residual).all())
        for value in stats.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_relative_update_clipping(self):
        x = _complex_ones((2, 1, 4, 4))
        reference = 100.0 * x
        maps = _complex_ones((2, 2, 4, 4))
        mask = torch.zeros(2, 1, 4, 4)
        measurement = torch.zeros((2, 2, 4, 4), dtype=torch.complex64)
        _, stats = apply_lgfc_frequency_correction(
            x,
            reference,
            maps,
            mask,
            measurement,
            unsampled_weight=1.0,
            max_relative_update=0.05,
        )
        self.assertTrue(torch.all(stats["update_was_clipped"]))
        self.assertTrue(torch.all(stats["raw_relative_update"] > 0.05))
        self.assertTrue(torch.all(stats["clipped_relative_update"] <= 0.050001))

    def test_dc_first_candidate_is_accepted(self):
        x = _complex_ones((1, 1, 4, 4))
        reference = 2.0 * x
        maps = _complex_ones((1, 1, 4, 4))
        mask = torch.zeros(4, 4)
        measurement = torch.zeros((1, 1, 4, 4), dtype=torch.complex64)
        output, stats = apply_lgfc_frequency_correction(
            x, reference, maps, mask, measurement
        )
        self.assertTrue(bool(stats["correction_accepted"].item()))
        self.assertEqual(int(stats["backtrack_count"].item()), 0)
        self.assertEqual(float(stats["effective_scale"].item()), 1.0)
        self.assertFalse(torch.equal(output, x))

    def _controlled_backtrack(self, candidate_sampled_values):
        x = torch.zeros((1, 1, 2, 2), dtype=torch.complex64)
        reference = _complex_ones((1, 1, 2, 2))
        maps = _complex_ones((1, 1, 2, 2))
        mask = torch.zeros(2, 2)
        mask[0, 0] = 1
        measurement = torch.zeros((1, 1, 2, 2), dtype=torch.complex64)
        measurement[0, 0, 0, 0] = 1
        k_x = torch.zeros_like(measurement)
        k_reference = torch.zeros_like(measurement)
        k_reference[0, 0, 0, 1] = 1
        candidates = []
        for sampled_value in candidate_sampled_values:
            encoded = torch.zeros_like(measurement)
            encoded[0, 0, 0, 0] = sampled_value
            candidates.append(encoded)
        side_effect = [k_x, k_reference, *candidates]
        with mock.patch(
            "eval_lgfc.frequency_correction.compute_full_multicoil_kspace",
            side_effect=side_effect,
        ):
            return apply_lgfc_frequency_correction(
                x,
                reference,
                maps,
                mask,
                measurement,
                dc_growth_limit=1.0,
                backtrack_factor=0.5,
                max_backtracks=len(candidate_sampled_values) - 1,
            )

    def test_dc_backtracking_accepts_reduced_candidate(self):
        output, stats = self._controlled_backtrack([3.0, 1.5])
        self.assertTrue(bool(stats["correction_accepted"].item()))
        self.assertEqual(int(stats["backtrack_count"].item()), 1)
        self.assertAlmostEqual(float(stats["effective_scale"].item()), 0.5)
        self.assertFalse(torch.count_nonzero(output) == 0)

    def test_dc_backtracking_skips_when_all_candidates_fail(self):
        output, stats = self._controlled_backtrack([3.0, 3.0, 3.0, 3.0])
        self.assertFalse(bool(stats["correction_accepted"].item()))
        self.assertEqual(float(stats["effective_scale"].item()), 0.0)
        self.assertEqual(int(stats["backtrack_count"].item()), 3)
        self.assertEqual(torch.count_nonzero(output), 0)
        self.assertTrue(torch.equal(stats["dc_after"], stats["dc_before"]))

    def test_frequency_update_interface_has_no_ground_truth(self):
        names = set(inspect.signature(apply_lgfc_frequency_correction).parameters)
        forbidden = {"clean", "gt", "ground_truth", "full_kspace"}
        self.assertFalse(names.intersection(forbidden))


class _CountingNet:
    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def round_sigma(self, values):
        return values

    def __call__(self, x, sigma, pos, labels):
        del sigma, pos, labels
        self.calls += 1
        return 0.9 * x


class _DummyInverseOperator:
    def __init__(self):
        self.maps = _complex_ones((1, 2, 4, 4))
        self.mask = torch.zeros(1, 1, 4, 4)
        self.mask[..., ::2] = 1

    def adjoint(self, measurement):
        del measurement
        return _complex_ones((1, 1, 4, 4))

    def forward(self, image_two_channel):
        image = torch.view_as_complex(
            image_two_channel.permute(0, 2, 3, 1).contiguous()
        ).unsqueeze(1)
        return self.mask * torch.fft.fft2(self.maps * image, norm="ortho")


def _install_lightweight_baseline_dependencies():
    """Allow importing ``eval/recon.py`` without a local BART executable."""

    import dnnlib.util

    dnnlib.util.configure_bart = lambda *args, **kwargs: ""
    fake_bart = types.ModuleType("bart")
    fake_bart.bart = lambda *args, **kwargs: None
    sys.modules["bart"] = fake_bart

    fake_utils = types.ModuleType("utils")
    fake_utils.fftmod = lambda value: value
    fake_utils.makeFigures = lambda *args, **kwargs: (0, 0, 0, 0, 0, 0)
    sys.modules["utils"] = fake_utils

    fake_denoise = types.ModuleType("denoise_padding")
    fake_denoise.getIndices = lambda *args, **kwargs: []
    fake_denoise.denoisedFromPatches = lambda *args, **kwargs: None
    sys.modules["denoise_padding"] = fake_denoise


class DenoiserBudgetTests(unittest.TestCase):
    def test_lgfc_and_original_have_same_denoiser_call_count(self):
        _install_lightweight_baseline_dependencies()
        baseline_module = __import__("recon")

        def deterministic_indices(spaced, patches, pad, psize):
            del spaced, patches, pad
            return [[0, psize, 0, psize]]

        def counted_denoiser(net, x, sigma, pos, labels, indices, **kwargs):
            del indices, kwargs
            return net(x, sigma, pos, labels)

        measurement = torch.full((1, 2, 4, 4), 0.1, dtype=torch.complex64)
        common = dict(
            latents=torch.zeros((1, 1, 4, 4), dtype=torch.complex64),
            latents_pos=torch.zeros((1, 2, 4, 4)),
            inverseop=_DummyInverseOperator(),
            measurement=measurement,
            num_steps=2,
            inner_loops=2,
            pad=0,
            psize=2,
            device=torch.device("cpu"),
            randn_like=torch.zeros_like,
            zeta=0.1,
        )

        baseline_net = _CountingNet()
        with mock.patch.object(
            baseline_module, "getIndices", deterministic_indices
        ), mock.patch.object(
            baseline_module, "denoisedFromPatches", counted_denoiser
        ):
            baseline_module.dps2(net=baseline_net, **common)

        lgfc_net = _CountingNet()
        with mock.patch.object(
            recon_lgfc,
            "_load_patch_functions",
            return_value=(counted_denoiser, deterministic_indices),
        ):
            recon_lgfc.dps2_lgfc(
                net=lgfc_net,
                lgfc_config=LGFCConfig(
                    enabled=True,
                    reference_mode="ema",
                    start_outer=1,
                    correction_every=1,
                ),
                **common,
            )

        self.assertEqual(baseline_net.calls, 4)
        self.assertEqual(lgfc_net.calls, baseline_net.calls)


class CudaMemoryTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_peak_memory_is_recorded(self):
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        x = _complex_ones((1, 1, 32, 32), device=device)
        reference = 1.1 * x
        maps = _complex_ones((1, 4, 32, 32), device=device)
        mask = torch.zeros(32, 32, device=device)
        measurement = torch.zeros((1, 4, 32, 32), dtype=torch.complex64, device=device)
        output, _ = apply_lgfc_frequency_correction(
            x, reference, maps, mask, measurement
        )
        torch.cuda.synchronize(device)
        self.assertTrue(torch.isfinite(output).all())
        print(
            "CUDA max_memory_allocated_mb=",
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        )
        print(
            "CUDA max_memory_reserved_mb=",
            torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
