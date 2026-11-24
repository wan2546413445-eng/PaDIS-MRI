#!/usr/bin/env python3
"""CPU numerical checks for the stochastic posterior bridge."""

import importlib

import torch

import eval_pula.recon_pula as original_sampler
from eval_pula.recon_bridge import (
    posterior_bridge,
    reconstruct_bridge,
    solve_clean_dc,
)
from eval_pula.recon_pula import real_unit_noise
from eval_pula.run import karras_schedule


def assert_close(actual, expected, *, atol=1e-6, rtol=1e-6, message=""):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        error = torch.max(torch.abs(actual - expected)).item()
        raise AssertionError(f"{message} max error={error}")


class SyntheticSenseSystem:
    """Small fully sampled complex SENSE operator with an analytic normal."""

    def __init__(self, maps, truth, *, cg_iters=30, cg_rtol=1e-10):
        self.maps = maps
        self.y = self.forward_raw(truth)
        self.cg_iters = cg_iters
        self.cg_rtol = cg_rtol
        self.forward_calls = 0
        self.adjoint_calls = 0

    def forward_raw(self, image):
        coil_images = self.maps * image
        return torch.fft.fft2(coil_images, norm="ortho")

    def forward(self, image):
        self.forward_calls += 1
        return self.forward_raw(image)

    def adjoint(self, kspace):
        self.adjoint_calls += 1
        coil_images = torch.fft.ifft2(kspace, norm="ortho")
        return torch.sum(self.maps.conj() * coil_images, dim=1, keepdim=True)


def test_clean_dc_synthetic_mri():
    torch.manual_seed(7)
    truth = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    maps = torch.tensor([0.8 + 0.1j, 0.3 - 0.5j], dtype=torch.complex64).view(1, 2, 1, 1)
    denoised = torch.randn_like(truth)
    tau = 0.37
    system = SyntheticSenseSystem(maps, truth)
    corrected, info = solve_clean_dc(system, denoised, tau)
    coil_power = maps.abs().square().sum(dim=1, keepdim=True)
    expected = (coil_power * truth + denoised / tau ** 2) / (coil_power + 1 / tau ** 2)
    assert_close(corrected, expected, atol=2e-5, rtol=2e-5,
                 message="synthetic complex MRI clean-DC")
    if info["relative_residual"] is None or info["relative_residual"] > 1e-6:
        raise AssertionError(f"CG linear residual too large: {info}")


def test_identity_analytic():
    truth = torch.tensor([[[[2 + 3j, -1 + 0.5j]]]], dtype=torch.complex64)
    denoised = torch.tensor([[[[0.2 - 0.4j, 1.2 + 0.7j]]]], dtype=torch.complex64)

    class IdentitySystem:
        y = truth
        cg_iters = 10
        cg_rtol = 1e-10
        forward_calls = 0
        adjoint_calls = 0

        def forward(self, value):
            self.forward_calls += 1
            return value

        def adjoint(self, value):
            self.adjoint_calls += 1
            return value

    tau = 0.5
    corrected, _ = solve_clean_dc(IdentitySystem(), denoised, tau)
    expected = (truth + denoised / tau ** 2) / (1 + 1 / tau ** 2)
    assert_close(corrected, expected, message="identity clean-DC")


def test_bridge_eta_endpoints_and_final_layer():
    state = torch.tensor([[[[1 + 2j]]]], dtype=torch.complex64)
    denoised = torch.tensor([[[[0.2 - 0.1j]]]], dtype=torch.complex64)
    corrected = torch.tensor([[[[0.8 + 0.4j]]]], dtype=torch.complex64)

    generator = torch.Generator().manual_seed(123)
    rng_before = generator.get_state().clone()
    out0, parts0 = posterior_bridge(
        state, denoised, corrected, 1.0, 0.4, 0.0, generator,
    )
    expected0 = corrected + 0.4 * (state - denoised)
    assert_close(out0, expected0, message="eta=0 transition")
    assert_close(parts0["injected_noise"], torch.zeros_like(state), message="eta=0 noise")
    if not torch.equal(rng_before, generator.get_state()):
        raise AssertionError("eta=0 unexpectedly consumed Gaussian RNG.")

    out1, parts1 = posterior_bridge(
        state, denoised, corrected, 1.0, 0.4, 1.0,
        torch.Generator().manual_seed(123),
    )
    assert_close(parts1["carry"], torch.zeros_like(state), message="eta=1 carry")
    if torch.count_nonzero(parts1["injected_noise"]).item() == 0:
        raise AssertionError("eta=1 did not retain the new Gaussian term.")
    assert_close(out1, corrected + parts1["injected_noise"], message="eta=1 transition")

    final, final_parts = posterior_bridge(
        state, denoised, corrected, 0.003, 0.0, 0.5,
        torch.Generator().manual_seed(123),
    )
    assert_close(final, corrected, message="last layer sigma_next=0")
    assert_close(final_parts["injected_noise"], torch.zeros_like(state), message="last layer noise")

    deterministic, _ = posterior_bridge(
        state, denoised, corrected, 1.0, 0.4, 0.5,
        torch.Generator().manual_seed(123), mode="deterministic",
    )
    assert_close(deterministic, corrected, message="deterministic control")


def test_padis_real_noise_variance():
    noise = real_unit_noise(
        torch.empty(300_000, dtype=torch.complex64),
        torch.Generator().manual_seed(123),
    )
    real_var = noise.real.var(unbiased=True).item()
    imag_var = noise.imag.var(unbiased=True).item()
    if abs(real_var - 1.0) > 0.02 or abs(imag_var - 1.0) > 0.02:
        raise AssertionError(f"padis_real variance mismatch: real={real_var}, imag={imag_var}")


class CountingSystem:
    def __init__(self):
        self.y = torch.zeros(1, 1, 2, 2, dtype=torch.complex64)
        self.generator = torch.Generator().manual_seed(123)
        self.cg_iters = 2
        self.cg_rtol = 1e-6
        self.convention = "padis_real"
        self.forward_calls = 0
        self.adjoint_calls = 0
        self.solve_calls = 0

    def reset_update(self):
        pass

    def initialization_rhs(self, sigma):
        return torch.zeros_like(self.y), torch.zeros_like(self.y)

    def update_rhs(self, state, score, sigma, gamma):
        return torch.zeros_like(state), torch.zeros_like(state)

    def solve(self, rhs, initial, sigma):
        self.solve_calls += 1
        return torch.zeros_like(initial), {
            "iterations": 0, "residual_norm": 0.0, "relative_residual": None,
        }

    def forward(self, value):
        self.forward_calls += 1
        return value

    def adjoint(self, value):
        self.adjoint_calls += 1
        return value


class CountingOracle:
    def __init__(self):
        self.calls = 0

    def denoise(self, state, sigma, shift):
        self.calls += 1
        return state


def test_fixed_budget_and_schedule():
    grid = karras_schedule(47, 0.003, 1.0, 7.0)
    grid[0], grid[-1] = 1.0, 0.003
    if len(grid) != 47 or grid[0] != 1.0 or grid[-1] != 0.003:
        raise AssertionError("fixed47 Karras grid endpoints changed.")
    system, oracle = CountingSystem(), CountingOracle()
    state, timing = reconstruct_bridge(
        system, oracle, grid, [(0, 0)] * 5, 0.5,
        bridge_start_level=23, bridge_eta=0.5, bridge_tau_scale=1.0,
        report=lambda _: None,
    )
    if not torch.isfinite(state).all().item():
        raise AssertionError("fixed-budget trajectory returned NaN or Inf.")
    expected = {
        "network_forwards": 235,
        "pula_updates": 211,
        "bridge_updates": 24,
        "update_stage_cg_solves": 235,
        "cg_solves": 236,
        "network_backprops": 0,
    }
    for key, value in expected.items():
        if timing[key] != value:
            raise AssertionError(f"{key}: expected {value}, got {timing[key]}")
    if oracle.calls != 235 or system.solve_calls != 212:
        raise AssertionError("Mock call counts are inconsistent with 1 init + 211 pULA solves.")
    if timing["bridge_layers"][-1]["sigma_next"] != 0.0:
        raise AssertionError("Final bridge layer did not use sigma_next=0.")


def test_nonfinite_rejected_and_original_sampler_unchanged():
    original_reconstruct = original_sampler.reconstruct
    importlib.reload(importlib.import_module("eval_pula.recon_bridge"))
    if original_sampler.reconstruct is not original_reconstruct:
        raise AssertionError("Importing Bridge modified the original pULA sampler.")
    nan = torch.tensor([[[[complex(float("nan"), 0)]]]], dtype=torch.complex64)
    zero = torch.zeros_like(nan)
    try:
        posterior_bridge(
            zero, zero, nan, 1.0, 0.5, 0.5,
            torch.Generator().manual_seed(123),
        )
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Bridge accepted a nonfinite state.")


def main():
    tests = [
        test_clean_dc_synthetic_mri,
        test_identity_analytic,
        test_bridge_eta_endpoints_and_final_layer,
        test_padis_real_noise_variance,
        test_fixed_budget_and_schedule,
        test_nonfinite_rejected_and_original_sampler_unchanged,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} Bridge CPU checks")


if __name__ == "__main__":
    main()

