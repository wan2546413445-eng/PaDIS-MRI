"""pULA with clean-domain MRI consistency and a stochastic posterior bridge.

This module is intentionally separate from ``recon_pula.py``.  It reuses the
existing pULA system, whitening convention, patch denoiser, and matrix-free CG,
but does not modify any existing sampler.
"""

from collections import defaultdict
from contextlib import contextmanager
from math import sqrt
from time import perf_counter

import torch

from eval_pula.cg import solve_cg
from eval_pula.recon_pula import real_unit_noise


NUM_LEVELS = 47
UPDATES_PER_LEVEL = 5


def _rms(value):
    return value.abs().square().mean().sqrt().item()


def _norm(value):
    return torch.linalg.vector_norm(value).item()


def _finite(name, value):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def measurement_residual(system, image):
    """Return whitened MRI residual diagnostics, distinct from CG residuals."""
    residual = system.y - system.forward(image)
    _finite("whitened MRI residual", residual)
    return {
        "whitened_mri_residual_norm": _norm(residual),
        "whitened_mri_residual_rms": _rms(residual),
    }


@torch.no_grad()
def solve_clean_dc(system, denoised, tau):
    r"""Solve the clean-domain proximal MRI problem without forming A^H A.

    The incoming ``system`` already contains the ``padis_real`` whitened
    operator and measurement.  Consequently the system is exactly

        (A_w^H A_w + tau^-2 I) z = A_w^H y_w + tau^-2 denoised,

    with no additional measurement-noise scaling.
    """
    if not (tau > 0):
        raise ValueError("Bridge tau must be positive.")
    precision = 1.0 / tau ** 2
    rhs = system.adjoint(system.y) + precision * denoised

    def normal(value):
        return system.adjoint(system.forward(value)) + precision * value

    corrected, info = solve_cg(
        normal, rhs, denoised, system.cg_iters, system.cg_rtol,
    )
    _finite("clean-domain DC result", corrected)
    return corrected, info


@torch.no_grad()
def posterior_bridge(state, denoised, corrected, sigma_current, sigma_next,
                     eta, generator, mode="stochastic"):
    """Apply the approximate bridge transition and return its components.

    ``sigma_current`` and ``sigma_next`` use PaDIS per-real-channel units.
    The stochastic mode therefore draws noise whose real and imaginary parts
    each have unit variance.  This transition is an approximation and does not
    claim an exact posterior invariant distribution.
    """
    if mode not in ("stochastic", "deterministic"):
        raise ValueError(f"Unknown bridge mode: {mode}")
    if not (0 <= eta <= 1):
        raise ValueError("Bridge eta must lie in [0, 1].")
    if not (sigma_current > 0 and sigma_next >= 0):
        raise ValueError("Bridge sigmas must satisfy sigma_current>0 and sigma_next>=0.")

    zeros = torch.zeros_like(state)
    if mode == "deterministic" or sigma_next == 0:
        next_state = corrected
        carry = zeros
        injected = zeros
    else:
        carry_scale = sqrt(max(0.0, 1.0 - eta ** 2)) * sigma_next / sigma_current
        carry = carry_scale * (state - denoised)
        if eta == 0:
            injected = zeros
        else:
            injected = eta * sigma_next * real_unit_noise(state, generator)
        next_state = corrected + carry + injected

    _finite("bridge state", next_state)
    return next_state, {"carry": carry, "injected_noise": injected}


@torch.no_grad()
def reconstruct_bridge(system, oracle, sigmas, shifts, gamma, *,
                       bridge_start_level=23, bridge_eta=0.5,
                       bridge_tau_scale=1.0, bridge_mode="stochastic",
                       report=print):
    """Run the fixed 47x5 pULA/bridge trajectory.

    ``bridge_start_level`` is a count in user-facing 1-based language: a value
    of 23 keeps levels 1--23 as five-step pULA and starts the bridge at level 24.
    Python level indices are zero based, so the bridge condition is
    ``level_index >= bridge_start_level``.
    """
    if system.convention != "padis_real":
        raise ValueError("The first Bridge implementation is defined only for padis_real.")
    if len(sigmas) != NUM_LEVELS or len(shifts) != UPDATES_PER_LEVEL:
        raise ValueError("Bridge requires exactly 47 sigma levels and five shared shifts.")
    if not (0 <= bridge_start_level < NUM_LEVELS):
        raise ValueError("bridge_start_level must be in [0, 46].")
    if not (bridge_tau_scale > 0):
        raise ValueError("bridge_tau_scale must be positive.")
    if any(not (float(sigmas[i]) > float(sigmas[i + 1])) for i in range(NUM_LEVELS - 1)):
        raise ValueError("Bridge sigma levels must be strictly decreasing.")

    device = system.y.device
    events = defaultdict(list)
    bridge_records = []
    bridge_markers = []

    @contextmanager
    def section(name):
        if device.type == "cuda":
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            yield
            end.record()
            events[name].append((begin, end))
        else:
            begin = perf_counter()
            yield
            events[name].append(perf_counter() - begin)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        wall_origin = torch.cuda.Event(enable_timing=True)
        wall_origin.record()
    else:
        wall_origin = perf_counter()
    wall_start = perf_counter()

    network_calls = 0
    pula_updates = 0
    bridge_updates = 0
    pula_cg = []
    bridge_cg = []

    system.reset_update()
    init_rhs, initial = system.initialization_rhs(float(sigmas[0]))
    with section("initialization_cg_seconds"):
        state, init_info = system.solve(init_rhs, initial, float(sigmas[0]))
    _finite("initial state", state)

    for level, sigma_value in enumerate(sigmas):
        sigma_current = float(sigma_value)
        sigma_next = float(sigmas[level + 1]) if level + 1 < NUM_LEVELS else 0.0
        bridge_level = level >= bridge_start_level

        for step, shift in enumerate(shifts):
            state_before = state
            with section("network_forward_seconds"):
                denoised = oracle.denoise(state_before, sigma_current, shift)
            network_calls += 1
            _finite("patch denoised prediction", denoised)

            if not bridge_level or step < UPDATES_PER_LEVEL - 1:
                prior_score = (denoised - state_before) / sigma_current ** 2
                rhs, warm_start = system.update_rhs(
                    state_before, prior_score, sigma_current, gamma,
                )
                with section("pula_update_cg_seconds"):
                    delta, info = system.solve(rhs, warm_start, sigma_current)
                state = state_before + delta
                _finite("pULA state", state)
                pula_updates += 1
                pula_cg.append(info)
                continue

            tau = bridge_tau_scale * sigma_current
            residual_before = measurement_residual(system, denoised)
            with section("bridge_dc_cg_seconds"):
                corrected, info = solve_clean_dc(system, denoised, tau)
            residual_after = measurement_residual(system, corrected)
            state, components = posterior_bridge(
                state_before, denoised, corrected, sigma_current, sigma_next,
                bridge_eta, system.generator, bridge_mode,
            )
            bridge_updates += 1
            bridge_cg.append(info)

            record = {
                "level_index_zero_based": level,
                "level_number_one_based": level + 1,
                "sigma_current": sigma_current,
                "sigma_next": sigma_next,
                "eta": bridge_eta,
                "tau": tau,
                "bridge_mode": bridge_mode,
                "patch_shift": list(shift),
                "denoise_change_norm": _norm(denoised - state_before),
                "denoise_change_rms": _rms(denoised - state_before),
                **{f"dc_before_{key}": value for key, value in residual_before.items()},
                **{f"dc_after_{key}": value for key, value in residual_after.items()},
                "dc_correction_norm": _norm(corrected - denoised),
                "dc_correction_rms": _rms(corrected - denoised),
                "bridge_carry_rms": _rms(components["carry"]),
                "bridge_noise_rms": _rms(components["injected_noise"]),
                "state_transition_rms": _rms(state - state_before),
                "cg_linear_iterations": info["iterations"],
                "cg_linear_residual_norm": info["residual_norm"],
                "cg_linear_relative_residual": info["relative_residual"],
                "network_calls_cumulative": network_calls,
            }
            bridge_records.append(record)
            if device.type == "cuda":
                marker = torch.cuda.Event(enable_timing=True)
                marker.record()
                bridge_markers.append((record, marker))
            else:
                record["runtime_seconds_cumulative"] = perf_counter() - wall_origin
            report(
                f"Bridge level {level + 1}/{NUM_LEVELS} sigma={sigma_current:.6g} "
                f"next={sigma_next:.6g} tau={tau:.6g} "
                f"MRI residual RMS {residual_before['whitened_mri_residual_rms']:.6g}->"
                f"{residual_after['whitened_mri_residual_rms']:.6g}; "
                f"CG relative residual={info['relative_residual']}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        for record, marker in bridge_markers:
            record["runtime_seconds_cumulative"] = wall_origin.elapsed_time(marker) / 1000.0

    expected_bridge = NUM_LEVELS - bridge_start_level
    expected_pula = bridge_start_level * UPDATES_PER_LEVEL + expected_bridge * (UPDATES_PER_LEVEL - 1)
    expected_network = NUM_LEVELS * UPDATES_PER_LEVEL
    if (network_calls != expected_network or pula_updates != expected_pula
            or bridge_updates != expected_bridge):
        raise RuntimeError("Bridge sampler violated its fixed compute budget.")

    timing = {"reconstruction_seconds": perf_counter() - wall_start}
    for name, intervals in events.items():
        timing[name] = (
            sum(begin.elapsed_time(end) / 1000.0 for begin, end in intervals)
            if device.type == "cuda" else sum(intervals)
        )
    all_update_cg = pula_cg + bridge_cg
    timing.update(
        sampler="pula_clean_dc_stochastic_posterior_bridge_v1",
        bridge_mode=bridge_mode,
        bridge_start_level=bridge_start_level,
        bridge_start_index_zero_based=bridge_start_level,
        bridge_eta=bridge_eta,
        bridge_tau_scale=bridge_tau_scale,
        network_forwards=network_calls,
        main_network_forwards=network_calls,
        network_backprops=0,
        pula_updates=pula_updates,
        bridge_updates=bridge_updates,
        logical_updates=pula_updates + bridge_updates,
        update_stage_cg_solves=len(all_update_cg),
        initialization_cg_solves=1,
        cg_solves=1 + len(all_update_cg),
        cg_iterations=init_info["iterations"] + sum(row["iterations"] for row in all_update_cg),
        pula_cg_iterations=sum(row["iterations"] for row in pula_cg),
        bridge_cg_iterations=sum(row["iterations"] for row in bridge_cg),
        a_calls=system.forward_calls,
        ah_calls=system.adjoint_calls,
        mri_operator_calls=system.forward_calls + system.adjoint_calls,
        sigma_levels=[float(value) for value in sigmas],
        initialization_cg=init_info,
        pula_update_cg=pula_cg,
        bridge_update_cg=bridge_cg,
        bridge_layers=bridge_records,
    )
    timing["cg_seconds"] = (
        timing.get("initialization_cg_seconds", 0.0)
        + timing.get("pula_update_cg_seconds", 0.0)
        + timing.get("bridge_dc_cg_seconds", 0.0)
    )
    if device.type == "cuda":
        timing.update(
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30,
        )
    return state.squeeze(1), timing

