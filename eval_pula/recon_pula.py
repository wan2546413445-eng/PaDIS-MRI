"""pULA in two real channels, stored as complex image and k-space tensors."""

from collections import defaultdict
from contextlib import contextmanager
from math import sqrt
from time import perf_counter

import torch
import torch.nn.functional as F

from eval_pula.cg import solve_cg
from eval_pula.score_opa import compare_grids, real_channels


def real_unit_noise(like, generator):
    # Native complex randn has variance 1/2 in each real channel.
    return sqrt(2.0) * torch.randn(like.shape, dtype=like.dtype, device=like.device, generator=generator)


def native_complex_noise(like, generator):
    """Match torch.randn_like(complex): variance 1/2 in each real channel."""
    return torch.randn(like.shape, dtype=like.dtype, device=like.device, generator=generator)


def whiten_measurement(operator, measurement, complex_covariance):
    """Whiten both SENSE maps and y to unit variance per real channel."""
    if isinstance(complex_covariance, float):
        weight = sqrt(2.0 / complex_covariance)
        maps, y = weight * operator.maps, weight * measurement
    else:
        covariance = complex_covariance.to(device=measurement.device, dtype=torch.complex128)
        coils = measurement.shape[1]
        if covariance.shape != (coils, coils) or not torch.allclose(covariance, covariance.mH):
            raise ValueError("Expected a Hermitian complex coil covariance with shape [coils, coils].")
        factor = torch.linalg.cholesky(covariance / 2).to(measurement.dtype)

        def apply(value):
            flat = value.reshape(coils, -1)
            return torch.linalg.solve_triangular(factor, flat, upper=False).reshape_as(value)

        maps, y = apply(operator.maps), apply(measurement)
    # A common binary mask commutes with the constant coil whitening matrix.
    return type(operator)(mask=operator.mask, maps=maps), y


class PULASystem:
    """B=W A and b=W y, with W defined by the real-channel noise covariance."""

    def __init__(self, operator, measurement, generator, cg_iters, cg_rtol):
        self.operator = operator
        self.y = measurement
        self.generator = generator
        self.cg_iters = cg_iters
        self.cg_rtol = cg_rtol
        self.forward_calls = 0
        self.adjoint_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return self.operator.forward(real_channels(x))

    def adjoint(self, y):
        self.adjoint_calls += 1
        return self.operator.adjoint(y)

    def solve(self, rhs, initial, sigma):
        def normal(v):
            return self.adjoint(self.forward(v)) + v / sigma ** 2

        return solve_cg(normal, rhs, initial, self.cg_iters, self.cg_rtol)

    def noise_rhs(self, template, sigma):
        measurement_noise = real_unit_noise(self.y, self.generator)
        image_noise = real_unit_noise(template, self.generator)
        rhs = self.adjoint(measurement_noise) + image_noise / sigma
        return rhs, image_noise

    def initialization_rhs(self, sigma):
        mean_rhs = self.adjoint(self.y)
        noise, image_noise = self.noise_rhs(mean_rhs, sigma)
        return mean_rhs + noise, sigma * image_noise

    def update_rhs(self, x, prior_score, sigma, gamma):
        likelihood = self.adjoint(self.y - self.forward(x))
        noise, image_noise = self.noise_rhs(x, sigma)
        diffusion = sqrt(2.0 * gamma)
        rhs = gamma * (likelihood + prior_score) + diffusion * noise
        # BART's state warm start expressed as an increment.
        initial = gamma * sigma ** 2 * prior_score + diffusion * sigma * image_noise
        return rhs, initial


@torch.no_grad()
def reconstruct(system, oracle, sigmas, shifts, gamma, *, diagnose_levels=(), diagnostic_shifts=(),
                terminal=False, report=print, pula_levels=None, dps_operator=None,
                dps_measurement=None, dps_sigmas=(), dps_shifts=(), dps_inner_loops=10,
                dps_zeta=3.0):
    """sigmas[0] initializes the chain, sigmas[1:] drive the Langevin levels."""
    events = defaultdict(list)
    device = system.y.device

    @contextmanager
    def section(name):
        if device.type == "cuda":
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            yield
            end.record()
            events[name].append((start, end))
        else:
            start = perf_counter()
            yield
            events[name].append(perf_counter() - start)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = perf_counter()
    network_calls = 0
    forward_timer = None

    def start_forward(*_):
        nonlocal forward_timer
        forward_timer = section("network_forward_seconds")
        forward_timer.__enter__()

    def finish_forward(*_):
        nonlocal network_calls, forward_timer
        forward_timer.__exit__(None, None, None)
        forward_timer = None
        network_calls += 1

    init_rhs, initial = system.initialization_rhs(sigmas[0])
    with section("initialization_cg_seconds"):
        state, init_info = system.solve(init_rhs, initial, sigmas[0])
    cg_records, diagnostics, dps_records = [], [], []
    dps_tail_seconds = 0.0
    diagnostic_calls = 0
    terminal_image = None
    selected_sigmas = sigmas[1:] if pula_levels is None else sigmas[1:pula_levels + 1]
    with oracle.net.register_forward_pre_hook(start_forward), oracle.net.register_forward_hook(finish_forward):
        for level, sigma in enumerate(selected_sigmas):
            for step, shift in enumerate(shifts):
                denoised = oracle.denoise(state, sigma, shift)
                if step == 0 and level in diagnose_levels:
                    before = network_calls
                    predictions = [denoised if other == shift else oracle.denoise(state, sigma, other)
                                   for other in diagnostic_shifts]
                    diagnostic_calls += network_calls - before
                    diagnostics.append({"level": level, **compare_grids(state, sigma, diagnostic_shifts, predictions)})
                score = (denoised - state) / sigma ** 2
                rhs, initial = system.update_rhs(state, score, sigma, gamma)
                with section("update_cg_seconds"):
                    delta, info = system.solve(rhs, initial, sigma)
                state = state + delta
                cg_records.append(info)
            report(f"pULA level {level + 1}/{len(selected_sigmas)} sigma={sigma:.5g} "
                   f"CG relative residual={info['relative_residual']}")

        if dps_sigmas:
            if dps_operator is None or dps_measurement is None:
                raise ValueError("DPS tail requires the original unwhitened MRI operator and measurement.")
            if len(dps_shifts) != len(dps_sigmas) * dps_inner_loops:
                raise ValueError("DPS tail shift count does not match tail levels and inner loops.")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dps_tail_start = perf_counter()
            padded = F.pad(state, (oracle.pad,) * 4)
            fov = slice(oracle.pad, oracle.pad + oracle.image_size)
            shift_index = 0
            with torch.enable_grad():
                for outer, sigma in enumerate(dps_sigmas):
                    sigma = float(sigma)
                    alpha = 0.5 * sigma ** 2
                    for inner in range(dps_inner_loops):
                        shift = dps_shifts[shift_index]
                        shift_index += 1
                        current = padded.detach().requires_grad_(True)
                        noisy = current + sigma * native_complex_noise(current, system.generator)
                        denoised = oracle.denoise_padded(noisy, sigma, shift)
                        denoised_fov = denoised[:, :, fov, fov]
                        residual = dps_measurement - dps_operator.forward(real_channels(denoised_fov))
                        residual_norm = torch.linalg.vector_norm(residual.reshape(residual.shape[0], -1), dim=-1)
                        sse = residual_norm.square()
                        with section("dps_backward_seconds"):
                            likelihood_grad = torch.autograd.grad(sse.sum(), current)[0]
                        score = (denoised - noisy) / sigma ** 2
                        updated = current - (dps_zeta / residual_norm[:, None, None, None]) * likelihood_grad
                        updated = updated + (alpha / 2) * score
                        if outer < len(dps_sigmas) - 1:
                            updated = updated + sqrt(alpha) * native_complex_noise(updated, system.generator)
                        padded = updated
                        dps_records.append({
                            "outer_level": outer,
                            "inner_step": inner,
                            "sigma": sigma,
                            "measurement_residual_norm": residual_norm.detach().cpu().tolist(),
                        })
                    report(f"DPS tail level {outer + 1}/{len(dps_sigmas)} sigma={sigma:.5g} "
                           f"residual={residual_norm.detach().mean().item():.6g}")
            state = padded[:, :, fov, fov].detach()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dps_tail_seconds = perf_counter() - dps_tail_start

        if terminal:
            before = network_calls
            terminal_sigma = float(dps_sigmas[-1]) if dps_sigmas else sigmas[-1]
            terminal_image = oracle.denoise(state, terminal_sigma, shifts[-1])
            terminal_calls = network_calls - before
        else:
            terminal_calls = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timing = {"reconstruction_seconds": perf_counter() - wall_start}
    for name, intervals in events.items():
        timing[name] = sum(a.elapsed_time(b) / 1000 for a, b in intervals) if device.type == "cuda" else sum(intervals)
    timing.update(
        cg_seconds=timing["initialization_cg_seconds"] + timing["update_cg_seconds"],
        network_forwards=network_calls,
        main_network_forwards=network_calls - diagnostic_calls - terminal_calls,
        diagnostic_network_forwards=diagnostic_calls, terminal_network_forwards=terminal_calls,
        langevin_updates=len(cg_records), cg_solves=1 + len(cg_records),
        cg_iterations=init_info["iterations"] + sum(row["iterations"] for row in cg_records),
        a_calls=system.forward_calls, ah_calls=system.adjoint_calls,
        network_backprops=len(dps_records), dps_updates=len(dps_records),
        dps_a_calls=len(dps_records), pula_levels=len(selected_sigmas),
        dps_tail_seconds=dps_tail_seconds,
        initialization_cg=init_info, update_cg=cg_records, dps_tail=dps_records,
    )
    if device.type == "cuda":
        timing.update(
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30,
        )
    return state.squeeze(1), None if terminal_image is None else terminal_image.squeeze(1), timing, diagnostics
