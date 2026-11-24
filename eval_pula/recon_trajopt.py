"""PaDIS pULA with measurement-guided noisy-state proximal optimization.

SITCOM-*inspired* diagnostic proposal, NOT an implementation of SITCOM itself.

Fixed47 and K=5.  In early levels, all 5 slots are original pULA.  In
late levels, slots 0-2 are original pULA, slot 3 is a *single* VJP
measurement-guided optimization of the DENOISER INPUT with a trust region,
and slot 4 denoises this optimized noisy state before taking a full stochastic
pULA step.  We never replace the pULA stochastic state update with a clean
image or a deterministic Gaussian bridge.

Counting with first 23 original levels:
  115 early pULA + 72 late pre-pULA + 24 late final pULA = 211 pULA
  24 VJP optimized-input steps = 24 additional preconditioned CG
  235 denoiser forward calls, 24 network backward passes, 236 CG including init.

A one-step optimizer uses a first-order proximal surrogate, not a converged
inner-loop minimizer.  The correct claim is 'trust-region constrained single
preconditioned gradient step'; no exact posterior/stationary distribution is
claimed.
"""

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
import math

import torch
import torch.nn.functional as F

from eval_pula.score_opa import real_channels

NUM_LEVELS = 47
STEPS = 5


def rms(x):
    return x.abs().square().mean().sqrt().item()


def require_finite(name, x):
    if not bool(torch.isfinite(x).all().item()):
        raise FloatingPointError(f"{name}: NaN/Inf")


@torch.enable_grad()
def optimize_noisy_state(system, oracle, state, sigma, shift,
                         *, step_scale=1.0, trust_sigma=0.5):
    """One trust-limited preconditioned step on the noisy denoiser INPUT.

    Loss (complex Euclidean, per-real-channel whitening):
        L(u) = 0.5 ||A_w D_pi(u,sigma)-y_w||^2.
    At u=state, g=J_D^H A_w^H (A_w D_pi(u)-y_w).
    We solve (A_w^H A_w + sigma^-2 I) delta = -g, and then clip the
    step's complex RMS to `trust_sigma * sigma`.  This approximates a
    proximal subproblem which keeps u near the incoming noisy state.

    Important: final denoising on optimized u is a NEW forward evaluation
    performed outside this function.  No optimizer graph is propagated across
    pULA updates or sigma levels.
    """
    if not (math.isfinite(sigma) and sigma > 0 and
            math.isfinite(step_scale) and step_scale > 0 and
            math.isfinite(trust_sigma) and trust_sigma > 0):
        raise ValueError("sigma, step_scale and trust_sigma must be finite and positive")
    if not hasattr(oracle, "denoise_padded"):
        raise TypeError("Optimizer requires PatchScore.denoise_padded with autograd")
    if not all(hasattr(oracle, name) for name in ("pad", "image_size")):
        raise TypeError("Optimizer requires oracle.pad and oracle.image_size")

    u = state.detach().requires_grad_(True)
    padded = F.pad(u, (oracle.pad,) * 4)
    predicted_padded = oracle.denoise_padded(padded, sigma, shift)
    fov = slice(oracle.pad, oracle.pad + oracle.image_size)
    pred = predicted_padded[..., fov, fov]
    residual = system.forward(pred) - system.y
    # 0.5*sum |r|^2: real-variable Euclidean gradient is J_D^H A_w^H r.
    loss = 0.5 * residual.abs().square().sum()
    grad = torch.autograd.grad(loss, u, retain_graph=False, create_graph=False)[0]
    require_finite("noisy-input measurement gradient", grad)

    with torch.no_grad():
        # CG is an additional solve; it is NOT one of the pULA update CGs.
        direction, info = system.solve(-grad.detach(), torch.zeros_like(state), sigma)
        require_finite("preconditioned input step", direction)
        preclip = float(step_scale) * direction
        max_rms = float(trust_sigma) * sigma
        raw_rms = rms(preclip)
        factor = min(1.0, max_rms / raw_rms) if raw_rms > 0 else 1.0
        step = preclip * factor
        optimized = (state.detach() + step).detach()
        require_finite("optimized noisy state", optimized)
        diagnostics = {
            "sigma": float(sigma),
            "input_loss_half_sse": float(loss.detach().item()),
            "measurement_rms_before": rms(residual.detach()),
            "gradient_rms": rms(grad.detach()),
            "raw_step_rms": raw_rms,
            "step_rms": rms(step),
            "trust_radius_rms": max_rms,
            "step_clip_factor": factor,
            "cg_iterations": info["iterations"],
            "cg_relative_residual": info["relative_residual"],
        }
    return optimized, diagnostics, info


@torch.no_grad()
def reconstruct_trajopt(system, oracle, sigmas, shifts, gamma,
                        *, start_level=23, step_scale=1.0, trust_sigma=0.5,
                        report=print):
    """Fixed 47-level, 5-call/level pULA/noisy-input-optimization trajectory."""
    if system.convention != "padis_real":
        raise ValueError("This v1 controlled comparison uses padis_real only")
    if len(sigmas) != NUM_LEVELS or len(shifts) != STEPS:
        raise ValueError("Expected exactly 47 Karras levels and 5 cycle_k shifts")
    if not 0 <= start_level < NUM_LEVELS:
        raise ValueError("start_level must be in [0, 46]; 23 means first optimized level=24")
    if not (math.isfinite(gamma) and gamma > 0):
        raise ValueError("gamma must be positive")
    if any(float(sigmas[i]) <= float(sigmas[i+1]) for i in range(NUM_LEVELS-1)):
        raise ValueError("Sigma levels must strictly decrease")

    device = system.y.device
    system.reset_update()
    events = defaultdict(list)
    diagnostics = []
    pula_cg = []
    opt_cg = []
    network_forwards = 0
    network_backprops = 0
    pula_updates = 0
    opt_updates = 0

    @contextmanager
    def section(name):
        if device.type == "cuda":
            a,b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            yield
            b.record()
            events[name].append((a,b))
        else:
            a=perf_counter()
            yield
            events[name].append(perf_counter()-a)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = perf_counter()
    init_rhs, initial = system.initialization_rhs(float(sigmas[0]))
    with section("initialization_cg_seconds"):
        state, init_info = system.solve(init_rhs, initial, float(sigmas[0]))
    require_finite("initial state", state)

    def original_pula(x, pred, sigma):
        nonlocal pula_updates
        prior_score = (pred - x) / float(sigma)**2
        rhs, warm_start = system.update_rhs(x, prior_score, sigma, gamma)
        with section("pula_update_cg_seconds"):
            delta, info = system.solve(rhs, warm_start, sigma)
        updated = x + delta
        require_finite("pULA state", updated)
        pula_updates += 1
        pula_cg.append(info)
        return updated

    for level, value in enumerate(sigmas):
        sigma = float(value)
        if level < start_level:
            for shift in shifts:
                with section("network_forward_seconds"):
                    pred = oracle.denoise(state, sigma, shift)
                network_forwards += 1
                state = original_pula(state, pred, sigma)
            continue

        # Preserve 3 *sequential stochastic* pULA updates at this sigma.
        for shift in shifts[:3]:
            with section("network_forward_seconds"):
                pred = oracle.denoise(state, sigma, shift)
            network_forwards += 1
            state = original_pula(state, pred, sigma)

        # Fourth network call: one differentiable noisy-state optimization.
        with section("optimization_forward_backward_cg_seconds"):
            optimized, stats, info = optimize_noisy_state(
                system, oracle, state, sigma, shifts[3],
                step_scale=step_scale, trust_sigma=trust_sigma,
            )
        network_forwards += 1
        network_backprops += 1
        opt_updates += 1
        opt_cg.append(info)

        # Fifth network call: re-evaluate the prior at the UPDATED noisy state,
        # then take a full pULA stochastic update. This explicitly avoids
        # Bridge's clean-image state substitution/terminal collapse.
        with section("network_forward_seconds"):
            pred_updated = oracle.denoise(optimized, sigma, shifts[4])
        network_forwards += 1
        require_finite("optimized-input prediction", pred_updated)
        with torch.no_grad():
            after_residual = system.forward(pred_updated) - system.y
        state = original_pula(optimized, pred_updated, sigma)
        stats.update(
            level_index_zero_based=level,
            level_number_one_based=level+1,
            shift_for_input_optimization=list(shifts[3]),
            shift_for_pula_continuation=list(shifts[4]),
            measurement_rms_after_reprediction=rms(after_residual),
            output_state_change_rms=rms(state-optimized),
            network_forwards_cumulative=network_forwards,
        )
        diagnostics.append(stats)
        report(f"TrajOpt level {level+1:2d}/47 sigma={sigma:.6g} "
               f"meas_rms={stats['measurement_rms_before']:.5g}->"
               f"{stats['measurement_rms_after_reprediction']:.5g} "
               f"input_step={stats['step_rms']:.5g} "
               f"clip={stats['step_clip_factor']:.3g} "
               f"pULA_continuation=1")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    expected_opt = NUM_LEVELS-start_level
    expected_pula = start_level*STEPS + expected_opt*4
    if (network_forwards != 235 or network_backprops != expected_opt or
        pula_updates != expected_pula or opt_updates != expected_opt):
        raise RuntimeError("Noisy-state optimizer violated fixed47 235-NFE budget")
    if 1+len(pula_cg)+len(opt_cg) != 236:
        raise RuntimeError("CG count not equal to 236")

    timing={"reconstruction_seconds":perf_counter()-wall_start}
    for name, durations in events.items():
        timing[name] = (sum(a.elapsed_time(b)/1000 for a,b in durations)
                        if device.type=="cuda" else sum(durations))
    timing.update(
        sampler="pula_trust_noisy_state_optimization_v1",
        start_level=start_level,
        trust_sigma=trust_sigma,
        step_scale=step_scale,
        gamma=gamma,
        network_forwards=network_forwards,
        logical_denoiser_calls=network_forwards,
        network_backprops=network_backprops,
        pula_updates=pula_updates,
        optimized_input_updates=opt_updates,
        logical_updates=pula_updates+opt_updates,
        initialization_cg_solves=1,
        pula_cg_solves=len(pula_cg),
        optimization_cg_solves=len(opt_cg),
        cg_solves=1+len(pula_cg)+len(opt_cg),
        cg_iterations=init_info["iterations"]+sum(row["iterations"] for row in pula_cg+opt_cg),
        mri_forward_calls=system.forward_calls,
        mri_adjoint_calls=system.adjoint_calls,
        mri_operator_calls=system.forward_calls+system.adjoint_calls,
        sigma_levels=[float(s) for s in sigmas],
        opt_layers=diagnostics,
        initialization_cg=init_info,
        pula_update_cg=pula_cg,
        optimization_cg=opt_cg,
        no_clean_state_substitution=True,
    )
    if device.type=="cuda":
        timing.update(
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
        )
    return state.squeeze(1), timing
