"""Annealed multi-partition consensus reconstruction for PaDIS MRI.

Pilot of a different posterior solver, NOT a pULA or exact stationary ADMM sampler.
At each of 47 fixed sigma levels it uses K=5 distinct patch partitions on
one shared image state, then ONE MRI quadratic CG solve.

'mean': mean of K denoised predictions followed by a quadratic DC step.
'dual': one annealed consensus/ADMM-style dual step per level with scaled and
        bounded partition residuals; sigma changes mean it is not classical ADMM.

Both modes use K*47=235 patch-denoiser calls and 47 update-CG solves.
"""
from __future__ import annotations

from math import isfinite
from time import perf_counter

import torch

from eval_pula.cg import solve_cg


@torch.no_grad()
def reconstruct_consensus(system, oracle, sigmas, shifts, gamma, *,
                          mode="mean", terminal=False, report=print,
                          save_handoff=None, pula_levels=None,
                          pula_shifts_by_level=None,
                          dps_sigmas=(), diagnose_levels=(),
                          **unused_legacy_kwargs):
    if mode not in ("mean", "dual"):
        raise ValueError(f"Invalid consensus mode: {mode}")
    if dps_sigmas:
        raise ValueError("Consensus solver must use all 47 levels, no DPS tail")
    if len(sigmas) != 48 or (pula_levels is not None and pula_levels != 47):
        raise ValueError("Consensus pilot requires the shared 47-level sigma schedule")
    if len(shifts) != 5 or pula_shifts_by_level is not None or diagnose_levels:
        raise ValueError("Consensus pilot requires five shared fixed patch shifts without extra diagnostics")
    if system.pula_variant != "baseline":
        raise ValueError("Consensus pilot requires baseline white-data CG convention")

    device = system.y.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = perf_counter()
    net_calls = 0

    def count_network(*_):
        nonlocal net_calls
        net_calls += 1

    # Reuse exactly the same stochastic perturb-and-solve initialization as pULA.
    init_rhs, initial = system.initialization_rhs(sigmas[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    state, init_info = system.solve(init_rhs, initial, sigmas[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    init_seconds = perf_counter() - start

    # The initialized system stores A_w^H y_w. No duplicate MRI-adjoint call.
    ahy = system.adjoint_template
    k = len(shifts)
    duals = [torch.zeros_like(state) for _ in shifts]
    prev_sigma = None
    records, cg_records = [], []
    network_seconds = 0.0
    cg_seconds = 0.0
    denoiser_evaluations = 0
    terminal_image = None

    with oracle.net.register_forward_hook(count_network):
        for level, sigma in enumerate(sigmas[1:]):
            sigma = float(sigma)
            sigma_system = system.system_sigma(sigma)
            lam = 1.0 / (sigma_system * sigma_system)
            rho = lam / k  # K*rho = 1/sigma_system^2; average-partition prior convention.

            if mode == "dual" and prev_sigma is not None:
                # Rescale Lagrange states as the denoiser's noise scale anneals.
                ratio = sigma_system / prev_sigma
                duals = [u * ratio for u in duals]

            # K distinct patch partitions see one common state per sigma level.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            predictions = [oracle.denoise(state + u if mode == "dual" else state,
                                          sigma, shift)
                           for u, shift in zip(duals, shifts)]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            network_seconds += perf_counter() - start
            denoiser_evaluations += k

            mean_prediction = torch.stack(predictions, dim=0).mean(dim=0)
            if mode == "dual":
                consensus_target = mean_prediction - torch.stack(duals, dim=0).mean(dim=0)
            else:
                consensus_target = mean_prediction

            # Image domain and whitened MRI measurements are solved TOGETHER.
            # (A_w^H A_w + K*rho*I)x = A_w^H y_w + rho*sum_j(D_j - u_j)
            rhs = ahy + lam * consensus_target

            def normal(v):
                return system.adjoint(system.forward(v)) + lam * v

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            next_state, info = solve_cg(normal, rhs, state, system.cg_iters, system.cg_rtol)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cg_seconds += perf_counter() - start
            cg_records.append(info)

            # Diagnostics: patch-partition disagreement; not a learned uncertainty.
            disagreement_rms = torch.stack(
                [(p - mean_prediction).abs().square().mean() for p in predictions]
            ).mean().sqrt().item()
            change_rms = (next_state - state).abs().square().mean().sqrt().item()
            if mode == "dual":
                # ADMM-like scaled dual update. Pixelwise cap prevents one partition
                # from sending an arbitrarily large denoiser input next level.
                max_magnitude = 2.0 * sigma_system
                updated = []
                for u, p in zip(duals, predictions):
                    v = u + next_state - p
                    v = v * (max_magnitude / v.abs().clamp(min=max_magnitude))
                    updated.append(v)
                duals = updated
            state = next_state
            prev_sigma = sigma_system
            records.append(dict(level=level, sigma=sigma, rho=rho,
                                discrepancy_rms=disagreement_rms,
                                state_change_rms=change_rms,
                                cg_iterations=info["iterations"],
                                cg_relative_residual=info["relative_residual"]))
            report(f"consensus {mode} level {level + 1}/47 "
                   f"sigma={sigma:.6g} patch_disagree={disagreement_rms:.5g} "
                   f"change={change_rms:.5g} CG={info['iterations']}")

        if save_handoff is not None:
            save_handoff(state.squeeze(1))
        if terminal:
            terminal_image = oracle.denoise(state, float(sigmas[-1]), shifts[-1])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = perf_counter() - wall_start
    timing = dict(
        algorithm="annealed_multi_partition_consensus_v1",
        consensus_mode=mode,
        reconstruction_seconds=runtime,
        initialization_cg_seconds=init_seconds,
        update_cg_seconds=cg_seconds,
        network_inference_seconds=network_seconds,
        cg_seconds=init_seconds + cg_seconds,
        initialization_cg=init_info,
        update_cg=cg_records,
        consensus_levels=47,
        consensus_updates=47,
        consensus_partition_evaluations=denoiser_evaluations,
        logical_updates=47,
        logical_nfe=denoiser_evaluations,
        network_forwards=net_calls,
        main_network_forwards=net_calls - (1 if terminal else 0),
        diagnostic_network_forwards=0,
        terminal_network_forwards=(1 if terminal else 0),
        network_backprops=0,
        cg_solves=1 + len(cg_records),
        cg_iterations=init_info["iterations"] + sum(i["iterations"] for i in cg_records),
        a_calls=system.forward_calls,
        ah_calls=system.adjoint_calls,
        langevin_updates=0,
        dps_updates=0,
        dps_tail_seconds=0.0,
        pula_levels=0,
        pula_seconds=runtime,
        consensus_records=records,
        shared_partition_count=k,
        rng_policy="same_pula_gaussian_initialization_then_deterministic_consensus",
        sigma_schedule="shared_fixed47_karras_grid",
    )
    if device.type == "cuda":
        timing.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
    return state.squeeze(1), None if terminal_image is None else terminal_image.squeeze(1), timing, []
