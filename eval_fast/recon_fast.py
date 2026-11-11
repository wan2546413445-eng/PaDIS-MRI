from copy import copy
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval_fast.fh_covariance import FHPrecision


def _channels(image):
    return torch.view_as_real(image.squeeze(1)).permute(0, 3, 1, 2)


def _dps_update(
    net, x, sigma, alpha, latents_pos, inverseop, measurement, indices, fov, zeta, add_noise,
):
    from denoise_padding import denoisedFromPatches

    with torch.set_grad_enabled(True):
        x = x.detach().requires_grad_(True)
        x_noisy = x + sigma * torch.randn_like(x)
        denoised = denoisedFromPatches(
            net, _channels(x_noisy), sigma, latents_pos, None, indices,
            t_goal=0, wrong=False,
        )
        cropped = denoised[:, :, fov, fov]
        residual = measurement - inverseop.forward(cropped)
        sse = torch.norm(residual.flatten(1), dim=-1) ** 2
        likelihood_grad = torch.autograd.grad(sse.sum(), x)[0]

    prediction = torch.complex(denoised[:, 0], denoised[:, 1]).unsqueeze(1)
    score = (prediction - x_noisy) / sigma ** 2
    prediction_fov = prediction[:, :, fov, fov]
    residual_norm = sse.clamp_min(torch.finfo(sse.dtype).tiny).sqrt()
    x = x - (zeta / residual_norm[:, None, None, None]) * likelihood_grad
    x = x + (alpha / 2) * score
    if add_noise:
        x = x + torch.sqrt(alpha) * torch.randn_like(x)
    return x, prediction_fov


def _cg_correction(
    inverseop, adjoint_y, prediction, cg_lambda, cg_iters, *,
    precision=None, noise_var=None, counts=None, profile=None,
):
    """Solve (A* A + regularizer) delta = A* y - A* A prediction."""
    cg_start = profile.begin() if profile else None
    residual = adjoint_y - inverseop.adjoint(inverseop.forward(_channels(prediction)))
    delta = torch.zeros_like(residual)
    direction = residual
    residual_sq = torch.vdot(residual.flatten(), residual.flatten()).real
    tiny = torch.finfo(residual_sq.dtype).tiny

    for _ in range(cg_iters):
        normal = inverseop.adjoint(inverseop.forward(_channels(direction)))
        if precision is None:
            normal = normal + cg_lambda * direction
        else:
            precision_start = profile.begin() if profile else None
            regularized = precision.apply(_channels(direction)[0])
            if profile:
                profile.end("cg_precision", precision_start)
            regularized = torch.complex(regularized[0], regularized[1])[None, None]
            normal = normal + noise_var * regularized
        curvature = torch.vdot(direction.flatten(), normal.flatten()).real
        step = residual_sq / curvature.clamp_min(tiny)
        delta = delta + step * direction
        residual = residual - step * normal
        next_sq = torch.vdot(residual.flatten(), residual.flatten()).real
        direction = residual + (next_sq / residual_sq.clamp_min(tiny)) * direction
        residual_sq = next_sq

    if profile:
        profile.end("cg_total", cg_start)
    if counts is not None:
        counts["cg_solves"] += 1
        counts["cg_iterations"] += cg_iters
        if precision is not None:
            counts["precision_applies"] += cg_iters
    return delta


@torch.no_grad()
def reconstruct_ct_cg_ddim(
    net, latents_pos, inverseop, measurement, *,
    num_steps, inner_loops, pad, psize, ddim_eta, cg_lambda, cg_iters,
    cg_gain=1.0, counts=None, profile=None, diagnostics=None,
):
    from denoise_padding import denoisedFromPatches, getIndices
    from recon import _ve_base_schedule

    adjoint_y = inverseop.adjoint(measurement)
    width = adjoint_y.shape[-1]
    fov = slice(pad, pad + width)
    padding = (pad, pad, pad, pad)
    x = F.pad(adjoint_y, padding)
    patches = width // psize + 1
    spaced = np.arange(patches) * psize
    schedule = _ve_base_schedule(net, num_steps, 0.003, 10.0, 7.0, x.device).float()
    x = x + schedule[0] * torch.randn_like(x)
    epsilon_scale = sqrt(inner_loops)
    carry_weight = sqrt(1 - ddim_eta ** 2)

    levels = zip(schedule[:-1], schedule[1:])
    for outer, (sigma, sigma_next) in enumerate(tqdm(levels, total=num_steps, desc="ct_cg_ddim")):
        prediction_sum = torch.zeros_like(x)
        epsilon_sum = torch.zeros_like(x)
        for inner in range(inner_loops):
            start = profile.begin() if profile else None
            indices = getIndices(spaced, patches, pad, psize)
            denoised = denoisedFromPatches(
                net, _channels(x), sigma, latents_pos, None, indices, t_goal=0, wrong=False,
            )
            if profile:
                profile.end("denoise", start)
            if counts is not None:
                counts["main_denoise"] += 1

            prediction = torch.complex(denoised[:, 0], denoised[:, 1]).unsqueeze(1)
            prediction_sum.add_(prediction)
            epsilon_sum.add_((x - prediction) / sigma)
            if inner + 1 < inner_loops:
                # 从本轮干净估计恢复当前噪声水平，供下一轮换网格去噪。
                x = prediction + sigma * torch.randn_like(prediction)

        # MRI 测量算子定义在真实 FOV 上，先裁掉分块用的 padding 再做 CG。
        v = prediction_sum[:, :, fov, fov] / inner_loops
        delta = _cg_correction(
            inverseop, adjoint_y, v, cg_lambda, cg_iters, counts=counts, profile=profile,
        )
        v_dc = v + cg_gain * delta
        if diagnostics is not None and outer + 1 == num_steps:
            diagnostics["pre_last_cg"] = v.squeeze(1).detach().cpu()
        if outer + 1 < num_steps:
            # 下一层以测量修正后的干净估计为起点，噪声仍由 CG 前的各轮预测融合。
            epsilon = carry_weight * (epsilon_sum / epsilon_scale)
            if ddim_eta:
                epsilon = epsilon + ddim_eta * torch.randn_like(epsilon)
            x = F.pad(v_dc, padding) + sigma_next * epsilon

    return v_dc.squeeze(1)


@torch.no_grad()
def reconstruct_mri_recurrent_cg(
    net, latents_pos, inverseop, measurement, *,
    num_steps, recurrent_branches, pad, psize, ddim_eta, cg_lambda, cg_iters, cg_gain,
    counts, recurrent_depth=2, profile=None, diagnostics=None,
    dps_tail_steps=0, dps_inner_loops=10, zeta=3.0,
):
    from denoise_padding import denoisedFromPatches, getIndices
    from recon import _ve_base_schedule

    adjoint_y = inverseop.adjoint(measurement)
    width = adjoint_y.shape[-1]
    fov = slice(pad, pad + width)
    padding = (pad,) * 4
    x = F.pad(adjoint_y, padding)
    patches = width // psize + 1
    spaced = np.arange(patches) * psize
    schedule = _ve_base_schedule(net, num_steps, 0.003, 10.0, 7.0, x.device).float()
    x = x + schedule[0] * torch.randn_like(x)
    carry_weight = sqrt(1 - ddim_eta ** 2)

    split = num_steps - dps_tail_steps
    levels = zip(schedule[:split], schedule[1:split + 1])
    stage = "mri_recurrent_hybrid CG" if dps_tail_steps else "mri_recurrent_cg"
    for outer, (sigma, sigma_next) in enumerate(tqdm(levels, total=split, desc=stage)):
        last_recurrent = outer + 1 == split
        prediction_sum = torch.zeros_like(x)
        epsilon_sum = torch.zeros_like(x)
        for branch in range(recurrent_branches):
            # 每个分支从同一带噪状态开始，中间轮用于同层重加噪。
            branch_x = x
            for sweep in range(recurrent_depth):
                start = profile.begin() if profile else None
                indices = getIndices(spaced, patches, pad, psize)
                denoised = denoisedFromPatches(
                    net, _channels(branch_x), sigma, latents_pos, None, indices,
                    t_goal=0, wrong=False,
                )
                if profile:
                    profile.end("denoise", start)
                counts["main_denoise"] += 1

                start = profile.begin() if profile else None
                prediction = torch.complex(denoised[:, 0], denoised[:, 1]).unsqueeze(1)
                if sweep + 1 < recurrent_depth:
                    branch_x = prediction + sigma * torch.randn_like(prediction)
                else:
                    prediction_sum.add_(prediction)
                    epsilon_sum.add_((branch_x - prediction) / sigma)
                if profile:
                    profile.end("recurrent_fusion", start)
                if last_recurrent and diagnostics is not None:
                    diagnostics[f"branch_{branch + 1}_round_{sweep + 1}"] = prediction[:, 0, fov, fov]

        start = profile.begin() if profile else None
        # 多线圈测量算子定义在真实 FOV 内。
        average = prediction_sum[:, :, fov, fov] / recurrent_branches
        epsilon = epsilon_sum / recurrent_branches
        if profile:
            profile.end("recurrent_fusion", start)

        if not last_recurrent or dps_tail_steps:
            delta = _cg_correction(
                inverseop, adjoint_y, average, cg_lambda, cg_iters, counts=counts, profile=profile,
            )
            corrected = average + cg_gain * delta
            if last_recurrent:
                # DPS 会自行加当前层噪声，交接时只传干净的 CG 修正图。
                x = F.pad(corrected, padding)
                if diagnostics is not None:
                    diagnostics["recurrent_terminal"] = average.squeeze(1)
                    diagnostics["recurrent_after_cg"] = corrected.squeeze(1)
                continue
            start = profile.begin() if profile else None
            # CG 修正干净图，跨层噪声仍取自去噪器的预测残差。
            epsilon = carry_weight * epsilon
            if ddim_eta:
                epsilon = epsilon + ddim_eta * torch.randn_like(epsilon)
            x = F.pad(corrected, padding) + sigma_next * epsilon
            if profile:
                profile.end("ve_ddim", start)

    if dps_tail_steps:
        counts["recurrent_main_denoise"] = counts["main_denoise"]
        for outer, sigma in enumerate(tqdm(schedule[split:-1], desc="mri_recurrent_hybrid DPS"), start=split):
            alpha = 0.5 * sigma ** 2
            for _ in range(dps_inner_loops):
                indices = getIndices(spaced, patches, pad, psize)
                x, _ = _dps_update(
                    net, x, sigma, alpha, latents_pos, inverseop, measurement, indices, fov, zeta,
                    add_noise=outer < num_steps - 1,
                )
                counts["main_denoise"] += 1
                counts["dps_backprops"] += 1
        counts["dps_main_denoise"] = counts["main_denoise"] - counts["recurrent_main_denoise"]
        result = x[:, :, fov, fov].squeeze(1)
        if diagnostics is not None:
            diagnostics["final_dps"] = result
    else:
        result = average.squeeze(1)
        if diagnostics is not None:
            diagnostics["terminal_average"] = result
    return result

@torch.no_grad()
def reconstruct(
    net, latents_pos, inverseop, measurement, *,
    num_steps, inner_loops, pad, psize, mode, cg_lambda, cg_iters,
    cg_gain=1.0, dps_tail_steps=0, zeta=3.0,
    fh_variance=None, fh_noise_var=None, fh_max_rank=16, counts=None, profile=None,
    diagnostics=None,
):
    from denoise_padding import denoisedFromPatches, getIndices
    from recon import _ve_base_schedule

    adjoint_y = inverseop.adjoint(measurement)
    width = adjoint_y.shape[-1]
    fov = slice(pad, pad + width)
    padding = (pad, pad, pad, pad)
    x = F.pad(adjoint_y, padding)
    patches = width // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)
    schedule = _ve_base_schedule(net, num_steps, 0.003, 10.0, 7.0, x.device)
    layer_dc = mode == "cg_layer"
    covariance = None
    previous_x_fov = None
    if fh_variance is not None:
        skip_reasons = {}
        if counts is not None:
            counts["bfgs_skip_reasons"] = skip_reasons
        fh_sigmas = schedule[:num_steps - dps_tail_steps].float().tolist()
        start = profile.begin() if profile else None
        covariance = FHPrecision(fh_variance.to(x.real), 0.5 * fh_sigmas[0] ** 2, fh_max_rank)
        if profile:
            profile.end("covariance_update", start)

    for outer, sigma in enumerate(tqdm(schedule[:-1], desc=mode)):
        sigma = sigma.float()
        alpha = 0.5 * sigma ** 2
        use_dps = mode == "hybrid" and outer >= num_steps - dps_tail_steps
        use_fh = covariance is not None and not use_dps
        if use_fh:
            sigma_value = fh_sigmas[outer]
            noise_variance = 0.5 * sigma_value ** 2
            if outer:
                start = profile.begin() if profile else None
                covariance.transfer(noise_variance)
                if profile:
                    profile.end("covariance_update", start)
                if counts is not None:
                    counts["covariance_transfers"] += 1
        if layer_dc:
            prediction_sum = torch.zeros_like(adjoint_y)

        for inner in range(inner_loops):
            indices = getIndices(spaced, patches, pad, psize)
            if use_dps:
                x, prediction_fov = _dps_update(
                    net, x, sigma, alpha, latents_pos, inverseop, measurement, indices, fov, zeta,
                    add_noise=outer < num_steps - 1,
                )
                if counts is not None:
                    counts["main_denoise"] += 1
                    counts["dps_backprops"] += 1
                continue

            x_noisy = x + sigma * torch.randn_like(x)
            denoised = denoisedFromPatches(
                net, _channels(x_noisy), sigma, latents_pos, None, indices,
                t_goal=0, wrong=False,
            )
            if counts is not None:
                counts["main_denoise"] += 1

            if use_fh:
                x_fov = x[:, :, fov, fov]
                if inner == 0 and 1 <= sigma_value <= 5:
                    skip_reason = "no_previous_state"
                    if previous_x_fov is not None:
                        skip_reason = "invalid_displacement"
                        displacement = x_fov - previous_x_fov
                        rms = _channels(displacement).square().mean().sqrt()
                        if bool(torch.isfinite(rms) & (rms > 0)):
                            s = displacement * (0.01 * sigma_value / rms)
                            probe_input = x_noisy + F.pad(s, padding)
                            # denoisedFromPatches draws noise even at t_goal=0.
                            devices = [x.device] if x.is_cuda else []
                            with torch.random.fork_rng(devices=devices):
                                start = profile.begin() if profile else None
                                probe = denoisedFromPatches(
                                    net, _channels(probe_input), sigma, latents_pos,
                                    None, indices, t_goal=0, wrong=False,
                                )
                                if profile:
                                    profile.end("probe_forward", start)
                            if counts is not None:
                                counts["probe_denoise"] += 1
                            start = profile.begin() if profile else None
                            actual_s = (probe_input - x_noisy)[:, :, fov, fov]
                            q = noise_variance * (probe - denoised)[:, :, fov, fov]
                            skip_reason = covariance.update(_channels(actual_s)[0], q[0])
                            if profile:
                                profile.end("covariance_update", start)
                    if counts is not None:
                        if skip_reason is None:
                            counts["bfgs_accepted"] += 1
                        else:
                            counts["bfgs_skipped"] += 1
                            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                if inner == inner_loops - 1:
                    previous_x_fov = x_fov

            prediction = torch.complex(denoised[:, 0], denoised[:, 1]).unsqueeze(1)
            score = (prediction - x_noisy) / sigma ** 2
            prediction_fov = prediction[:, :, fov, fov]

            if layer_dc:
                prediction_sum.add_(prediction_fov)
            else:
                delta = _cg_correction(
                    inverseop, adjoint_y, prediction_fov, cg_lambda, cg_iters,
                    precision=covariance, noise_var=fh_noise_var, counts=counts, profile=profile,
                )
                if diagnostics is not None and use_fh and inner == 0 and outer + 1 in diagnostics.steps:
                    dct_precision = copy(covariance)
                    dct_precision.modes = covariance.modes[:, :0]
                    dct_precision.weights = covariance.weights[:0]
                    isotropic = _cg_correction(inverseop, adjoint_y, prediction_fov, cg_lambda, cg_iters)
                    dct_only = _cg_correction(
                        inverseop, adjoint_y, prediction_fov, cg_lambda, cg_iters,
                        precision=dct_precision, noise_var=fh_noise_var,
                    )
                    diagnostics.record_corrections(outer + 1, sigma_value, isotropic, dct_only, delta)
                x = x + cg_gain * F.pad(delta, padding)

            x = x + (alpha / 2) * score
            if outer < num_steps - 1:
                x = x + torch.sqrt(alpha) * torch.randn_like(x)

        if layer_dc:
            delta = _cg_correction(
                inverseop, adjoint_y, prediction_sum / inner_loops, cg_lambda, cg_iters,
                counts=counts, profile=profile,
            )
            # Carry only the measurement correction into the VE sampler state.
            x = x + cg_gain * F.pad(delta, padding)
        if diagnostics is not None and outer + 1 == diagnostics.cg_steps:
            diagnostics.record_transition(x[:, :, fov, fov], prediction_fov)

    return x[:, :, fov, fov].squeeze(1)
