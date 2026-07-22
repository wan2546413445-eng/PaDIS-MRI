"""PaDIS reconstruction loop with optional late-stage LGFC insertion."""

import csv
import importlib
import os
import sys
import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EVAL_DIR = os.path.join(_REPO_ROOT, "eval")
for _path in (_REPO_ROOT, _EVAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from eval_lgfc.frequency_correction import (
    LGFCConfig,
    apply_lgfc_frequency_correction,
    update_reference,
    validate_lgfc_config,
)


_RUN_PROFILE_PEAKS = {
    "run_peak_memory_allocated_mb": 0.0,
    "run_peak_memory_reserved_mb": 0.0,
}


def reset_lgfc_runtime_peaks() -> None:
    """Reset peaks preserved across per-correction CUDA peak-stat resets."""

    _RUN_PROFILE_PEAKS["run_peak_memory_allocated_mb"] = 0.0
    _RUN_PROFILE_PEAKS["run_peak_memory_reserved_mb"] = 0.0


def get_lgfc_runtime_peaks() -> dict:
    """Return the largest CUDA peaks observed by the enabled LGFC loop."""

    return dict(_RUN_PROFILE_PEAKS)


def _preserve_current_cuda_peaks(device: torch.device) -> None:
    allocated = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
    _RUN_PROFILE_PEAKS["run_peak_memory_allocated_mb"] = max(
        _RUN_PROFILE_PEAKS["run_peak_memory_allocated_mb"], allocated
    )
    _RUN_PROFILE_PEAKS["run_peak_memory_reserved_mb"] = max(
        _RUN_PROFILE_PEAKS["run_peak_memory_reserved_mb"], reserved
    )


def _load_baseline_dps2() -> Callable:
    """Load the current branch's original dps2 implementation lazily."""

    return importlib.import_module("recon").dps2


def _load_patch_functions() -> Tuple[Callable, Callable]:
    module = importlib.import_module("denoise_padding")
    return module.denoisedFromPatches, module.getIndices


def _ve_base_schedule(
    net,
    num_sigmas: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    """Mirror the VE schedule in ``eval/recon.py`` without changing it."""

    idx = torch.arange(num_sigmas, dtype=torch.float64, device=device)
    t = (
        sigma_max ** (1.0 / rho)
        + (idx / (num_sigmas - 1.0))
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t = net.round_sigma(t)
    return torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)], dim=0)


def _should_correct(outer_step: int, config: LGFCConfig) -> bool:
    return (
        outer_step >= config.start_outer
        and (outer_step - config.start_outer) % config.correction_every == 0
    )


def _tensor_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _write_diagnostics(rows: list[dict], save_dir: str, tag: Optional[str]) -> str:
    diagnostic_dir = os.path.join(save_dir, "lgfc_diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    safe_tag = tag if tag is not None else "sample"
    csv_path = os.path.join(diagnostic_dir, f"{safe_tag}_lgfc.csv")
    fieldnames = [
        "outer_step",
        "sigma",
        "reference_mode",
        "reference_to_state_norm",
        "unsampled_k_difference_norm",
        "raw_update_norm",
        "raw_relative_update",
        "clipped_update_norm",
        "clipped_relative_update",
        "update_was_clipped",
        "effective_scale",
        "backtrack_count",
        "correction_accepted",
        "dc_before",
        "dc_after",
        "dc_relative_change",
        "final_state_update_norm",
        "correction_wall_time_ms",
        "memory_allocated_before_mb",
        "memory_reserved_before_mb",
        "correction_peak_allocated_mb",
        "correction_peak_reserved_mb",
        "correction_extra_peak_allocated_mb",
        "correction_extra_peak_reserved_mb",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def dps2_lgfc(
    net,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: Optional[torch.Tensor],
    num_steps: int = 104,
    inner_loops: int = 10,
    sigma_min: float = 0.003,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    zeta: float = 3.0,
    pad: int = 64,
    psize: int = 64,
    randn_like=torch.randn_like,
    verbose: bool = False,
    clean: Optional[torch.Tensor] = None,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
    save_intermediate: bool = False,
    intermediate_every: int = 10,
    lgfc_config: Optional[LGFCConfig] = None,
    save_lgfc_diagnostics: bool = False,
    profile_memory: bool = False,
):
    """Run PaDIS with an optional post-inner-loop LGFC update.

    When LGFC is disabled, this function immediately dispatches to the
    original ``eval/recon.py::dps2``. When enabled, the original operation
    order is retained and only a detached current-outer-step temporal
    reference, one scheduled LGFC update, and scalar diagnostics are added.
    """

    config = validate_lgfc_config(lgfc_config or LGFCConfig(enabled=False))
    if not config.enabled:
        baseline_dps2 = _load_baseline_dps2()
        return baseline_dps2(
            net=net,
            latents=latents,
            latents_pos=latents_pos,
            inverseop=inverseop,
            measurement=measurement,
            num_steps=num_steps,
            inner_loops=inner_loops,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            zeta=zeta,
            pad=pad,
            psize=psize,
            randn_like=randn_like,
            verbose=verbose,
            clean=clean,
            device=device,
            save_dir=save_dir,
            tag=tag,
            save_intermediate=save_intermediate,
            intermediate_every=intermediate_every,
        )

    if measurement is None:
        raise ValueError("LGFC requires a measured MRI reconstruction")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if inner_loops < 1:
        raise ValueError("inner_loops must be at least 1")
    if (save_lgfc_diagnostics or profile_memory) and save_dir is None:
        raise ValueError("save_dir is required when LGFC diagnostics are requested")
    if not hasattr(inverseop, "maps") or not hasattr(inverseop, "mask"):
        raise TypeError("inverseop must expose maps and mask tensors for LGFC")

    denoised_from_patches, get_indices = _load_patch_functions()
    net.eval()
    w = latents.shape[-1]
    patches = w // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    x_init = inverseop.adjoint(measurement).detach()
    if x_init.ndim != 4 or x_init.shape[1] != 1 or not torch.is_complex(x_init):
        raise ValueError("inverseop.adjoint(measurement) must return complex [B,1,H,W]")
    height, width = x_init.shape[-2:]
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)
    t_steps = _ve_base_schedule(net, num_steps, sigma_min, sigma_max, rho, device)

    noisypsnr = denoisedpsnr = noisyssim = denoisedssim = 0.0
    noisynrmse = denoisednrmse = 0.0
    diagnostics_rows: list[dict] = []
    intermediate_rows: list[dict] = []
    intermediate_dir = None
    safe_tag = tag if tag is not None else "sample"
    intermediate_every = max(1, int(intermediate_every))
    if save_intermediate and save_dir is not None:
        intermediate_dir = os.path.join(save_dir, "intermediate_vis", safe_tag)
        os.makedirs(intermediate_dir, exist_ok=True)

    for i, (t_cur, t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])),
        total=len(t_steps) - 1,
    ):
        del t_next
        outer_step = i + 1
        reference = None
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        for _ in range(inner_loops):
            indices = get_indices(spaced, patches, pad, psize)
            x = x.detach().requires_grad_(True)
            x_noisy = x + (t_cur * randn_like(x))
            x_real_noisy = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
            d_real = denoised_from_patches(
                net,
                x_real_noisy,
                t_cur,
                latents_pos,
                None,
                indices,
                t_goal=0,
                wrong=False,
            )

            d_cplx = torch.complex(d_real[:, 0], d_real[:, 1]).unsqueeze(1)
            d_fov = d_cplx[:, :, pad : pad + height, pad : pad + width].detach()
            reference = update_reference(
                reference,
                d_fov,
                mode=config.reference_mode,
                ema_beta=config.ema_beta,
            )

            score = (d_cplx - x_noisy) / (t_cur ** 2)
            cropped_x0hat = (
                d_real
                if pad == 0
                else d_real[:, :, pad : pad + w, pad : pad + w]
            )
            ax = inverseop.forward(cropped_x0hat)
            residual = measurement - ax
            residual_flat = residual.reshape(x.shape[0], -1)
            sse_ind = torch.norm(residual_flat, dim=-1) ** 2
            sse = torch.sum(sse_ind)
            likelihood_grad = torch.autograd.grad(outputs=sse, inputs=x)[0]
            x = x - (
                zeta / torch.sqrt(sse_ind)[:, None, None, None]
            ) * likelihood_grad

            if i < num_steps - 1:
                x = x + (alpha / 2) * score + torch.sqrt(alpha) * randn_like(x)
            else:
                x = x + (alpha / 2) * score

            if verbose:
                with torch.no_grad():
                    print(
                        f"step {outer_step}/{num_steps} -> "
                        f"||x||={torch.linalg.norm(x).item():.4f}"
                    )

        if _should_correct(outer_step, config):
            if reference is None:
                raise RuntimeError("LGFC reference was not constructed")
            x_fov = x[:, :, pad : pad + height, pad : pad + width].detach()
            with torch.no_grad():
                if profile_memory and x.is_cuda:
                    torch.cuda.synchronize(x.device)
                    _preserve_current_cuda_peaks(x.device)
                    allocated_before_mb = torch.cuda.memory_allocated(x.device) / (
                        1024.0 ** 2
                    )
                    reserved_before_mb = torch.cuda.memory_reserved(x.device) / (
                        1024.0 ** 2
                    )
                    torch.cuda.reset_peak_memory_stats(x.device)
                else:
                    allocated_before_mb = float("nan")
                    reserved_before_mb = float("nan")

                start_time = time.perf_counter()
                x_corrected, correction_stats = apply_lgfc_frequency_correction(
                    x_fov=x_fov,
                    reference=reference,
                    maps=inverseop.maps,
                    mask=inverseop.mask,
                    measurement=measurement,
                    unsampled_weight=config.unsampled_weight,
                    max_relative_update=config.max_relative_update,
                    dc_growth_limit=config.dc_growth_limit,
                    backtrack_factor=config.backtrack_factor,
                    max_backtracks=config.max_backtracks,
                    eps=config.eps,
                )
                # Re-embed the corrected center FOV before reading CUDA peaks so
                # the reported LGFC increment includes the padded-state clone.
                x_next = x.detach().clone()
                x_next[:, :, pad : pad + height, pad : pad + width] = x_corrected

                if x.is_cuda:
                    torch.cuda.synchronize(x.device)
                wall_time_ms = (time.perf_counter() - start_time) * 1000.0

                if profile_memory and x.is_cuda:
                    peak_allocated_mb = torch.cuda.max_memory_allocated(x.device) / (
                        1024.0 ** 2
                    )
                    peak_reserved_mb = torch.cuda.max_memory_reserved(x.device) / (
                        1024.0 ** 2
                    )
                    _preserve_current_cuda_peaks(x.device)
                    extra_allocated_mb = max(0.0, peak_allocated_mb - allocated_before_mb)
                    extra_reserved_mb = max(0.0, peak_reserved_mb - reserved_before_mb)
                else:
                    peak_allocated_mb = float("nan")
                    peak_reserved_mb = float("nan")
                    extra_allocated_mb = float("nan")
                    extra_reserved_mb = float("nan")

            diagnostics_rows.append(
                {
                    "outer_step": outer_step,
                    "sigma": _tensor_scalar(t_cur),
                    "reference_mode": config.reference_mode,
                    "reference_to_state_norm": _tensor_scalar(
                        correction_stats["reference_to_state_norm"]
                    ),
                    "unsampled_k_difference_norm": _tensor_scalar(
                        correction_stats["unsampled_k_difference_norm"]
                    ),
                    "raw_update_norm": _tensor_scalar(
                        correction_stats["raw_update_norm"]
                    ),
                    "raw_relative_update": _tensor_scalar(
                        correction_stats["raw_relative_update"]
                    ),
                    "clipped_update_norm": _tensor_scalar(
                        correction_stats["clipped_update_norm"]
                    ),
                    "clipped_relative_update": _tensor_scalar(
                        correction_stats["clipped_relative_update"]
                    ),
                    "update_was_clipped": bool(
                        correction_stats["update_was_clipped"].cpu().item()
                    ),
                    "effective_scale": _tensor_scalar(
                        correction_stats["effective_scale"]
                    ),
                    "backtrack_count": int(
                        correction_stats["backtrack_count"].cpu().item()
                    ),
                    "correction_accepted": bool(
                        correction_stats["correction_accepted"].cpu().item()
                    ),
                    "dc_before": _tensor_scalar(correction_stats["dc_before"]),
                    "dc_after": _tensor_scalar(correction_stats["dc_after"]),
                    "dc_relative_change": _tensor_scalar(
                        correction_stats["dc_relative_change"]
                    ),
                    "final_state_update_norm": _tensor_scalar(
                        correction_stats["final_state_update_norm"]
                    ),
                    "correction_wall_time_ms": wall_time_ms,
                    "memory_allocated_before_mb": allocated_before_mb,
                    "memory_reserved_before_mb": reserved_before_mb,
                    "correction_peak_allocated_mb": peak_allocated_mb,
                    "correction_peak_reserved_mb": peak_reserved_mb,
                    "correction_extra_peak_allocated_mb": extra_allocated_mb,
                    "correction_extra_peak_reserved_mb": extra_reserved_mb,
                }
            )

            x = x_next

        should_save = (
            save_intermediate
            and intermediate_dir is not None
            and clean is not None
            and (i == 0 or outer_step % intermediate_every == 0 or i == num_steps - 1)
        )
        if should_save:
            make_figures = importlib.import_module("utils").makeFigures
            current_recon = x[:, :, pad : pad + w, pad : pad + w].detach().squeeze(1)
            metrics = make_figures(
                noisy2=x_init.squeeze(1),
                denoised2=current_recon,
                orig2=clean.squeeze(1),
                i=outer_step,
                out_dir=intermediate_dir,
                tag=f"{safe_tag}_step{outer_step:03d}",
                plot=True,
            )
            intermediate_rows.append(
                {
                    "step": outer_step,
                    "num_steps": num_steps,
                    "noisy_psnr": float(metrics[0]),
                    "recon_psnr": float(metrics[1]),
                    "noisy_ssim": float(metrics[2]),
                    "recon_ssim": float(metrics[3]),
                    "noisy_nrmse": float(metrics[4]),
                    "recon_nrmse": float(metrics[5]),
                }
            )

    if profile_memory and x.is_cuda:
        torch.cuda.synchronize(x.device)
        _preserve_current_cuda_peaks(x.device)

    if intermediate_rows and intermediate_dir is not None:
        csv_path = os.path.join(intermediate_dir, "intermediate_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(intermediate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(intermediate_rows)

    if (save_lgfc_diagnostics or profile_memory) and diagnostics_rows:
        diagnostic_path = _write_diagnostics(diagnostics_rows, save_dir, tag)
        print(f"[LGFC] Saved diagnostics to {diagnostic_path}")

    return (
        x[:, :, pad : pad + w, pad : pad + w].detach().squeeze(1),
        noisypsnr,
        denoisedpsnr,
        noisyssim,
        denoisedssim,
        noisynrmse,
        denoisednrmse,
    )
