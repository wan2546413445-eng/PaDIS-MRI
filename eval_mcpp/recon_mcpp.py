"""PaDIS reconstruction with measurement-calibrated shared patch-prior state."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm

try:
    from denoise_padding import denoisedFromPatches, getIndices
except ImportError:  # Standalone smoke test injects both functions.
    denoisedFromPatches = None
    getIndices = None

try:
    from .prior_calibration import MCPPHeadAdapter
except ImportError:
    from prior_calibration import MCPPHeadAdapter


def _center(tensor: torch.Tensor, pad: int, width: int) -> torch.Tensor:
    return tensor if pad == 0 else tensor[..., pad:pad + width, pad:pad + width]


def _ve_base_schedule(net, num_sigmas, sigma_min, sigma_max, rho, device):
    """Match ``eval/recon.py::dps2`` VE schedule."""
    indices = torch.arange(num_sigmas, dtype=torch.float64, device=device)
    schedule = (
        sigma_max ** (1.0 / rho)
        + (indices / (num_sigmas - 1.0))
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    schedule = net.round_sigma(schedule)
    return torch.cat(
        [schedule, torch.zeros(1, dtype=torch.float64, device=device)], dim=0
    )


@torch.no_grad()
def _prior_dc_alignment_tensor(
    score: torch.Tensor,
    grad_x: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    numerator = torch.real(torch.sum(torch.conj(score) * (-grad_x)))
    denominator = torch.linalg.vector_norm(score) * torch.linalg.vector_norm(grad_x) + eps
    return (numerator / denominator).detach()


def dps2_mcpp(
    net: torch.nn.Module,
    latents: torch.Tensor,
    latents_pos: torch.Tensor,
    inverseop,
    measurement: torch.Tensor,
    num_steps: int = 78,
    inner_loops: int = 10,
    sigma_min: float = 0.003,
    sigma_max: float = 10.0,
    rho: float = 7.0,
    zeta: float = 3.0,
    pad: int = 64,
    psize: int = 64,
    mcpp_lr: float = 1e-4,
    randn_like: Callable = torch.randn_like,
    verbose: bool = False,
    device: str = "cuda",
    adapter: Optional[MCPPHeadAdapter] = None,
    collect_diagnostics: bool = False,
    _denoise_fn: Optional[Callable] = None,
    _indices_fn: Optional[Callable] = None,
) -> Tuple[torch.Tensor, List[Dict[str, float]], Dict[str, int]]:
    """Run baseline PaDIS state updates while continuously adapting the shared output head."""
    if measurement is None or measurement.shape[0] != 1:
        raise ValueError("MCPP Stage-1 requires one measured subject (batch size 1)")
    if mcpp_lr < 0:
        raise ValueError("mcpp_lr must be non-negative")

    denoise_fn = denoisedFromPatches if _denoise_fn is None else _denoise_fn
    indices_fn = getIndices if _indices_fn is None else _indices_fn
    if denoise_fn is None or indices_fn is None:
        raise ImportError("eval/denoise_padding.py is unavailable; provide smoke-test injections")

    adapter = adapter or MCPPHeadAdapter(net)
    net.eval()

    # 与 baseline 一致，以 reconstruction latent 的原始 FOV 宽度定义 patch grid。
    width = int(latents.shape[-1])
    if measurement.shape[-2:] != (width, width):
        raise ValueError(
            f"measurement spatial shape {tuple(measurement.shape[-2:])} "
            f"does not match latent FOV {(width, width)}"
        )

    patches = width // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)
    x_init = inverseop.adjoint(measurement).detach()
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)
    t_steps = _ve_base_schedule(
        net, num_steps, sigma_min, sigma_max, rho, torch.device(device)
    )

    eps = 1e-12
    measurement_energy = torch.sum(torch.abs(measurement) ** 2).detach()
    global_update = 0
    denoiser_calls = 0

    # Diagnostics are optional so runtime benchmarking is not polluted by 780 GPU->CPU syncs.
    diagnostic_meta = []
    diagnostic_values = []

    for outer_step, (t_cur, _t_next) in tqdm.tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=len(t_steps) - 1
    ):
        t_cur = t_cur.float()
        alpha = 0.5 * t_cur ** 2

        for inner_step in range(inner_loops):
            # 保留 baseline 的随机 partition 与噪声调用；MCPP 不增加额外随机分支。
            indices = indices_fn(spaced, patches, pad, psize)
            x = x.detach().requires_grad_(True)
            x_noisy = x + t_cur * randn_like(x)
            x_real_noisy = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)

            # 一次 patch forward 同时服务于 posterior x gradient 和 prior phi gradient。
            D_real = denoise_fn(
                net, x_real_noisy, t_cur, latents_pos, None, indices,
                t_goal=0, wrong=False,
            )
            denoiser_calls += 1

            D_cplx = torch.complex(D_real[:, 0], D_real[:, 1]).unsqueeze(1)
            score = (D_cplx - x_noisy) / (t_cur ** 2)
            cropped_x0hat = _center(D_real, pad, width)

            Ax = inverseop.forward(cropped_x0hat)
            residual = measurement - Ax
            residual_flat = residual.reshape(1, -1)
            sse_ind = torch.linalg.vector_norm(residual_flat, dim=-1) ** 2
            sse = torch.sum(sse_ind)

            grads = torch.autograd.grad(
                outputs=sse,
                inputs=[x] + adapter.parameters,
                allow_unused=False,
                create_graph=False,
                retain_graph=False,
            )
            grad_x = grads[0].detach()
            normalized_phi_grads = [
                gradient.detach() / (measurement_energy + eps)
                for gradient in grads[1:]
            ]

            # autograd.grad 已完成：切断旧图后再更新 x/phi，避免旧参数图跨 inner loop 保留。
            score_for_update = score.detach()
            sse_ind_for_update = sse_ind.detach()

            if collect_diagnostics:
                relative_sse_t = (sse.detach() / (measurement_energy + eps))
                phi_grad_norm_t = adapter.gradient_norm_tensor(normalized_phi_grads)
                alignment_t = _prior_dc_alignment_tensor(
                    _center(score_for_update, pad, width),
                    _center(grad_x, pad, width),
                    eps,
                )

            # 完全保留 baseline posterior state 更新。
            x = x - (
                zeta / torch.sqrt(sse_ind_for_update).reshape(-1, 1, 1, 1)
            ) * grad_x
            if outer_step < num_steps - 1:
                x = x + (alpha / 2) * score_for_update + torch.sqrt(alpha) * randn_like(x)
            else:
                x = x + (alpha / 2) * score_for_update

            # 新 phi 只影响下一 inner loop，不额外 re-forward 当前 step。
            adapter.step(normalized_phi_grads, mcpp_lr)

            global_update += 1
            if collect_diagnostics:
                diagnostic_meta.append((global_update, outer_step + 1, inner_step + 1))
                diagnostic_values.append(torch.stack([
                    t_cur.detach(),
                    relative_sse_t,
                    phi_grad_norm_t,
                    adapter.relative_delta_tensor(eps),
                    alignment_t,
                ]))

            if verbose:
                rel = float((sse.detach() / (measurement_energy + eps)).cpu())
                delta = float(adapter.relative_delta_tensor(eps).cpu())
                print(
                    f"step {outer_step + 1}/{num_steps} "
                    f"inner {inner_step + 1}/{inner_loops} "
                    f"rel_sse={rel:.6e} phi_delta={delta:.6e}"
                )

    diagnostics: List[Dict[str, float]] = []
    if diagnostic_values:
        # 单次传回 CPU，避免每个 inner loop 都同步 GPU。
        values = torch.stack(diagnostic_values, dim=0).detach().cpu().numpy()
        for (update_idx, outer_idx, inner_idx), row in zip(diagnostic_meta, values):
            diagnostics.append({
                "global_update": int(update_idx),
                "outer_step": int(outer_idx),
                "inner_step": int(inner_idx),
                "sigma": float(row[0]),
                "relative_measurement_sse": float(row[1]),
                "phi_grad_norm": float(row[2]),
                "phi_relative_delta": float(row[3]),
                "prior_dc_alignment": float(row[4]),
            })

    recon = _center(x, pad, width).detach().squeeze(1)
    metadata = {
        "updates": global_update,
        "denoiser_calls": denoiser_calls,
    }
    return recon, diagnostics, metadata
