"""
BART-source-aligned PaDIS-pULA prototype.

Why this version exists
-----------------------
The previous "paper formula" prototype implemented the displayed pULA equations
literally. The author-provided BART implementation uses equivalent conventions
inside its complex-valued Euler-Maruyama routine, but the executable source code
differs in two operationally important ways:

1) `get_init()` initializes pULA by running one preconditioned EM step with
   step=2 from x=0 and then multiplying by 0.5.
2) `eulermaruyama_precond()` uses sqrt(step) and sqrt(step * diag) noise
   coefficients in BART's complex-Gaussian convention, not an explicit sqrt(2).

This file mirrors the BART implementation structure while replacing BART's
CUNet score with a PaDIS patch score.

A PaDIS-specific compatibility option is also included:
- score_mode="padis_noisy": query the patch denoiser using
      x_query = x + sigma * eps
  and score = (D(x_query, sigma) - x_query) / sigma^2,
  which matches the currently working PaDIS-DPS inference contract.
- score_mode="direct": query the denoiser at the current pULA state x,
  which is closer to BART's score-network interface but was numerically unstable
  in the first transfer attempt.

The source implementation being mirrored is:
- BART `sample.c::get_init`
- BART `iter/italgos.c::eulermaruyama_precond`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
EVAL_DIR = PROJECT_ROOT / "eval"
for _p in (str(EVAL_DIR), str(PROJECT_ROOT), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from denoise_padding import denoisedFromPatches, getIndices
from cg_solver import conjugate_gradient

Tensor = torch.Tensor


@dataclass
class PULASamplerConfig:
    # Noise schedule: BART sample.c default is exponential.
    num_steps: int = 60
    sigma_min: float = 0.01
    sigma_max: float = 10.0
    schedule: str = "exponential"   # "exponential" or "karras"
    rho: float = 7.0                # only for Karras comparison

    # Author main Cartesian experiment: K=4, NCG=10, gamma=0.5.
    corrector_steps: int = 4
    cg_iters: int = 10
    cg_tol: float = 0.0
    gamma: float = 0.5

    # Initialization:
    # "bart_source": mirror sample.c::get_init() source implementation
    # "precond_mean": deterministic M A^H y
    # "adjoint": A^H y, useful only for diagnostics
    init_mode: str = "bart_source"

    # PaDIS prior query mode.
    # "padis_noisy" matches current PaDIS-DPS denoiser usage.
    # "direct" queries D(x, sigma) directly.
    score_mode: str = "padis_noisy"

    # Diagnostic multipliers.
    prior_scale: float = 1.0
    likelihood_scale: float = 1.0
    noise_scale: float = 1.0

    # PaDIS patch settings.
    pad: int = 64
    psize: int = 64

    # Logging.
    verbose: bool = True
    log_every: int = 1


def _build_sigma_schedule(
    net: Any,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    schedule: str,
    rho: float,
    device: torch.device,
) -> Tensor:
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be positive")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max must be >= sigma_min")

    schedule = schedule.lower()
    if num_steps == 1:
        sigmas = torch.tensor([sigma_max], dtype=torch.float64, device=device)
    elif schedule in {"exp", "exponential", "log"}:
        sigmas = torch.exp(
            torch.linspace(
                math.log(sigma_max),
                math.log(sigma_min),
                num_steps,
                dtype=torch.float64,
                device=device,
            )
        )
    elif schedule == "karras":
        idx = torch.arange(num_steps, dtype=torch.float64, device=device)
        sigmas = (
            sigma_max ** (1.0 / rho)
            + (idx / (num_steps - 1.0))
            * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")

    if hasattr(net, "round_sigma"):
        sigmas = net.round_sigma(sigmas)
    return sigmas


def _complex_randn_like(x: Tensor) -> Tensor:
    # torch.randn_like(complex) is CN(0, I): real/imag std ≈ 1/sqrt(2).
    return torch.randn_like(x)


def _to_two_channel_real(x: Tensor) -> Tensor:
    if not torch.is_complex(x):
        raise TypeError("_to_two_channel_real expects a complex tensor.")
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")
    return torch.view_as_real(x.squeeze(1)).permute(0, 3, 1, 2).contiguous()


def _two_channel_to_complex(x: Tensor) -> Tensor:
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"Expected [B,2,H,W], got {tuple(x.shape)}")
    return torch.complex(x[:, 0], x[:, 1]).unsqueeze(1)


def _crop_fov(x_pad: Tensor, pad: int, h: int, w: int) -> Tensor:
    if pad == 0:
        return x_pad
    return x_pad[:, :, pad:pad + h, pad:pad + w]


def _embed_fov(x_fov: Tensor, pad: int) -> Tensor:
    if pad == 0:
        return x_fov
    return F.pad(x_fov, (pad, pad, pad, pad), mode="constant", value=0)


@torch.no_grad()
def _normal_fov(inverseop: Any, x_fov: Tensor) -> Tensor:
    Ax = inverseop.forward(_to_two_channel_real(x_fov))
    return inverseop.adjoint(Ax)


@torch.no_grad()
def _likelihood_grad_pad(
    inverseop: Any,
    x_pad: Tensor,
    AHy_fov: Tensor,
    pad: int,
    h: int,
    w: int,
) -> Tensor:
    x_fov = _crop_fov(x_pad, pad, h, w)
    AHAx_fov = _normal_fov(inverseop, x_fov)
    return _embed_fov(AHy_fov - AHAx_fov, pad)


@torch.no_grad()
def _measurement_noise_adjoint_pad(
    inverseop: Any,
    measurement: Tensor,
    pad: int,
) -> Tensor:
    n1 = _complex_randn_like(measurement)
    return _embed_fov(inverseop.adjoint(n1), pad)


def _make_preconditioner_operator(
    inverseop: Any,
    diag: float,
    pad: int,
    h: int,
    w: int,
):
    def op(v_pad: Tensor) -> Tensor:
        v_fov = _crop_fov(v_pad, pad, h, w)
        AHA_v_fov = _normal_fov(inverseop, v_fov)
        return _embed_fov(AHA_v_fov, pad) + diag * v_pad
    return op


def _patch_score(
    net: Any,
    x_pad: Tensor,
    sigma: Tensor,
    latents_pos: Tensor,
    pad: int,
    psize: int,
    score_mode: str,
) -> Tuple[Tensor, Tensor]:
    """
    Return (score, denoiser_query).

    score_mode="padis_noisy":
        x_query = x + sigma eps
        score = (D(x_query, sigma) - x_query) / sigma^2

    score_mode="direct":
        x_query = x
        score = (D(x, sigma) - x) / sigma^2
    """
    score_mode = score_mode.lower()
    if score_mode == "padis_noisy":
        x_query = x_pad + sigma.to(x_pad.real.dtype) * _complex_randn_like(x_pad)
    elif score_mode == "direct":
        x_query = x_pad
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    resolution = x_pad.shape[-1]
    fov = resolution - 2 * pad
    patches = fov // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)
    indices = getIndices(spaced, patches, pad, psize)

    D_real = denoisedFromPatches(
        net,
        _to_two_channel_real(x_query),
        sigma,
        latents_pos,
        None,
        indices,
        t_goal=0,
        wrong=False,
    )
    D_cplx = _two_channel_to_complex(D_real)
    score = (D_cplx - x_query) / (sigma ** 2)
    return score, x_query


@torch.no_grad()
def _preconditioned_mean_initialization(
    inverseop: Any,
    AHy_fov: Tensor,
    sigma_max: float,
    pad: int,
    h: int,
    w: int,
    cg_iters: int,
    cg_tol: float,
) -> Tuple[Tensor, Dict[str, Any]]:
    diag = 1.0 / max(float(sigma_max) ** 2, 1e-12)
    op = _make_preconditioner_operator(inverseop, diag, pad, h, w)
    rhs = _embed_fov(AHy_fov, pad)
    out = conjugate_gradient(op, rhs, x0=torch.zeros_like(rhs), max_iter=cg_iters, tol=cg_tol)
    return out.x.detach(), {
        "init_mode": "precond_mean",
        "init_cg_iters": int(out.info["iters"]),
        "init_cg_final_residual_mean": float(out.info["final_residual_norm"].mean()),
    }


@torch.no_grad()
def _bart_source_initialization(
    inverseop: Any,
    measurement: Tensor,
    AHy_fov: Tensor,
    sigma_max: float,
    pad: int,
    h: int,
    w: int,
    cg_iters: int,
    cg_tol: float,
    noise_scale: float,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Mirror BART sample.c::get_init() for pULA:

        em_conf.maxiter = 1
        em_conf.step = 2
        em_conf.precond_diag = 1 / sigma^2
        x starts from zero
        run preconditioned EM with zero prior score
        multiply final samples by 0.5
    """
    diag = 1.0 / max(float(sigma_max) ** 2, 1e-12)
    step = 2.0
    op = _make_preconditioner_operator(inverseop, diag, pad, h, w)

    rhs_base = _embed_fov(AHy_fov, pad)
    rhs = step * rhs_base
    x0 = torch.zeros_like(rhs)

    if noise_scale > 0:
        AHn1 = _measurement_noise_adjoint_pad(inverseop, measurement, pad)
        n2 = _complex_randn_like(rhs)
        rhs = (
            rhs
            + noise_scale * math.sqrt(step) * AHn1
            + noise_scale * math.sqrt(step * diag) * n2
        )
        # eulermaruyama_precond warm-start line:
        # x += sqrt(step / precond_diag) * n2
        x0 = x0 + noise_scale * math.sqrt(step / diag) * n2

    out = conjugate_gradient(op, rhs, x0=x0, max_iter=cg_iters, tol=cg_tol)
    x = 0.5 * out.x.detach()

    return x, {
        "init_mode": "bart_source",
        "init_step": step,
        "init_diag": diag,
        "init_cg_iters": int(out.info["iters"]),
        "init_cg_final_residual_mean": float(out.info["final_residual_norm"].mean()),
    }


@torch.no_grad()
def padis_pula_reconstruct(
    net: Any,
    inverseop: Any,
    measurement: Tensor,
    latents_pos: Tensor,
    config: Optional[PULASamplerConfig] = None,
    init: Optional[Tensor] = None,
    clean: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    cfg = config or PULASamplerConfig()
    if cfg.corrector_steps < 1 or cfg.cg_iters < 1:
        raise ValueError("corrector_steps and cg_iters must be >= 1")
    if cfg.gamma <= 0:
        raise ValueError("gamma must be > 0")

    device = measurement.device
    net.eval()

    AHy_fov = inverseop.adjoint(measurement).detach()
    if AHy_fov.ndim != 4 or AHy_fov.shape[1] != 1:
        raise ValueError(f"Expected adjoint result [B,1,H,W], got {tuple(AHy_fov.shape)}")

    batch_size, _, h, w = AHy_fov.shape
    sigmas = _build_sigma_schedule(
        net, cfg.num_steps, cfg.sigma_min, cfg.sigma_max, cfg.schedule, cfg.rho, device
    )
    sigma_start = float(sigmas[0].detach().cpu())

    if init is not None:
        x_pad = _embed_fov(init.detach().clone(), cfg.pad)
        init_info = {"init_mode": "provided"}
    else:
        mode = cfg.init_mode.lower()
        if mode in {"bart_source", "bart", "source"}:
            x_pad, init_info = _bart_source_initialization(
                inverseop, measurement, AHy_fov, sigma_start, cfg.pad, h, w,
                cfg.cg_iters, cfg.cg_tol, cfg.noise_scale,
            )
        elif mode in {"precond_mean", "mean"}:
            x_pad, init_info = _preconditioned_mean_initialization(
                inverseop, AHy_fov, sigma_start, cfg.pad, h, w, cfg.cg_iters, cfg.cg_tol
            )
        elif mode == "adjoint":
            x_pad = _embed_fov(AHy_fov.clone(), cfg.pad)
            init_info = {"init_mode": "adjoint"}
        else:
            raise ValueError(f"Unsupported init_mode: {cfg.init_mode}")

    expected_hw = (h + 2 * cfg.pad, w + 2 * cfg.pad)
    if tuple(latents_pos.shape[-2:]) != expected_hw:
        raise ValueError(
            f"latents_pos spatial size {tuple(latents_pos.shape[-2:])} does not match {expected_hw}"
        )

    diagnostics: Dict[str, Any] = {
        "config": asdict(cfg),
        "initialization": init_info,
        "sigma_schedule": [float(s.detach().cpu()) for s in sigmas],
        "cg_final_residual_mean": [],
        "cg_iters": [],
        "likelihood_grad_norm": [],
        "prior_score_norm": [],
        "state_norm": [],
        "query_state_norm": [],
    }

    iterator = enumerate(sigmas)
    if cfg.verbose:
        iterator = tqdm.tqdm(iterator, total=len(sigmas), desc="PaDIS-pULA/BART-source")

    for step_idx, sigma in iterator:
        sigma_f = float(sigma.detach().cpu())
        var = max(sigma_f ** 2, 1e-12)
        diag = 1.0 / var
        gamma = float(cfg.gamma)
        precond_op = _make_preconditioner_operator(inverseop, diag, cfg.pad, h, w)

        for _corr_idx in range(cfg.corrector_steps):
            score_pad, x_query = _patch_score(
                net=net,
                x_pad=x_pad,
                sigma=sigma.float(),
                latents_pos=latents_pos,
                pad=cfg.pad,
                psize=cfg.psize,
                score_mode=cfg.score_mode,
            )
            likelihood_grad_pad = _likelihood_grad_pad(
                inverseop, x_pad, AHy_fov, cfg.pad, h, w
            )

            # BART source structure:
            # o = (AHA + diag I)x + step * likelihood + t_prior
            # t_prior is the proximal displacement; approximated here as gamma * score.
            t_prior = gamma * cfg.prior_scale * score_pad
            rhs = (
                precond_op(x_pad)
                + gamma * cfg.likelihood_scale * likelihood_grad_pad
                + t_prior
            )

            x_warm = x_pad + (1.0 / diag) * t_prior

            if cfg.noise_scale > 0:
                AHn1 = _measurement_noise_adjoint_pad(inverseop, measurement, cfg.pad)
                n2 = _complex_randn_like(x_pad)

                # Mirror BART italgos.c::eulermaruyama_precond:
                # o += sqrt(step) * A^H n1
                # o += sqrt(step * diag) * n2
                # x += sqrt(step / diag) * n2
                rhs = (
                    rhs
                    + cfg.noise_scale * math.sqrt(gamma) * AHn1
                    + cfg.noise_scale * math.sqrt(gamma * diag) * n2
                )
                x_warm = x_warm + cfg.noise_scale * math.sqrt(gamma / diag) * n2

            out = conjugate_gradient(
                precond_op, rhs, x0=x_warm, max_iter=cfg.cg_iters, tol=cfg.cg_tol
            )
            x_pad = out.x.detach()

            diagnostics["cg_final_residual_mean"].append(
                float(out.info["final_residual_norm"].mean())
            )
            diagnostics["cg_iters"].append(int(out.info["iters"]))
            diagnostics["likelihood_grad_norm"].append(
                float(torch.linalg.vector_norm(likelihood_grad_pad.reshape(batch_size, -1), dim=1).mean().detach().cpu())
            )
            diagnostics["prior_score_norm"].append(
                float(torch.linalg.vector_norm(score_pad.reshape(batch_size, -1), dim=1).mean().detach().cpu())
            )
            diagnostics["state_norm"].append(
                float(torch.linalg.vector_norm(x_pad.reshape(batch_size, -1), dim=1).mean().detach().cpu())
            )
            diagnostics["query_state_norm"].append(
                float(torch.linalg.vector_norm(x_query.reshape(batch_size, -1), dim=1).mean().detach().cpu())
            )

        if cfg.verbose and ((step_idx + 1) % max(cfg.log_every, 1) == 0):
            msg = (
                f"[pULA/BART-source] step={step_idx + 1:03d}/{cfg.num_steps:03d} "
                f"sigma={sigma_f:.4g} "
                f"state_norm={diagnostics['state_norm'][-1]:.4e} "
                f"score_norm={diagnostics['prior_score_norm'][-1]:.4e} "
                f"cg_res={diagnostics['cg_final_residual_mean'][-1]:.4e}"
            )
            try:
                iterator.write(msg)
            except Exception:
                print(msg)

    recon_fov = _crop_fov(x_pad, cfg.pad, h, w).detach().squeeze(1)
    return recon_fov, diagnostics
