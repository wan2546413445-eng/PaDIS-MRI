"""
pULA sampler wired to a clean PaDIS score oracle.

This file intentionally does NOT modify `eval/recon.py::dps2()`.
The DPS baseline should remain intact.  pULA is a separate posterior sampler
with exact likelihood gradients and preconditioned Langevin/ULA updates.

The key engineering boundary is:
    prior_score = PaDISPatchScoreOracle(x_t, sigma_t).score
    posterior dynamics = pULA-specific update
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
EVAL_DIR = PROJECT_ROOT / "eval"
for _p in (str(EVAL_DIR), str(PROJECT_ROOT), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg_solver import conjugate_gradient
from padis_score_oracle import PaDISPatchScoreOracle, PaDISScoreOracleConfig

Tensor = torch.Tensor


@dataclass
class PULASamplerConfig:
    # BART/pULA-style exponential sigma ladder.
    num_steps: int = 60
    sigma_min: float = 0.01
    sigma_max: float = 10.0

    # Author-style nominal settings; reduce during debugging if needed.
    corrector_steps: int = 4
    cg_iters: int = 10
    cg_tol: float = 0.0
    gamma: float = 0.5

    # Keep pULA sampler mathematically clean by default.
    score_mode: str = "direct"         # "direct" or "legacy_dps2" for ablation only
    init_mode: str = "bart_source"     # "bart_source", "precond_mean", "adjoint"

    # Diagnostics / ablations.
    prior_scale: float = 1.0
    likelihood_scale: float = 1.0

    # Separate stochasticity in BART-style posterior initialization
    # from stochasticity injected at each pULA update step.
    init_noise_scale: float = 0.0
    step_noise_scale: float = 0.0

    # PaDIS patch geometry.
    image_size: int = 384
    pad: int = 64
    psize: int = 64
    freeze_patch_indices: bool = False

    # Numerical safety.
    abort_on_nonfinite: bool = True
    abort_state_norm: float = 1e12

    # Logging.
    verbose: bool = True
    log_every: int = 1


def _sigma_schedule(cfg: PULASamplerConfig, device: torch.device) -> Tensor:
    if cfg.num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if cfg.sigma_min <= 0 or cfg.sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be positive")
    if cfg.sigma_max < cfg.sigma_min:
        raise ValueError("sigma_max must be >= sigma_min")

    if cfg.num_steps == 1:
        return torch.tensor([cfg.sigma_max], dtype=torch.float64, device=device)

    return torch.exp(
        torch.linspace(
            math.log(cfg.sigma_max),
            math.log(cfg.sigma_min),
            cfg.num_steps,
            dtype=torch.float64,
            device=device,
        )
    )


def _embed_fov(x_fov: Tensor, pad: int) -> Tensor:
    return x_fov if pad == 0 else F.pad(x_fov, (pad, pad, pad, pad), mode="constant", value=0)


def _crop_fov(x_pad: Tensor, pad: int, h: int, w: int) -> Tensor:
    return x_pad if pad == 0 else x_pad[:, :, pad:pad+h, pad:pad+w]


def _to_two_channel_real(x: Tensor) -> Tensor:
    if not torch.is_complex(x):
        raise TypeError("Expected complex image tensor.")
    return torch.view_as_real(x.squeeze(1)).permute(0, 3, 1, 2).contiguous()


def _complex_randn_like(x: Tensor) -> Tensor:
    return torch.randn_like(x)


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
    return _embed_fov(AHy_fov - _normal_fov(inverseop, x_fov), pad)


@torch.no_grad()
def _measurement_noise_adjoint_pad(
    inverseop: Any,
    measurement: Tensor,
    pad: int,
) -> Tensor:
    return _embed_fov(inverseop.adjoint(_complex_randn_like(measurement)), pad)


def _make_preconditioner_op(
    inverseop: Any,
    diag: float,
    pad: int,
    h: int,
    w: int,
):
    def op(v_pad: Tensor) -> Tensor:
        v_fov = _crop_fov(v_pad, pad, h, w)
        return _embed_fov(_normal_fov(inverseop, v_fov), pad) + diag * v_pad
    return op


@torch.no_grad()
def _precond_mean_init(
    inverseop: Any,
    AHy_fov: Tensor,
    cfg: PULASamplerConfig,
    h: int,
    w: int,
) -> Tuple[Tensor, Dict[str, object]]:
    diag = 1.0 / (cfg.sigma_max ** 2)
    op = _make_preconditioner_op(inverseop, diag, cfg.pad, h, w)
    rhs = _embed_fov(AHy_fov, cfg.pad)
    out = conjugate_gradient(op, rhs, max_iter=cfg.cg_iters, tol=cfg.cg_tol)
    return out.x.detach(), {
        "mode": "precond_mean",
        "cg_iters": int(out.info["iters"]),
        "cg_residual_mean": float(out.info["final_residual_norm"].mean()),
    }


@torch.no_grad()
def _bart_source_init(
    inverseop: Any,
    measurement: Tensor,
    AHy_fov: Tensor,
    cfg: PULASamplerConfig,
    h: int,
    w: int,
) -> Tuple[Tensor, Dict[str, object]]:
    """
    Mirror BART `sample.c::get_init()` operationally:
      - start from zero
      - one preconditioned EM step with step=2
      - zero prior score
      - scale final state by 0.5
    """
    sigma = cfg.sigma_max
    diag = 1.0 / (sigma ** 2)
    step = 2.0
    op = _make_preconditioner_op(inverseop, diag, cfg.pad, h, w)

    rhs = step * _embed_fov(AHy_fov, cfg.pad)
    x0 = torch.zeros_like(rhs)

    if cfg.init_noise_scale > 0:
        AHn1 = _measurement_noise_adjoint_pad(inverseop, measurement, cfg.pad)
        n2 = _complex_randn_like(rhs)
        rhs = rhs + cfg.init_noise_scale * math.sqrt(step) * AHn1 + cfg.init_noise_scale * math.sqrt(step * diag) * n2
        x0 = x0 + cfg.init_noise_scale * math.sqrt(step / diag) * n2

    out = conjugate_gradient(op, rhs, x0=x0, max_iter=cfg.cg_iters, tol=cfg.cg_tol)
    return 0.5 * out.x.detach(), {
        "mode": "bart_source",
        "cg_iters": int(out.info["iters"]),
        "cg_residual_mean": float(out.info["final_residual_norm"].mean()),
    }


@torch.no_grad()
def padis_pula_reconstruct(
    *,
    net: Any,
    inverseop: Any,
    measurement: Tensor,
    latents_pos: Tensor,
    config: Optional[PULASamplerConfig] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    cfg = config or PULASamplerConfig()
    if cfg.corrector_steps < 1 or cfg.cg_iters < 1:
        raise ValueError("corrector_steps and cg_iters must be >= 1")

    net.eval()
    device = measurement.device
    AHy_fov = inverseop.adjoint(measurement).detach()
    if AHy_fov.ndim != 4 or AHy_fov.shape[1] != 1:
        raise ValueError(f"Expected AHy shape [B,1,H,W], got {tuple(AHy_fov.shape)}")

    b, _, h, w = AHy_fov.shape
    score_oracle = PaDISPatchScoreOracle(
        net,
        PaDISScoreOracleConfig(
            image_size=cfg.image_size,
            pad=cfg.pad,
            psize=cfg.psize,
            mode=cfg.score_mode,
            freeze_patch_indices=cfg.freeze_patch_indices,
        ),
    )

    init_mode = cfg.init_mode.lower()
    if init_mode == "bart_source":
        x_pad, init_info = _bart_source_init(inverseop, measurement, AHy_fov, cfg, h, w)
    elif init_mode == "precond_mean":
        x_pad, init_info = _precond_mean_init(inverseop, AHy_fov, cfg, h, w)
    elif init_mode == "adjoint":
        x_pad = _embed_fov(AHy_fov.clone(), cfg.pad)
        init_info = {"mode": "adjoint"}
    else:
        raise ValueError(f"Unsupported init_mode: {cfg.init_mode}")

    sigmas = _sigma_schedule(cfg, device)
    diagnostics: Dict[str, object] = {
        "config": asdict(cfg),
        "initialization": init_info,
        "sigma_schedule": [float(s.cpu()) for s in sigmas],
        "state_norm": [],
        "score_norm": [],
        "likelihood_norm": [],
        "cg_residual_mean": [],
        "precond_x_norm": [],
        "prior_disp_norm": [],
        "likelihood_disp_norm": [],
        "noise_ahn1_norm": [],
        "noise_n2_rhs_norm": [],
        "rhs_norm": [],
        "x_warm_norm": [],
    }

    iterator = enumerate(sigmas)
    if cfg.verbose:
        iterator = tqdm.tqdm(iterator, total=len(sigmas), desc="PaDIS-pULA-clean")

    for step_idx, sigma in iterator:
        sigma_f = float(sigma.detach().cpu())
        diag = 1.0 / max(sigma_f ** 2, 1e-12)
        precond_op = _make_preconditioner_op(inverseop, diag, cfg.pad, h, w)

        for _ in range(cfg.corrector_steps):
            oracle = score_oracle(x_pad, sigma.float(), latents_pos, mode=cfg.score_mode)
            prior_score = oracle.score
            likelihood_grad = _likelihood_grad_pad(inverseop, x_pad, AHy_fov, cfg.pad, h, w)

            # Structural pULA update.  The important thing is that the PaDIS prior
            # enters only through `prior_score`; no DPS denoiser-Jacobian term is used.
            prior_disp = cfg.gamma * cfg.prior_scale * prior_score
            precond_x = precond_op(x_pad)
            likelihood_disp = cfg.gamma * cfg.likelihood_scale * likelihood_grad

            rhs = precond_x + likelihood_disp + prior_disp
            x_warm = x_pad + prior_disp / diag

            noise_ahn1 = torch.zeros_like(x_pad)
            noise_n2_rhs = torch.zeros_like(x_pad)

            if cfg.step_noise_scale > 0:
                AHn1 = _measurement_noise_adjoint_pad(inverseop, measurement, cfg.pad)
                n2 = _complex_randn_like(x_pad)

                noise_ahn1 = cfg.step_noise_scale * math.sqrt(cfg.gamma) * AHn1
                noise_n2_rhs = cfg.step_noise_scale * math.sqrt(cfg.gamma * diag) * n2

                rhs = rhs + noise_ahn1 + noise_n2_rhs
                x_warm = x_warm + cfg.step_noise_scale * math.sqrt(cfg.gamma / diag) * n2

            out = conjugate_gradient(
                precond_op,
                rhs,
                x0=x_warm,
                max_iter=cfg.cg_iters,
                tol=cfg.cg_tol,
            )
            x_pad = out.x.detach()

            state_norm = torch.linalg.vector_norm(x_pad.reshape(b, -1), dim=1).mean()
            score_norm = torch.linalg.vector_norm(prior_score.reshape(b, -1), dim=1).mean()
            like_norm = torch.linalg.vector_norm(likelihood_grad.reshape(b, -1), dim=1).mean()
            cg_res = out.info["final_residual_norm"].mean()

            if cfg.abort_on_nonfinite:
                vals = [state_norm, score_norm, like_norm, cg_res]
                if not all(bool(torch.isfinite(v)) for v in vals):
                    raise FloatingPointError(
                        f"Non-finite value at step={step_idx+1}, sigma={sigma_f:.6g}"
                    )
                if float(state_norm.detach().cpu()) > cfg.abort_state_norm:
                    raise FloatingPointError(
                        f"State norm exceeded abort threshold at step={step_idx+1}, "
                        f"sigma={sigma_f:.6g}, norm={float(state_norm):.4e}"
                    )

            precond_x_norm = torch.linalg.vector_norm(precond_x.reshape(b, -1), dim=1).mean()
            prior_disp_norm = torch.linalg.vector_norm(prior_disp.reshape(b, -1), dim=1).mean()
            likelihood_disp_norm = torch.linalg.vector_norm(likelihood_disp.reshape(b, -1), dim=1).mean()
            noise_ahn1_norm = torch.linalg.vector_norm(noise_ahn1.reshape(b, -1), dim=1).mean()
            noise_n2_rhs_norm = torch.linalg.vector_norm(noise_n2_rhs.reshape(b, -1), dim=1).mean()
            rhs_norm = torch.linalg.vector_norm(rhs.reshape(b, -1), dim=1).mean()
            x_warm_norm = torch.linalg.vector_norm(x_warm.reshape(b, -1), dim=1).mean()

            diagnostics["state_norm"].append(float(state_norm.detach().cpu()))
            diagnostics["score_norm"].append(float(score_norm.detach().cpu()))
            diagnostics["likelihood_norm"].append(float(like_norm.detach().cpu()))
            diagnostics["cg_residual_mean"].append(float(cg_res.detach().cpu()))
            diagnostics["precond_x_norm"].append(float(precond_x_norm.detach().cpu()))
            diagnostics["prior_disp_norm"].append(float(prior_disp_norm.detach().cpu()))
            diagnostics["likelihood_disp_norm"].append(float(likelihood_disp_norm.detach().cpu()))
            diagnostics["noise_ahn1_norm"].append(float(noise_ahn1_norm.detach().cpu()))
            diagnostics["noise_n2_rhs_norm"].append(float(noise_n2_rhs_norm.detach().cpu()))
            diagnostics["rhs_norm"].append(float(rhs_norm.detach().cpu()))
            diagnostics["x_warm_norm"].append(float(x_warm_norm.detach().cpu()))

        if cfg.verbose and ((step_idx + 1) % max(cfg.log_every, 1) == 0):
            msg = (
                f"[pULA-clean] step={step_idx+1:03d}/{cfg.num_steps:03d} "
                f"sigma={sigma_f:.4g} "
                f"state={diagnostics['state_norm'][-1]:.4e} "
                f"score={diagnostics['score_norm'][-1]:.4e} "
                f"rhs={diagnostics['rhs_norm'][-1]:.4e} "
                f"n2rhs={diagnostics['noise_n2_rhs_norm'][-1]:.4e} "
                f"cg={diagnostics['cg_residual_mean'][-1]:.4e}"
            )
            try:
                iterator.write(msg)
            except Exception:
                print(msg)

    recon = _crop_fov(x_pad, cfg.pad, h, w).detach().squeeze(1)
    return recon, diagnostics
