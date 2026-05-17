"""
Batched complex-valued conjugate gradient.

This file is intentionally independent from the PaDIS sampler so that
preconditioned posterior solvers can reuse it without touching the original
PaDIS DPS baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

Tensor = torch.Tensor
LinearOperator = Callable[[Tensor], Tensor]


@dataclass
class CGResult:
    x: Tensor
    info: Dict[str, object]


def _batch_inner(x: Tensor, y: Tensor) -> Tensor:
    if x.shape != y.shape:
        raise ValueError(f"CG inner product shape mismatch: {x.shape} vs {y.shape}")
    b = x.shape[0]
    return torch.sum(torch.conj(x).reshape(b, -1) * y.reshape(b, -1), dim=1).real


def _expand(coeff: Tensor, ref: Tensor) -> Tensor:
    return coeff.reshape(coeff.shape[0], *([1] * (ref.ndim - 1)))


@torch.no_grad()
def conjugate_gradient(
    operator: LinearOperator,
    rhs: Tensor,
    x0: Optional[Tensor] = None,
    max_iter: int = 10,
    tol: float = 0.0,
    eps: float = 1e-12,
    collect_history: bool = False,
) -> CGResult:
    """
    Solve operator(x) = rhs for Hermitian positive-definite operators.

    Args:
        operator: function mapping rhs-shaped tensors to rhs-shaped tensors.
        rhs: batched complex tensor [B, ...].
        x0: optional warm start.
        max_iter: maximum CG iterations.
        tol: absolute residual-norm stopping threshold; <=0 disables early stop.
        eps: numerical stabilizer.
        collect_history: collect residual norms for debugging.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    r = rhs - operator(x)
    p = r.clone()
    rr = _batch_inner(r, r).clamp_min(eps)

    hist = []
    if collect_history:
        hist.append(torch.sqrt(rr).detach().cpu())

    converged = torch.zeros(rhs.shape[0], dtype=torch.bool, device=rhs.device)
    iters = 0

    for it in range(max_iter):
        Ap = operator(p)
        pAp = _batch_inner(p, Ap)

        if not bool(torch.all(torch.isfinite(pAp))):
            raise FloatingPointError("CG encountered non-finite pAp.")
        pAp = pAp.clamp_min(eps)

        alpha = rr / pAp
        x = x + _expand(alpha, x) * p
        r = r - _expand(alpha, r) * Ap

        rr_new = _batch_inner(r, r)
        if not bool(torch.all(torch.isfinite(rr_new))):
            raise FloatingPointError("CG residual became non-finite.")
        rr_new = rr_new.clamp_min(eps)

        iters = it + 1
        res_norm = torch.sqrt(rr_new)
        if collect_history:
            hist.append(res_norm.detach().cpu())

        if tol > 0:
            converged = converged | (res_norm <= tol)
            if bool(torch.all(converged)):
                rr = rr_new
                break

        beta = rr_new / rr
        p = r + _expand(beta, p) * p
        rr = rr_new

    info: Dict[str, object] = {
        "iters": iters,
        "final_residual_norm": torch.sqrt(rr).detach().cpu(),
        "converged": converged.detach().cpu(),
    }
    if collect_history:
        info["residual_history"] = hist

    return CGResult(x=x, info=info)
