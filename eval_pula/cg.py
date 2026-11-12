"""Matrix-free complex CG for a single Hermitian positive-definite system."""

import math

import torch


def _inner(x, y):
    return torch.vdot(x.reshape(-1), y.reshape(-1)).real


@torch.no_grad()
def solve_cg(normal, rhs, initial, max_iters, rtol=1e-6):
    x = initial
    residual = rhs - normal(x)
    direction = residual
    rhs_norm = torch.linalg.vector_norm(rhs).item()
    target_sq = (rtol * rhs_norm) ** 2
    rsq = _inner(residual, residual).item()
    iterations = 0
    while iterations < max_iters and rsq > target_sq:
        projected = normal(direction)
        curvature = _inner(direction, projected).item()
        if not math.isfinite(curvature) or curvature <= 0:
            raise FloatingPointError(f"pULA CG received invalid curvature {curvature}.")
        alpha = rsq / curvature
        x = x + alpha * direction
        residual = residual - alpha * projected
        next_rsq = _inner(residual, residual).item()
        direction = residual + (next_rsq / rsq) * direction
        rsq = next_rsq
        iterations += 1

    # Report the true residual, including roundoff from the CG recurrence.
    residual_norm = torch.linalg.vector_norm(rhs - normal(x)).item()
    if not math.isfinite(residual_norm):
        raise FloatingPointError("pULA CG returned a nonfinite residual.")
    relative = residual_norm / rhs_norm if rhs_norm else None
    return x, {"iterations": iterations, "residual_norm": residual_norm, "relative_residual": relative}
