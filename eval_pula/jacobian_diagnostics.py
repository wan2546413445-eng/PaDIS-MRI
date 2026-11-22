"""Read-only diagnostics of the existing PaDIS-DPS denoiser-pullback gradient.

No model calls, new random draws, second backward, or state modifications.
Complex inner products use the real Euclidean convention (Re sum conj(a)*b).
"""

import csv
import json
import math
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F


def partition_masks(height, width, pad, image_size, psize, shift, band=2, device="cpu"):
    """Return padded-exterior, actual-shift seam band, and FOV interior masks.

    PatchScore uses getIndices(arange(image_size//psize+1)*psize, ..., pad=0)
    and adds (row_shift, col_shift), so seams have coordinates shift + n*psize.
    The mask includes both sides of every seam and excludes exterior padding.
    """
    row_shift, col_shift = shift
    if not (0 <= row_shift < psize and 0 <= col_shift < psize):
        raise ValueError("Partition shifts must lie in [0, psize).")
    if not (0 < band < psize // 2):
        raise ValueError("The seam band must be narrower than half a patch.")
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    in_rows = (rows >= pad) & (rows < pad + image_size)
    in_cols = (cols >= pad) & (cols < pad + image_size)
    fov = in_rows[:, None] & in_cols[None, :]
    r = (rows - row_shift).remainder(psize)
    c = (cols - col_shift).remainder(psize)
    row_band = (r < band) | (r >= psize - band)
    col_band = (c < band) | (c >= psize - band)
    seam = fov & (row_band[:, None] | col_band[None, :])
    return {"padding": ~fov, "boundary": seam, "interior": fov & ~seam, "fov": fov}


def finite_number(value):
    return value if (value is not None and math.isfinite(value)) else None


def real_dot(a, b):
    return (a.conj() * b).real.sum()


def direction_cosine(a, b, eps=1e-12):
    an = torch.linalg.vector_norm(a)
    bn = torch.linalg.vector_norm(b)
    if not torch.isfinite(an) or not torch.isfinite(bn) or an <= eps or bn <= eps:
        return None
    return finite_number((real_dot(a, b) / (an * bn + eps)).item())


def _masked_stats(g, v, mask):
    """Spatial region statistics, with area-normalized RMS magnitudes."""
    g_area = g[..., mask]
    v_area = v[..., mask]
    n = int(mask.sum().item())
    return {
        "pixels": n,
        "g_rms": finite_number(g_area.abs().square().mean().sqrt().item()) if n else None,
        "v_rms": finite_number(v_area.abs().square().mean().sqrt().item()) if n else None,
        "cosine": direction_cosine(g_area, v_area) if n else None,
    }


def observe(current, likelihood_grad, residual, operator, *, sigma, outer, inner,
            shift, pad, image_size, psize, network_forwards,
            network_backwards, boundary_band=2):
    """Measure a single already-computed DPS gradient, without changing it.

    residual is y-A C D(u), so -A^H residual = A^H(A C D(u)-y).
    The original gradient w.r.t. current is 2 J_D(u)^H C^* A^H(...)
    because u=current+fixed_noise and du/dcurrent=I.
    """
    start = perf_counter()
    with torch.no_grad():
        # Detach before calling the *same raw operator* used in DPS.
        v_fov = -operator.adjoint(residual.detach())
        if v_fov.shape != current.shape[:-2] + (image_size, image_size):
            raise ValueError(f"MRI adjoint shape {v_fov.shape} is incompatible with DPS state {current.shape}.")
        v = F.pad(v_fov, (pad, pad, pad, pad))
        g = likelihood_grad.detach() * 0.5
        if g.shape != v.shape:
            raise ValueError("Padded Jacobian gradient and physical gradient have different shapes.")
        if current.shape[-2:] != (image_size + 2 * pad,) * 2:
            raise ValueError("Unexpected padded DPS state size.")
        masks = partition_masks(*current.shape[-2:], pad, image_size, psize,
                                shift, band=boundary_band, device=current.device)
        full = _masked_stats(g, v, torch.ones_like(masks["fov"]))
        fov = _masked_stats(g, v, masks["fov"])
        padding = _masked_stats(g, v, masks["padding"])
        boundary = _masked_stats(g, v, masks["boundary"])
        interior = _masked_stats(g, v, masks["interior"])
        total_g_energy = g.abs().square().sum().item()
        padding_g_energy = g[..., masks["padding"]].abs().square().sum().item()
        residual_norm = torch.linalg.vector_norm(residual.detach()).item()
        finite = (bool(torch.isfinite(g).all().item()) and
                  bool(torch.isfinite(v).all().item()) and
                  bool(torch.isfinite(residual.detach()).all().item()))
        zero_gradient = total_g_energy == 0.0
        padding_fraction = finite_number(padding_g_energy / total_g_energy) if total_g_energy else None
        ratio = finite_number((torch.linalg.vector_norm(g) /
                               (torch.linalg.vector_norm(v) + 1e-12)).item())
        status = "nonfinite" if not finite else ("zero_gradient" if zero_gradient else
                 ("extreme_ratio" if ratio > 1e4 or ratio < 1e-6 else "ok"))
        record = {
            "outer_level": int(outer), "inner_step": int(inner),
            "sigma": float(sigma), "shift_row": int(shift[0]), "shift_col": int(shift[1]),
            "boundary_band_px": int(boundary_band),
            "jacobian_cosine": full["cosine"], "jacobian_norm_ratio": ratio,
            "fov_cosine": fov["cosine"],
            "padding_g_energy_fraction": padding_fraction,
            "residual_l2": finite_number(residual_norm),
            "fov": fov, "padding": padding, "boundary": boundary, "interior": interior,
            "boundary_pixel_fraction_of_fov": boundary["pixels"] / fov["pixels"],
            "finite": finite, "zero_gradient": zero_gradient, "status": status,
            "network_forwards_so_far": int(network_forwards),
            "network_backwards_so_far": int(network_backwards),
            "extra_network_forwards": 0, "extra_network_backwards": 0,
            "extra_mri_forward_calls": 0, "extra_mri_adjoint_calls": 1,
        }
        record["diagnostic_wall_seconds"] = perf_counter() - start
        return record


FIELDS = [
    "outer_level", "inner_step", "sigma", "shift_row", "shift_col", "boundary_band_px",
    "jacobian_cosine", "jacobian_norm_ratio", "fov_cosine", "padding_g_energy_fraction",
    "residual_l2", "boundary_pixel_fraction_of_fov",
    "fov_g_rms", "fov_v_rms", "boundary_g_rms", "boundary_v_rms", "boundary_cosine",
    "interior_g_rms", "interior_v_rms", "interior_cosine", "padding_g_rms",
    "network_forwards_so_far", "network_backwards_so_far", "extra_network_forwards",
    "extra_network_backwards", "extra_mri_forward_calls", "extra_mri_adjoint_calls",
    "diagnostic_wall_seconds", "finite", "zero_gradient", "status",
]


def flatten(record):
    row = {k: record.get(k) for k in FIELDS}
    for region in ("fov", "boundary", "interior", "padding"):
        for key in ("g_rms", "v_rms", "cosine"):
            name = f"{region}_{key}"
            if name in row:
                row[name] = record[region][key]
    return row


def write_outputs(folder, rows, metadata):
    """Produce stable machine-readable artifacts. Called only after reconstruction."""
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    obj = {"schema_version": 1, "metadata": metadata, "records": rows,
           "units": {"sigma": "model units", "residual_l2": "raw k-space L2",
                     "diagnostic_wall_seconds": "seconds", "boundary_band_px": "pixels",
                     "gradient_rms": "padded complex image units", "cosine": "dimensionless"}}
    with (target / "jacobian_diagnostics.json").open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, ensure_ascii=False, allow_nan=False)
    with (target / "jacobian_diagnostics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(flatten(row) for row in rows)
    return obj
