#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate standalone AGGA mechanism visualizations only.
No statistics / no metric bar charts.

Outputs:
  01_mri_magnitude.png
  02_anatomical_support.png
  03_patch_anatomical_content_Au.png
  04_normalized_patch_gradient_Gu.png
  05_uniform_reference.png
  06_simplified_agga_logratio.png
  07_original_agga_mild_logratio.png
  08_agga_mechanism_all_in_one.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Formal geometry
# ============================================================
ORIGINAL_IMAGE_SIZE = 384
PADDING = 64

# ============================================================
# Simplified AGGA constants
# ============================================================
SUPPORT_THRESHOLD_RATIO = 0.05
GRADIENT_CLIP_QUANTILE = 0.95
SIMPLIFIED_RHO = 0.65
SIMPLIFIED_GAMMA = 1.0

# ============================================================
# Original AGGA-mild constants
# ============================================================
ORIGINAL_RHO = 0.65
ORIGINAL_THR_RATIO = 0.05
ORIGINAL_BASE = 0.08
ORIGINAL_RADIUS_SCALE = 1.15
ORIGINAL_TAU_SCALE = 0.30
ORIGINAL_MIN_PIXELS = 64
ORIGINAL_GRAD_ALPHA = 1.0
ORIGINAL_GRAD_CLIP_Q = 0.95
ORIGINAL_GATE_BASE = 0.25

DEFAULT_DATA = (
    "/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/"
    "brain_train_d384_s200/32dB/noisy.pt"
)


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AGGA mechanism visualizations only."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(DEFAULT_DATA),
        help="Path to noisy.pt",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Zero-based sample index",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        choices=[16, 32, 64],
        default=64,
        help="Patch size used for visualization",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/agga_core_visualization"),
        help="Output directory",
    )
    return parser.parse_args()


# ============================================================
# Data loading
# ============================================================
def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_training_sample(path: Path, sample_index: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")

    data = _torch_load(path)
    if not isinstance(data, dict) or "x_est_gt" not in data:
        raise KeyError(f"{path} must contain key 'x_est_gt'.")

    samples = data["x_est_gt"]
    if not torch.is_tensor(samples):
        raise TypeError("data['x_est_gt'] must be a torch.Tensor.")

    if sample_index < 0 or sample_index >= samples.shape[0]:
        raise IndexError(
            f"sample_index={sample_index} is outside [0, {samples.shape[0]-1}]"
        )

    sample = samples[sample_index].detach().cpu()

    if torch.is_complex(sample):
        if sample.ndim != 2:
            raise ValueError(f"Expected complex HxW sample, got shape {tuple(sample.shape)}")
        image = torch.view_as_real(sample).permute(2, 0, 1).contiguous()
    elif sample.ndim == 3 and sample.shape[0] >= 2:
        image = sample[:2]
    elif sample.ndim == 3 and sample.shape[-1] >= 2:
        image = sample[..., :2].permute(2, 0, 1).contiguous()
    else:
        raise ValueError(
            "Expected complex [H,W] or real/imag channels in [C,H,W] or [H,W,C]."
        )

    image = image[:2].to(torch.float32)

    if tuple(image.shape) != (2, ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE):
        raise ValueError(
            f"Expected (2,384,384), got {tuple(image.shape)}"
        )

    padded = F.pad(image, (PADDING, PADDING, PADDING, PADDING), value=0.0)
    return padded


# ============================================================
# Core maps
# ============================================================
def structure_maps(image: torch.Tensor):
    use_channels = min(2, image.shape[0])
    magnitude = image[:use_channels].float().square().sum(dim=0).sqrt()

    peak = magnitude.amax()
    if not torch.isfinite(peak) or float(peak.item()) <= 0:
        raise ValueError("Non-positive magnitude peak.")

    support = (magnitude > SUPPORT_THRESHOLD_RATIO * peak).float()

    grad_x = torch.zeros_like(magnitude)
    grad_y = torch.zeros_like(magnitude)
    grad_x[:, 1:] = magnitude[:, 1:] - magnitude[:, :-1]
    grad_y[1:, :] = magnitude[1:, :] - magnitude[:-1, :]
    gradient = torch.sqrt(grad_x.square() + grad_y.square())

    return magnitude, support, gradient


def patch_average(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    return F.avg_pool2d(
        image[None, None],
        kernel_size=(patch_size, patch_size),
        stride=1,
    )[0, 0]


def normalize_simplified_gradient(gradient_score: torch.Tensor) -> torch.Tensor:
    positive = gradient_score[gradient_score > 0]
    if positive.numel() == 0:
        return torch.zeros_like(gradient_score)

    clip_value = torch.quantile(
        positive, GRADIENT_CLIP_QUANTILE
    ).clamp_min(torch.finfo(gradient_score.dtype).eps)

    return (gradient_score / clip_value).clamp(0.0, 1.0)


def normalize_original_gradient(gradient_score: torch.Tensor) -> torch.Tensor:
    positive = gradient_score.flatten()
    positive = positive[positive > 0]
    if positive.numel() < 8:
        return torch.zeros_like(gradient_score)

    clip_value = torch.quantile(positive, ORIGINAL_GRAD_CLIP_Q).clamp_min(1e-8)
    return (gradient_score / clip_value).clamp(0.0, 1.0)


def uniform_probability(shape):
    count = int(shape[0] * shape[1])
    return torch.full(shape, 1.0 / count, dtype=torch.float32)


def mix_score_with_uniform(score: torch.Tensor, rho: float, uniform: torch.Tensor):
    flat = torch.nan_to_num(score.flatten(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = flat.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return uniform.clone()
    guided = (flat / total).reshape_as(score)
    return (1.0 - rho) * uniform + rho * guided


def simplified_distribution(support: torch.Tensor, gradient: torch.Tensor, patch_size: int):
    anatomy = patch_average(support, patch_size)
    gradient_mean = patch_average(gradient, patch_size)
    gradient_normalized = normalize_simplified_gradient(gradient_mean)
    score = anatomy * (1.0 + SIMPLIFIED_GAMMA * gradient_normalized)
    uniform = uniform_probability(tuple(score.shape))
    probability = mix_score_with_uniform(score, SIMPLIFIED_RHO, uniform)
    return probability, anatomy, gradient_normalized


def original_distribution(magnitude: torch.Tensor, patch_size: int, uniform: torch.Tensor):
    height, width = magnitude.shape
    max_top = height - patch_size
    max_left = width - patch_size

    peak = magnitude.amax().clamp_min(1e-8)
    support = (magnitude > ORIGINAL_THR_RATIO * peak).float()
    mass = support.sum()

    grad_x = torch.zeros_like(magnitude)
    grad_y = torch.zeros_like(magnitude)
    grad_x[:, 1:] = magnitude[:, 1:] - magnitude[:, :-1]
    grad_y[1:, :] = magnitude[1:, :] - magnitude[:-1, :]
    gradient = torch.sqrt(grad_x.square() + grad_y.square())

    if float(mass.item()) < ORIGINAL_MIN_PIXELS:
        return uniform.clone()

    yy = torch.arange(height, dtype=torch.float32).view(height, 1).expand(height, width)
    xx = torch.arange(width, dtype=torch.float32).view(1, width).expand(height, width)
    center_y = (support * yy).sum() / mass
    center_x = (support * xx).sum() / mass

    radius = torch.sqrt(mass / math.pi) * ORIGINAL_RADIUS_SCALE
    tau = torch.clamp(radius * ORIGINAL_TAU_SCALE, min=8.0)

    candidate_y = torch.arange(max_top + 1, dtype=torch.float32) + patch_size / 2.0
    candidate_x = torch.arange(max_left + 1, dtype=torch.float32) + patch_size / 2.0
    grid_y, grid_x = torch.meshgrid(candidate_y, candidate_x, indexing="ij")
    distance = torch.sqrt((grid_y - center_y) ** 2 + (grid_x - center_x) ** 2)
    outside = torch.clamp(distance - radius, min=0.0)
    radial_score = torch.exp(-0.5 * (outside / tau) ** 2)

    gradient_mean = patch_average(gradient, patch_size)
    gradient_normalized = normalize_original_gradient(gradient_mean)
    support_fraction = patch_average(support, patch_size).clamp(0.0, 1.0)
    anatomy_gate = ORIGINAL_GATE_BASE + (1.0 - ORIGINAL_GATE_BASE) * support_fraction

    score = (
        ORIGINAL_BASE
        + radial_score
        * anatomy_gate
        * (1.0 + ORIGINAL_GRAD_ALPHA * gradient_normalized)
    )
    score = score.flatten().clamp_min(1e-8).reshape_as(uniform)
    guided = score / score.sum()
    probability = (1.0 - ORIGINAL_RHO) * uniform + ORIGINAL_RHO * guided
    return probability


# ============================================================
# Plot helpers
# ============================================================
def candidate_extent(height: int, width: int, patch_size: int):
    half = patch_size / 2.0
    return (
        half - 0.5,
        width - half + 0.5,
        height - half + 0.5,
        half - 0.5,
    )


def shared_crop_from_support(support: np.ndarray, patch_size: int):
    rows, cols = np.nonzero(support > 0)
    h, w = support.shape
    edge = patch_size / 2.0 - 0.5
    if rows.size == 0:
        return edge, w - edge, edge, h - edge
    margin = float(patch_size) / 2.0
    x0 = max(edge, float(cols.min()) - margin)
    x1 = min(w - edge, float(cols.max() + 1) + margin)
    y0 = max(edge, float(rows.min()) - margin)
    y1 = min(h - edge, float(rows.max() + 1) + margin)
    return x0, x1, y0, y1


def style_axis(ax, crop):
    x0, x1, y0, y1 = crop
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#555555")


def save_single_map(
    path: Path,
    array: np.ndarray,
    title: str,
    cmap: str,
    extent,
    crop,
    vmin=None,
    vmax=None,
    colorbar_label=None,
    overlay_magnitude=None,
    overlay_alpha=0.65,
):
    fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)

    if overlay_magnitude is not None:
        ax.imshow(overlay_magnitude, cmap="gray", extent=(0, overlay_magnitude.shape[1], overlay_magnitude.shape[0], 0))
        im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, alpha=overlay_alpha)
    else:
        im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x (pixel)", fontsize=9)
    ax.set_ylabel("y (pixel)", fontsize=9)
    style_axis(ax, crop)

    if colorbar_label is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.03)
        cbar.set_label(colorbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_uniform_reference(path: Path, magnitude: np.ndarray, patch_size: int, crop):
    fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
    ax.imshow(magnitude, cmap="gray", extent=(0, magnitude.shape[1], magnitude.shape[0], 0))
    ax.text(
        0.5, 0.06,
        "1× everywhere / log₂ ratio = 0",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10,
        color="#202020",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="#666666",
            linewidth=0.6,
            alpha=0.90,
        ),
    )
    ax.set_title("Uniform", fontsize=12)
    ax.set_xlabel("x (pixel)", fontsize=9)
    ax.set_ylabel("y (pixel)", fontsize=9)
    style_axis(ax, crop)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_all_in_one(
    path: Path,
    magnitude: np.ndarray,
    support: np.ndarray,
    anatomy: np.ndarray,
    gradient_norm: np.ndarray,
    log_relative_simplified: np.ndarray,
    log_relative_original: np.ndarray,
    patch_size: int,
):
    height, width = magnitude.shape
    image_extent = (0.0, float(width), float(height), 0.0)
    score_extent = candidate_extent(height, width, patch_size)

    crop_image = shared_crop_from_support(support, patch_size)
    crop_patch = (score_extent[0], score_extent[1], score_extent[3], score_extent[2])

    vmax = max(
        0.25,
        float(np.quantile(np.abs(log_relative_simplified[np.isfinite(log_relative_simplified)]), 0.99)),
        float(np.quantile(np.abs(log_relative_original[np.isfinite(log_relative_original)]), 0.99)),
    )

    fig = plt.figure(figsize=(13.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

    # top row
    im0 = axes[0].imshow(magnitude, cmap="gray", extent=image_extent)
    axes[0].set_title("(a) MRI magnitude", fontsize=11)

    im1 = axes[1].imshow(support, cmap="gray", vmin=0, vmax=1, extent=image_extent)
    axes[1].set_title("(b) Anatomical support", fontsize=11)

    im2 = axes[2].imshow(anatomy, cmap="viridis", vmin=0, vmax=1, extent=score_extent)
    axes[2].set_title("(c) Patch anatomical content, A(u)", fontsize=11)

    im3 = axes[3].imshow(gradient_norm, cmap="viridis", vmin=0, vmax=1, extent=score_extent)
    axes[3].set_title("(d) Normalized patch gradient, G(u)", fontsize=11)

    # bottom row
    axes[4].imshow(magnitude, cmap="gray", extent=image_extent)
    axes[4].text(
        0.5, 0.06,
        "1× everywhere / log₂ ratio = 0",
        transform=axes[4].transAxes,
        ha="center", va="bottom",
        fontsize=9,
        color="#202020",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="#666666",
            linewidth=0.6,
            alpha=0.90,
        ),
    )
    axes[4].set_title("(e) Uniform", fontsize=11)

    axes[5].imshow(magnitude, cmap="gray", extent=image_extent)
    den1 = axes[5].imshow(
        log_relative_simplified,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=score_extent,
        alpha=0.68,
    )
    axes[5].set_title("(f) Simplified AGGA", fontsize=11)

    axes[6].imshow(magnitude, cmap="gray", extent=image_extent)
    axes[6].imshow(
        log_relative_original,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=score_extent,
        alpha=0.68,
    )
    axes[6].set_title("(g) Original AGGA-mild", fontsize=11)

    axes[7].axis("off")

    for ax in axes[:2]:
        ax.set_xlabel("x (pixel)", fontsize=9)
        ax.set_ylabel("y (pixel)", fontsize=9)
        style_axis(ax, crop_image)

    for ax in axes[2:4]:
        ax.set_xlabel("x (pixel)", fontsize=9)
        ax.set_ylabel("y (pixel)", fontsize=9)
        style_axis(ax, crop_image)

    axes[4].set_xlabel("x (pixel)", fontsize=9)
    axes[4].set_ylabel("y (pixel)", fontsize=9)
    style_axis(axes[4], crop_patch)

    axes[5].set_xlabel("x (pixel)", fontsize=9)
    axes[5].set_ylabel("y (pixel)", fontsize=9)
    style_axis(axes[5], crop_patch)

    axes[6].set_xlabel("x (pixel)", fontsize=9)
    axes[6].set_ylabel("y (pixel)", fontsize=9)
    style_axis(axes[6], crop_patch)

    cbar = fig.colorbar(den1, ax=[axes[4], axes[5], axes[6]], orientation="horizontal", shrink=0.86, pad=0.08)
    cbar.set_label(r"Relative sampling density, $\log_2(p/p_{\mathrm{uniform}})$", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    padded_image = load_training_sample(args.data, args.sample_index)
    magnitude, support, gradient = structure_maps(padded_image)

    p_simplified, anatomy, gradient_norm = simplified_distribution(
        support, gradient, args.patch_size
    )
    p_uniform = uniform_probability(tuple(p_simplified.shape))
    p_original = original_distribution(magnitude, args.patch_size, p_uniform)

    magnitude_np = magnitude.numpy()
    support_np = support.numpy()
    anatomy_np = anatomy.numpy()
    gradient_np = gradient_norm.numpy()

    uniform_np = p_uniform.numpy().astype(np.float64, copy=False)
    simplified_np = p_simplified.numpy().astype(np.float64, copy=False)
    original_np = p_original.numpy().astype(np.float64, copy=False)

    log_relative_simplified = np.log2(simplified_np / uniform_np)
    log_relative_original = np.log2(original_np / uniform_np)

    h, w = magnitude_np.shape
    image_extent = (0.0, float(w), float(h), 0.0)
    score_extent = candidate_extent(h, w, args.patch_size)

    crop_image = shared_crop_from_support(support_np, args.patch_size)
    crop_patch = (score_extent[0], score_extent[1], score_extent[3], score_extent[2])

    # single panels
    save_single_map(
        args.output_dir / "01_mri_magnitude.png",
        magnitude_np,
        "MRI magnitude",
        cmap="gray",
        extent=image_extent,
        crop=crop_image,
    )

    save_single_map(
        args.output_dir / "02_anatomical_support.png",
        support_np,
        "Anatomical support",
        cmap="gray",
        extent=image_extent,
        crop=crop_image,
        vmin=0,
        vmax=1,
    )

    save_single_map(
        args.output_dir / "03_patch_anatomical_content_Au.png",
        anatomy_np,
        "Patch anatomical content, A(u)",
        cmap="viridis",
        extent=score_extent,
        crop=crop_image,
        vmin=0,
        vmax=1,
        colorbar_label="A(u)",
    )

    save_single_map(
        args.output_dir / "04_normalized_patch_gradient_Gu.png",
        gradient_np,
        "Normalized patch gradient, G(u)",
        cmap="viridis",
        extent=score_extent,
        crop=crop_image,
        vmin=0,
        vmax=1,
        colorbar_label="G(u)",
    )

    save_uniform_reference(
        args.output_dir / "05_uniform_reference.png",
        magnitude_np,
        args.patch_size,
        crop_patch,
    )

    vmax = max(
        0.25,
        float(np.quantile(np.abs(log_relative_simplified[np.isfinite(log_relative_simplified)]), 0.99)),
        float(np.quantile(np.abs(log_relative_original[np.isfinite(log_relative_original)]), 0.99)),
    )

    save_single_map(
        args.output_dir / "06_simplified_agga_logratio.png",
        log_relative_simplified,
        "Simplified AGGA",
        cmap="RdBu_r",
        extent=score_extent,
        crop=crop_patch,
        vmin=-vmax,
        vmax=vmax,
        colorbar_label=r"$\log_2(p/p_{\mathrm{uniform}})$",
        overlay_magnitude=magnitude_np,
        overlay_alpha=0.68,
    )

    save_single_map(
        args.output_dir / "07_original_agga_mild_logratio.png",
        log_relative_original,
        "Original AGGA-mild",
        cmap="RdBu_r",
        extent=score_extent,
        crop=crop_patch,
        vmin=-vmax,
        vmax=vmax,
        colorbar_label=r"$\log_2(p/p_{\mathrm{uniform}})$",
        overlay_magnitude=magnitude_np,
        overlay_alpha=0.68,
    )

    save_all_in_one(
        args.output_dir / "08_agga_mechanism_all_in_one.png",
        magnitude_np,
        support_np,
        anatomy_np,
        gradient_np,
        log_relative_simplified,
        log_relative_original,
        args.patch_size,
    )

    print("\nSaved files:")
    for name in [
        "01_mri_magnitude.png",
        "02_anatomical_support.png",
        "03_patch_anatomical_content_Au.png",
        "04_normalized_patch_gradient_Gu.png",
        "05_uniform_reference.png",
        "06_simplified_agga_logratio.png",
        "07_original_agga_mild_logratio.png",
        "08_agga_mechanism_all_in_one.png",
    ]:
        print(args.output_dir / name)


if __name__ == "__main__":
    main()