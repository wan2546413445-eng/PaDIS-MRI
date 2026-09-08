"""Focused smoke tests for GCC-PaDIS hard global conditioning."""

import inspect
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .cg_sense import (
        acquired_measurement_mse,
        cg_data_consistency,
        measurement_residual_sse,
        sense_forward,
    )
    from .gcc_adapter import GCCScanAdapter
    from .kspace_split import ACS_START, ACS_STOP, split_mask7_columns
    from .recon_gcc import (
        GCC_TTA_EVENTS,
        conditional_score_from_clean,
        formal_budget,
        reconstruct_gcc,
    )
except ImportError:
    from cg_sense import (
        acquired_measurement_mse,
        cg_data_consistency,
        measurement_residual_sse,
        sense_forward,
    )
    from gcc_adapter import GCCScanAdapter
    from kspace_split import ACS_START, ACS_STOP, split_mask7_columns
    from recon_gcc import (
        GCC_TTA_EVENTS,
        conditional_score_from_clean,
        formal_budget,
        reconstruct_gcc,
    )


class TinyPatchPrior(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x, sigma, x_pos=None, class_labels=None):
        del sigma, x_pos, class_labels
        return x + 0.1 * self.conv(x)


def formal_mask7() -> torch.Tensor:
    mask = torch.zeros(1, 1, 384, 384)
    mask[..., :, ACS_START:ACS_STOP] = 1
    mask[..., :, list(range(15)) + list(range(369, 384))] = 1
    return mask


def main() -> None:
    torch.manual_seed(17)
    random.seed(17)

    # A: hard finite CG strongly reduces measured residual and, for a simple
    # single-coil Cartesian case, leaves unacquired Fourier coefficients from D.
    h = w = 16
    maps_small = torch.ones(1, 1, h, w, dtype=torch.complex64)
    mask_small = torch.zeros(1, 1, h, w)
    mask_small[..., :, ::4] = 1
    truth_small = torch.randn(1, 1, h, w, dtype=torch.complex64)
    prior_small = torch.randn_like(truth_small)
    measurement_small = sense_forward(
        truth_small, maps_small, mask_small
    )

    residual_before = measurement_residual_sse(
        prior_small, measurement_small, maps_small, mask_small
    )
    conditional_small, cg_info = cg_data_consistency(
        prior_small,
        measurement_small,
        maps_small,
        mask_small,
        iterations=5,
        differentiable=False,
        return_info=True,
    )
    residual_after = measurement_residual_sse(
        conditional_small, measurement_small, maps_small, mask_small
    )
    assert residual_after < residual_before

    prior_k = torch.fft.fft2(prior_small, dim=(-2, -1), norm="ortho")
    cond_k = torch.fft.fft2(conditional_small, dim=(-2, -1), norm="ortho")
    unsampled = 1 - mask_small
    assert torch.allclose(
        unsampled * cond_k,
        unsampled * prior_k,
        atol=2e-6,
        rtol=2e-6,
    )

    # B: the formal 11/54 split is reused and differentiable conditional CG
    # leaves a nonzero held-out gradient path to the prior.
    full_mask = formal_mask7()
    split = split_mask7_columns(full_mask, seed=123)
    assert torch.equal(split.mask_cond + split.mask_hold, full_mask)
    assert len(split.holdout_columns) == 11

    maps = torch.ones(1, 1, 384, 384, dtype=torch.complex64)
    truth = torch.randn(1, 1, 384, 384, dtype=torch.complex64)
    measurement = sense_forward(truth, maps, full_mask)

    differentiable_prior = torch.zeros_like(truth, requires_grad=True)
    z_cond, _ = cg_data_consistency(
        differentiable_prior,
        split.mask_cond * measurement,
        maps,
        split.mask_cond,
        iterations=2,
        differentiable=True,
    )
    holdout_loss = acquired_measurement_mse(
        z_cond, measurement, maps, split.mask_hold
    )
    gradient_out = torch.autograd.grad(
        holdout_loss, differentiable_prior
    )[0]
    assert torch.isfinite(gradient_out).all()
    assert torch.count_nonzero(gradient_out).item() > 0

    # C: formal score target is the full-conditioned clean image and the old
    # explicit DPS gradient branch is absent.
    unconditional_padded = torch.randn(
        1, 1, 12, 12, dtype=torch.complex64
    )
    conditional_crop = torch.randn(
        1, 1, 8, 8, dtype=torch.complex64
    )
    x_noisy = torch.randn(1, 1, 12, 12, dtype=torch.complex64)
    sigma = torch.tensor(0.5)
    score, conditional_padded = conditional_score_from_clean(
        conditional_crop,
        unconditional_padded,
        x_noisy,
        sigma,
        pad=2,
    )
    assert torch.equal(
        conditional_padded[:, :, 2:10, 2:10], conditional_crop
    )
    assert torch.allclose(
        score, (conditional_padded - x_noisy) / sigma.square()
    )

    source = inspect.getsource(reconstruct_gcc)
    assert "autograd.grad" not in source
    assert "cg_data_consistency(" in source
    assert "conditional_score_from_clean(" in source

    # D: formal schedule and budget.
    assert list(GCC_TTA_EVENTS) == [40, 30, 20, 10]
    budget = formal_budget()
    assert budget == {
        "formal_denoiser_calls": 780,
        "tta_denoiser_calls": 20,
        "denoiser_calls": 800,
        "tta_backprops": 20,
        "full_cg_calls": 780,
        "tta_cg_calls": 20,
        "total_cg_iterations": 4000,
    }

    print(
        "Hard CG residual:",
        cg_info.measurement_residual_before,
        "->",
        cg_info.measurement_residual_after,
    )
    print("Prior-preserving hard data conditioning: PASS")
    print("Differentiable cross-mask gradient: PASS")
    print("Conditional score / no old DPS gradient: PASS")
    print("Formal budget:", budget)
    print("GCC-PaDIS hard-DC smoke test: PASS")


if __name__ == "__main__":
    main()
