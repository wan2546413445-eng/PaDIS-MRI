"""Focused CPU smoke tests for Global Cross-Conditioned PaDIS."""

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
        measurement_residual_sse,
        proximal_cg,
        sense_adjoint,
        sense_forward,
    )
    from .gcc_adapter import GCCScanAdapter
    from .kspace_split import ACS_START, ACS_STOP, split_mask7_columns
    from .recon_gcc import (
        GCC_TTA_EVENTS,
        conditional_score_from_clean,
        reconstruct_gcc,
        formal_budget,
    )
except ImportError:
    from cg_sense import (
        acquired_measurement_mse,
        measurement_residual_sse,
        proximal_cg,
        sense_adjoint,
        sense_forward,
    )
    from gcc_adapter import GCCScanAdapter
    from kspace_split import ACS_START, ACS_STOP, split_mask7_columns
    from recon_gcc import (
        GCC_TTA_EVENTS,
        conditional_score_from_clean,
        reconstruct_gcc,
        formal_budget,
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

    # A: proximal CG reduces the acquired residual while retaining both terms.
    maps_small = torch.ones(1, 1, 8, 8, dtype=torch.complex64)
    mask_small = torch.ones(1, 1, 8, 8)
    truth_small = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    measurement_small = sense_forward(truth_small, maps_small, mask_small)
    prior_small = torch.zeros_like(truth_small)
    residual_before = measurement_residual_sse(
        prior_small, measurement_small, maps_small, mask_small
    )
    conditional_small, cg_info = proximal_cg(
        prior_small,
        measurement_small,
        maps_small,
        mask_small,
        iterations=5,
        gamma=1.0,
        differentiable=False,
        return_info=True,
    )
    residual_after = measurement_residual_sse(
        conditional_small, measurement_small, maps_small, mask_small
    )
    least_squares_small = sense_adjoint(
        measurement_small, maps_small, mask_small
    )
    assert residual_after < residual_before
    assert not torch.allclose(conditional_small, prior_small)
    assert not torch.allclose(conditional_small, least_squares_small)
    assert not conditional_small.requires_grad

    # B: the formal split is reused and the holdout loss reaches theta.
    full_mask = formal_mask7()
    split = split_mask7_columns(full_mask, seed=123)
    assert torch.equal(split.mask_cond + split.mask_hold, full_mask)
    assert len(split.holdout_columns) == 11

    maps = torch.ones(1, 1, 384, 384, dtype=torch.complex64)
    truth = torch.randn(1, 1, 384, 384, dtype=torch.complex64)
    measurement = sense_forward(truth, maps, full_mask)
    net = TinyPatchPrior()
    adapter = GCCScanAdapter(net, lr=1e-3, gamma=1.0)
    optimizer = adapter.new_subject_optimizer()
    current_x = torch.randn(1, 1, 512, 512, dtype=torch.complex64)
    positions = torch.zeros(1, 2, 512, 512)
    spaced = np.linspace(0, 384, 7, dtype=int)
    parameters_before = {
        name: value.detach().clone()
        for name, value in net.named_parameters()
    }
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    adaptation = adapter.refine(
        current_x=current_x,
        sigma=torch.tensor(0.5),
        latents_pos=positions,
        measurement=measurement,
        maps=maps,
        full_mask=full_mask,
        optimizer=optimizer,
        spaced=spaced,
        patches=7,
        pad=64,
        psize=64,
        refinement_iters=1,
        cg_iters=2,
        event=1,
        subject_seed=123,
    )
    assert np.isfinite(adaptation.rows[0]["holdout_loss"])
    assert any(
        not torch.equal(value.detach(), parameters_before[name])
        for name, value in net.named_parameters()
    )
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    adapter.reset()
    assert adapter.is_exactly_reset()

    # Direct differentiable operator check with the true held-out mean.
    differentiable_prior = torch.zeros_like(truth, requires_grad=True)
    z_cond, _ = proximal_cg(
        differentiable_prior,
        split.mask_cond * measurement,
        maps,
        split.mask_cond,
        iterations=2,
        gamma=1.0,
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

    # C: the formal score target is the embedded full-conditioned clean image.
    unconditional_padded = torch.randn(
        1, 1, 12, 12, dtype=torch.complex64
    )
    conditional_crop = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    x_noisy = torch.randn(1, 1, 12, 12, dtype=torch.complex64)
    sigma = torch.tensor(0.5)
    score, conditional_padded = conditional_score_from_clean(
        conditional_crop, unconditional_padded, x_noisy, sigma, pad=2
    )
    assert torch.equal(conditional_padded[:, :, 2:10, 2:10], conditional_crop)
    assert torch.equal(conditional_padded[:, :, :2], unconditional_padded[:, :, :2])
    assert torch.allclose(score, (conditional_padded - x_noisy) / sigma.square())
    reconstruction_source = inspect.getsource(reconstruct_gcc)
    assert "autograd.grad" not in reconstruction_source
    assert "proximal_cg(" in reconstruction_source
    assert "conditional_score_from_clean(" in reconstruction_source

    # D/E: fixed event schedule and complete method budget.
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
        "Proximal CG residual:",
        cg_info.measurement_residual_before,
        "->",
        cg_info.measurement_residual_after,
    )
    print("Proximal prior/measurement coupling: PASS")
    print("Differentiable holdout TTA gradient: PASS")
    print("Conditional score and no old DPS gradient: PASS")
    print("TTA Python/Torch RNG isolation: PASS")
    print("Formal budget:", budget)
    print("GCC-PaDIS smoke test: PASS")


if __name__ == "__main__":
    main()
