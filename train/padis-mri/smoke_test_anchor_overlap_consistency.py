"""Smoke tests for AnchorOverlapConsistencyPatchEDMLoss.

Run from the PaDIS-MRI repository root after copying:
  patch_anchor_overlap_consistency_loss.py
to:
  train/padis-mri/training/patch_anchor_overlap_consistency_loss.py

Example:
  PYTHONPATH=train/padis-mri python \
      train/padis-mri/smoke_test_anchor_overlap_consistency.py
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from training.patch_loss_simplified_agg import SimplifiedAggPatchEDMLoss
from training.patch_anchor_overlap_consistency_loss import (
    AnchorOverlapConsistencyPatchEDMLoss,
    _extract_patches,
    _overlap_views,
    _sample_legal_half_shift,
)


class TinyDenoiser(torch.nn.Module):
    """Minimal network with the PaDIS loss-call signature."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.85))
        self.bias = torch.nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        x,
        sigma,
        x_pos=None,
        class_labels=None,
        augment_labels=None,
    ):
        # Include a very small position dependence so both x and x_pos paths
        # are exercised without changing tensor shape.
        pos_term = 0.0
        if x_pos is not None:
            pos_term = 1e-3 * x_pos[:, :1]
        return self.gain * x + self.bias + pos_term


def _seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_grad(module):
    return {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in module.named_parameters()
    }


def test_lambda_zero_exact_equivalence(device):
    """lambda_oc=0 must be identical to the existing single-AGG path."""
    images = torch.randn(4, 2, 128, 128, device=device)
    base = SimplifiedAggPatchEDMLoss(
        agg_rho=0.65,
        agg_gamma=1.0,
    )
    new = AnchorOverlapConsistencyPatchEDMLoss(
        agg_rho=0.65,
        agg_gamma=1.0,
        lambda_oc=0.0,
        active_patch_size=64,
    )

    net = TinyDenoiser().to(device)

    _seed_everything(12345)
    net.zero_grad(set_to_none=True)
    loss_base = base(
        net=net,
        images=images,
        patch_size=64,
        resolution=128,
        labels=None,
        augment_pipe=None,
    )
    scalar_base = loss_base.sum()
    scalar_base.backward()
    grad_base = _clone_grad(net)

    _seed_everything(12345)
    net.zero_grad(set_to_none=True)
    loss_new = new(
        net=net,
        images=images,
        patch_size=64,
        resolution=128,
        labels=None,
        augment_pipe=None,
    )
    scalar_new = loss_new.sum()
    scalar_new.backward()
    grad_new = _clone_grad(net)

    if not torch.equal(loss_base, loss_new):
        max_diff = (loss_base - loss_new).abs().max().item()
        raise AssertionError(
            f"lambda=0 loss is not bitwise identical; max diff={max_diff}"
        )

    for name in grad_base:
        if grad_base[name] is None and grad_new[name] is None:
            continue
        if not torch.equal(grad_base[name], grad_new[name]):
            max_diff = (grad_base[name] - grad_new[name]).abs().max().item()
            raise AssertionError(
                f"lambda=0 gradient differs for {name}; max diff={max_diff}"
            )

    print("[PASS] lambda_oc=0 exactly matches SimplifiedAggPatchEDMLoss")


def test_half_shift_and_shared_physical_noise(device):
    """The two views must share identical noise on the same physical pixels."""
    batch = 8
    height = width = 160
    patch_size = 64

    # Explicit legal anchors covering central and boundary cases.
    top = torch.tensor([0, 16, 32, 48, 64, 80, 96, 32], device=device)
    left = torch.tensor([0, 16, 32, 48, 64, 80, 96, 64], device=device)

    _seed_everything(7)
    top_aux, left_aux, direction, overlap = _sample_legal_half_shift(
        top, left, height, width, patch_size
    )

    stride = patch_size - overlap
    displacement = (
        (top_aux - top).abs() + (left_aux - left).abs()
    )
    if not torch.all(displacement == stride):
        raise AssertionError("auxiliary patch is not an exact half-shift")

    noise_full = torch.randn(batch, 2, height, width, device=device)
    noise_anchor = _extract_patches(
        noise_full, top, left, patch_size
    )
    noise_aux = _extract_patches(
        noise_full, top_aux, left_aux, patch_size
    )
    overlap_anchor, overlap_aux = _overlap_views(
        noise_anchor, noise_aux, direction, overlap
    )

    if not torch.equal(overlap_anchor, overlap_aux):
        max_diff = (overlap_anchor - overlap_aux).abs().max().item()
        raise AssertionError(
            f"shared physical overlap noise mismatch; max diff={max_diff}"
        )

    print("[PASS] auxiliary geometry is legal half-shift")
    print("[PASS] physical overlap uses exactly the same noise realization")


def test_positive_lambda_forward_backward(device):
    """Positive OC weight must produce finite loss and finite gradients."""
    images = torch.randn(4, 2, 160, 160, device=device)
    net = TinyDenoiser().to(device)
    loss_fn = AnchorOverlapConsistencyPatchEDMLoss(
        agg_rho=0.65,
        agg_gamma=1.0,
        lambda_oc=0.3,
        active_patch_size=64,
    )

    _seed_everything(99)
    net.zero_grad(set_to_none=True)
    loss = loss_fn(
        net=net,
        images=images,
        patch_size=64,
        resolution=160,
        labels=None,
        augment_pipe=None,
    )

    if not torch.isfinite(loss).all():
        raise AssertionError("positive-lambda loss contains NaN/Inf")

    loss.sum().backward()
    for name, param in net.named_parameters():
        if param.grad is None:
            raise AssertionError(f"missing gradient for {name}")
        if not torch.isfinite(param.grad).all():
            raise AssertionError(f"non-finite gradient for {name}")

    print("[PASS] lambda_oc>0 forward/backward is finite")


def test_nonactive_scale_delegates(device):
    """Non-active scales must remain ordinary single-AGG training."""
    images = torch.randn(4, 2, 128, 128, device=device)
    base = SimplifiedAggPatchEDMLoss(
        agg_rho=0.65,
        agg_gamma=1.0,
    )
    new = AnchorOverlapConsistencyPatchEDMLoss(
        agg_rho=0.65,
        agg_gamma=1.0,
        lambda_oc=0.3,
        active_patch_size=64,
    )
    net = TinyDenoiser().to(device)

    _seed_everything(2026)
    out_base = base(
        net=net,
        images=images,
        patch_size=32,
        resolution=128,
        labels=None,
        augment_pipe=None,
    )
    _seed_everything(2026)
    out_new = new(
        net=net,
        images=images,
        patch_size=32,
        resolution=128,
        labels=None,
        augment_pipe=None,
    )

    if not torch.equal(out_base, out_new):
        max_diff = (out_base - out_new).abs().max().item()
        raise AssertionError(
            f"non-active scale changed; max diff={max_diff}"
        )

    print("[PASS] non-active patch scales exactly delegate to single-AGG")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    test_lambda_zero_exact_equivalence(device)
    test_half_shift_and_shared_physical_noise(device)
    test_positive_lambda_forward_backward(device)
    test_nonactive_scale_delegates(device)
    print("\nAll anchor-OC smoke tests passed.")


if __name__ == "__main__":
    main()
