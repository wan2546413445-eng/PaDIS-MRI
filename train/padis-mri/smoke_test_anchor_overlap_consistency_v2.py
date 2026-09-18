"""Smoke tests for boundary-focused Anchor-OC v2.

Copy this file to:
    train/padis-mri/smoke_test_anchor_overlap_consistency_v2.py

Run:
    PYTHONPATH=train/padis-mri \
    python train/padis-mri/smoke_test_anchor_overlap_consistency_v2.py
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
    _interior_boundary_terms,
    _overlap_views,
    _sample_legal_half_shift,
)


class TinyDenoiser(torch.nn.Module):
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
        pos_term = 0.0
        if x_pos is not None:
            pos_term = 1e-3 * x_pos[:, :1]
        return self.gain * x + self.bias + pos_term


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grads(module):
    return {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in module.named_parameters()
    }


def test_lambda_zero_exact(device):
    images = torch.randn(4, 2, 128, 128, device=device)
    base = SimplifiedAggPatchEDMLoss(agg_rho=0.65, agg_gamma=1.0)
    v2 = AnchorOverlapConsistencyPatchEDMLoss(
        agg_rho=0.65,
        agg_gamma=1.0,
        lambda_oc=0.0,
        active_patch_size=64,
    )
    net = TinyDenoiser().to(device)

    seed_all(12345)
    net.zero_grad(set_to_none=True)
    a = base(
        net=net,
        images=images,
        patch_size=64,
        resolution=128,
        labels=None,
        augment_pipe=None,
    )
    a.sum().backward()
    ga = grads(net)

    seed_all(12345)
    net.zero_grad(set_to_none=True)
    b = v2(
        net=net,
        images=images,
        patch_size=64,
        resolution=128,
        labels=None,
        augment_pipe=None,
    )
    b.sum().backward()
    gb = grads(net)

    assert torch.equal(a, b), "lambda=0 loss changed"
    for key in ga:
        if ga[key] is None and gb[key] is None:
            continue
        assert torch.equal(ga[key], gb[key]), f"lambda=0 grad changed: {key}"

    print("[PASS] lambda_oc=0 exact single-AGG loss/gradient equivalence")


def test_geometry_noise_and_boundary_focus(device):
    batch = 8
    h = w = 160
    p = 64

    top = torch.tensor([0, 16, 32, 48, 64, 80, 96, 32], device=device)
    left = torch.tensor([0, 16, 32, 48, 64, 80, 96, 64], device=device)

    seed_all(7)
    ta, la, direction, overlap = _sample_legal_half_shift(
        top, left, h, w, p
    )

    displacement = (ta - top).abs() + (la - left).abs()
    assert torch.all(displacement == p - overlap)

    noise_full = torch.randn(batch, 2, h, w, device=device)
    n1 = _extract_patches(noise_full, top, left, p)
    n2 = _extract_patches(noise_full, ta, la, p)
    o1, o2 = _overlap_views(n1, n2, direction, overlap)
    assert torch.equal(o1, o2), "physical overlap noise is not identical"

    (
        interior,
        boundary,
        confidence,
        active,
        d_int,
        d_bnd,
        band,
    ) = _interior_boundary_terms(
        o1,
        o2,
        direction,
        p,
        overlap,
    )

    assert band == 8, f"expected P64 boundary band 8, got {band}"
    assert bool(active.any()), "boundary-focused active set is empty"
    assert torch.all(d_int[active] > d_bnd[active])
    assert torch.all(confidence[~active] == 0)
    assert torch.isfinite(confidence).all()

    # For half-overlap P64 and band 8, 16 of 32 canonical overlap columns are
    # active (two 8-pixel edge bands), so active fraction should be 0.5.
    frac = float(active.float().mean().item())
    assert abs(frac - 0.5) < 1e-6, f"unexpected active fraction: {frac}"

    print("[PASS] legal fixed half-shift")
    print("[PASS] identical noise on the same physical overlap pixels")
    print("[PASS] P64 boundary-focused active fraction = 0.5")
    print("[PASS] interior view is strictly farther from the patch boundary")


def test_positive_lambda(device):
    images = torch.randn(4, 2, 160, 160, device=device)
    net = TinyDenoiser().to(device)
    loss_fn = AnchorOverlapConsistencyPatchEDMLoss(
        agg_rho=0.65,
        agg_gamma=1.0,
        lambda_oc=0.3,
        active_patch_size=64,
    )

    seed_all(99)
    net.zero_grad(set_to_none=True)
    loss = loss_fn(
        net=net,
        images=images,
        patch_size=64,
        resolution=160,
        labels=None,
        augment_pipe=None,
    )

    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"missing grad: {name}"
        assert torch.isfinite(p.grad).all(), f"nonfinite grad: {name}"

    print("[PASS] lambda_oc>0 forward/backward finite")


def test_nonactive_exact(device):
    images = torch.randn(4, 2, 128, 128, device=device)
    base = SimplifiedAggPatchEDMLoss(agg_rho=0.65, agg_gamma=1.0)
    v2 = AnchorOverlapConsistencyPatchEDMLoss(
        agg_rho=0.65,
        agg_gamma=1.0,
        lambda_oc=0.3,
        active_patch_size=64,
    )
    net = TinyDenoiser().to(device)

    seed_all(2026)
    a = base(
        net=net,
        images=images,
        patch_size=32,
        resolution=128,
        labels=None,
        augment_pipe=None,
    )
    seed_all(2026)
    b = v2(
        net=net,
        images=images,
        patch_size=32,
        resolution=128,
        labels=None,
        augment_pipe=None,
    )
    assert torch.equal(a, b)

    print("[PASS] non-active scales exactly delegate to single-AGG")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    test_lambda_zero_exact(device)
    test_geometry_noise_and_boundary_focus(device)
    test_positive_lambda(device)
    test_nonactive_exact(device)
    print("\nAll Anchor-OC v2 smoke tests passed.")


if __name__ == "__main__":
    main()
