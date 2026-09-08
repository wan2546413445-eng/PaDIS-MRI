"""Focused CPU smoke test for MRI projected-target Scan-TTA."""

import random
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .cg_sense import cg_sense, measurement_residual_sse, sense_forward
    from .scan_tta import (
        ScanTTAAdapter,
        complex_mse,
        denoised_from_patches_tta,
        get_tta_indices,
        refinement_diffusion_indices,
    )
except ImportError:
    from cg_sense import cg_sense, measurement_residual_sse, sense_forward
    from scan_tta import (
        ScanTTAAdapter,
        complex_mse,
        denoised_from_patches_tta,
        get_tta_indices,
        refinement_diffusion_indices,
    )


class TinyPatchPrior(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x, sigma, x_pos=None, class_labels=None):
        del sigma, x_pos, class_labels
        return x + 0.1 * self.conv(x)


def _prediction(net, x_complex, pos, indices, pad):
    x_real = torch.view_as_real(x_complex.squeeze(1)).permute(0, 3, 1, 2)
    output = denoised_from_patches_tta(net, x_real, torch.tensor(0.5), pos, indices)
    cropped = output[:, :, pad:-pad, pad:-pad]
    return torch.complex(cropped[:, 0], cropped[:, 1]).unsqueeze(1)


def main() -> None:
    torch.manual_seed(7)
    random.seed(7)

    # A/B: a five-step projected target improves acquired-k-space residual.
    maps = torch.ones(1, 1, 4, 4, dtype=torch.complex64)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., ::2, :] = 1
    truth = torch.complex(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
    measurement = sense_forward(truth, maps, mask)
    initial = torch.zeros_like(truth)
    before = measurement_residual_sse(initial, measurement, maps, mask)
    target, cg_info = cg_sense(
        initial, measurement, maps, mask, iterations=5, return_info=True
    )
    after = measurement_residual_sse(target, measurement, maps, mask)
    assert after < before
    assert not torch.equal(target, initial)

    # C/D: projected-target loss reaches all trainable parameters and changes output.
    net = TinyPatchPrior()
    adapter = ScanTTAAdapter(net, lr=1e-3)
    optimizer = adapter.new_subject_optimizer()
    padded = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    pos = torch.zeros(1, 2, 8, 8)
    spaced = np.array([0, 2, 4])
    indices = get_tta_indices(spaced, 3, 2, 2, random.Random(99))
    prediction_before = _prediction(net, padded, pos, indices, pad=2).detach()
    adapter.unfreeze()
    optimizer.zero_grad(set_to_none=True)
    prediction = _prediction(net, padded, pos, indices, pad=2)
    projected, _ = cg_sense(
        prediction.detach(), measurement, maps, mask, iterations=5
    )
    loss = complex_mse(prediction, projected)
    loss.backward()
    assert all(parameter.grad is not None for parameter in net.parameters())
    optimizer.step()
    prediction_after = _prediction(net, padded, pos, indices, pad=2).detach()
    assert not torch.equal(prediction_before, prediction_after)
    optimizer.zero_grad(set_to_none=True)
    adapter.freeze()

    # E: frozen theta receives no .grad during an x-only PaDIS-style graph.
    x_only = torch.randn(1, 2, 2, 2, requires_grad=True)
    x_output = net(x_only, torch.tensor(0.5), torch.zeros_like(x_only), None)
    x_grad = torch.autograd.grad(x_output.square().sum(), x_only)[0]
    assert torch.isfinite(x_grad).all()
    assert all(parameter.grad is None for parameter in net.parameters())

    # F: the 78/K=10 schedule is exactly seven events, 70 down to 10.
    event_indices = refinement_diffusion_indices(78, 10)
    assert event_indices == [70, 60, 50, 40, 30, 20, 10]

    # G: local partitioning and one full refinement do not move global RNGs.
    adapter.reset()
    rng_optimizer = adapter.new_subject_optimizer()
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    adapter.refine(
        fixed_x=padded,
        sigma=torch.tensor(0.5),
        latents_pos=pos,
        measurement=measurement,
        inverseop=SimpleNamespace(maps=maps, mask=mask),
        optimizer=rng_optimizer,
        spaced=spaced,
        patches=3,
        pad=2,
        psize=2,
        refinement_iters=1,
        cg_iters=1,
        event=1,
        outer_step=9,
        diffusion_index=70,
        local_seed=12345,
    )
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)

    # H: subject reset restores the exact pretrained CPU state.
    assert not adapter.is_exactly_reset()
    adapter.reset()
    assert adapter.is_exactly_reset()

    print("CG residual:", cg_info.residual_before, "->", cg_info.residual_after)
    print("Projected target differs from D: PASS")
    print("Full-network gradient and output change: PASS")
    print("Frozen-theta x-only gradient separation: PASS")
    print("Event schedule:", event_indices)
    print("Global Python/Torch RNG preservation: PASS")
    print("Subject reset: PASS")
    print("Scan-TTA smoke test: PASS")


if __name__ == "__main__":
    main()
