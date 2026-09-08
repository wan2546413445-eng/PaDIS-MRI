"""Focused CPU smoke test for differentiable proximal MRI Scan-TTA."""

import random
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .cg_sense import (
        cg_sense_prox_differentiable,
        measurement_residual_mse,
        measurement_residual_sse,
        sense_forward,
    )
    from .scan_tta import (
        ScanTTAAdapter,
        denoised_from_patches_tta,
        get_tta_indices,
        refinement_diffusion_indices,
    )
except ImportError:
    from cg_sense import (
        cg_sense_prox_differentiable,
        measurement_residual_mse,
        measurement_residual_sse,
        sense_forward,
    )
    from scan_tta import (
        ScanTTAAdapter,
        denoised_from_patches_tta,
        get_tta_indices,
        refinement_diffusion_indices,
    )


class TinyPatchPrior(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            2, 2, kernel_size=1
        )

    def forward(
        self,
        x,
        sigma,
        x_pos=None,
        class_labels=None,
    ):
        del sigma, x_pos, class_labels
        return x + 0.1 * self.conv(x)


def _prediction(
    net,
    x_complex,
    pos,
    indices,
    pad,
):
    x_real = torch.view_as_real(
        x_complex.squeeze(1)
    ).permute(0, 3, 1, 2)
    output = denoised_from_patches_tta(
        net,
        x_real,
        torch.tensor(0.5),
        pos,
        indices,
    )
    cropped = output[
        :, :, pad:-pad, pad:-pad
    ]
    return torch.complex(
        cropped[:, 0],
        cropped[:, 1],
    ).unsqueeze(1)


def main() -> None:
    torch.manual_seed(7)
    random.seed(7)

    maps = torch.ones(
        1, 1, 4, 4,
        dtype=torch.complex64,
    )
    mask = torch.zeros(
        1, 1, 4, 4
    )
    mask[..., ::2, :] = 1

    truth = torch.complex(
        torch.randn(1, 1, 4, 4),
        torch.randn(1, 1, 4, 4),
    )
    measurement = sense_forward(
        truth, maps, mask
    )

    # A: proximal CG improves measurement consistency but remains
    # differentiable with respect to its denoiser input.
    initial = torch.zeros_like(
        truth, requires_grad=True
    )
    before = measurement_residual_sse(
        initial, measurement, maps, mask
    )
    conditional, info = (
        cg_sense_prox_differentiable(
            initial,
            measurement,
            maps,
            mask,
            iterations=5,
            gamma=1.0,
            return_info=True,
        )
    )
    after = measurement_residual_sse(
        conditional,
        measurement,
        maps,
        mask,
    )
    assert after < before
    assert not torch.equal(
        conditional.detach(),
        initial.detach(),
    )

    cond_loss = measurement_residual_mse(
        conditional,
        measurement,
        maps,
        mask,
    )
    input_grad = torch.autograd.grad(
        cond_loss,
        initial,
        retain_graph=False,
    )[0]
    assert torch.isfinite(input_grad).all()
    assert torch.linalg.vector_norm(
        input_grad
    ) > 0

    # B: the adapter's differentiable measurement loss reaches all
    # trainable network parameters and changes the network output.
    net = TinyPatchPrior()
    adapter = ScanTTAAdapter(
        net,
        lr=1e-3,
        cg_gamma=1.0,
    )
    optimizer = (
        adapter.new_subject_optimizer()
    )

    padded = torch.complex(
        torch.randn(1, 1, 8, 8),
        torch.randn(1, 1, 8, 8),
    )
    pos = torch.zeros(
        1, 2, 8, 8
    )
    spaced = np.array([0, 2, 4])
    indices = get_tta_indices(
        spaced,
        3,
        2,
        2,
        random.Random(99),
    )
    prediction_before = _prediction(
        net,
        padded,
        pos,
        indices,
        pad=2,
    ).detach()

    result = adapter.refine(
        fixed_x=padded,
        sigma=torch.tensor(0.5),
        latents_pos=pos,
        measurement=measurement,
        inverseop=SimpleNamespace(
            maps=maps,
            mask=mask,
        ),
        optimizer=optimizer,
        spaced=spaced,
        patches=3,
        pad=2,
        psize=2,
        refinement_iters=1,
        cg_iters=2,
        event=1,
        outer_step=39,
        diffusion_index=40,
        local_seed=12345,
    )

    prediction_after = _prediction(
        net,
        padded,
        pos,
        indices,
        pad=2,
    ).detach()
    assert not torch.equal(
        prediction_before,
        prediction_after,
    )
    assert result.row[
        "tta_objective"
    ] == "differentiable_prox_measurement"

    # C: frozen theta receives no parameter gradients during x-only graph.
    x_only = torch.randn(
        1, 2, 2, 2,
        requires_grad=True,
    )
    x_output = net(
        x_only,
        torch.tensor(0.5),
        torch.zeros_like(x_only),
        None,
    )
    x_grad = torch.autograd.grad(
        x_output.square().sum(),
        x_only,
    )[0]
    assert torch.isfinite(x_grad).all()
    assert all(
        parameter.grad is None
        for parameter in net.parameters()
    )

    # D: schedule remains unchanged; late40 filtering is done by runner/recon.
    event_indices = (
        refinement_diffusion_indices(
            78, 10
        )
    )
    assert event_indices == [
        70, 60, 50, 40, 30, 20, 10
    ]

    # E: local TTA logic does not consume global Python/Torch RNG.
    adapter.reset()
    rng_optimizer = (
        adapter.new_subject_optimizer()
    )
    python_state = random.getstate()
    torch_state = (
        torch.get_rng_state().clone()
    )

    adapter.refine(
        fixed_x=padded,
        sigma=torch.tensor(0.5),
        latents_pos=pos,
        measurement=measurement,
        inverseop=SimpleNamespace(
            maps=maps,
            mask=mask,
        ),
        optimizer=rng_optimizer,
        spaced=spaced,
        patches=3,
        pad=2,
        psize=2,
        refinement_iters=1,
        cg_iters=1,
        event=1,
        outer_step=39,
        diffusion_index=40,
        local_seed=54321,
    )

    assert random.getstate() == python_state
    assert torch.equal(
        torch.get_rng_state(),
        torch_state,
    )

    # F: reset restores exact pretrained state.
    assert not adapter.is_exactly_reset()
    adapter.reset()
    assert adapter.is_exactly_reset()

    print(
        "Prox CG residual:",
        info.measurement_residual_before,
        "->",
        info.measurement_residual_after,
    )
    print(
        "Differentiable CG input gradient: PASS"
    )
    print(
        "Full-network proximal measurement update: PASS"
    )
    print(
        "Frozen-theta x-only separation: PASS"
    )
    print(
        "Event schedule:",
        event_indices,
    )
    print(
        "Global Python/Torch RNG preservation: PASS"
    )
    print(
        "Subject reset: PASS"
    )
    print(
        "Differentiable proximal Scan-TTA smoke test: PASS"
    )


if __name__ == "__main__":
    main()
