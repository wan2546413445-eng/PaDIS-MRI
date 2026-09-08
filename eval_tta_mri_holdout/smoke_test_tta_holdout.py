"""Necessary smoke tests for formal k-space holdout Scan-TTA."""

import random
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .cg_sense import (
        heldout_kspace_loss,
        proximal_cg_sense,
        sense_forward_unmasked,
    )
    from .kspace_split import (
        ACS_START,
        ACS_STOP,
        deterministic_refinement_seed,
        split_mask7_columns,
    )
    from .scan_tta_holdout import (
        CG_ITERS,
        HoldoutScanTTAAdapter,
        REFINEMENT_ITERS,
        denoised_from_patches_holdout,
        formal_event_indices,
        get_local_patch_indices,
        local_complex_randn_like,
    )
except ImportError:
    from cg_sense import heldout_kspace_loss, proximal_cg_sense, sense_forward_unmasked
    from kspace_split import (
        ACS_START,
        ACS_STOP,
        deterministic_refinement_seed,
        split_mask7_columns,
    )
    from scan_tta_holdout import (
        CG_ITERS,
        HoldoutScanTTAAdapter,
        REFINEMENT_ITERS,
        denoised_from_patches_holdout,
        formal_event_indices,
        get_local_patch_indices,
        local_complex_randn_like,
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
    outer = list(range(15)) + list(range(369, 384))
    mask[..., :, outer] = 1
    return mask


def patch_prediction(net, x_noisy, positions, indices):
    x_real = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
    d_real = denoised_from_patches_holdout(
        net, x_real, torch.tensor(0.5), positions, indices
    )
    d_crop = d_real[:, :, 64:448, 64:448]
    return torch.complex(d_crop[:, 0], d_crop[:, 1]).unsqueeze(1)


def main() -> None:
    torch.manual_seed(9)
    random.seed(9)
    mask_full = formal_mask7()
    split = split_mask7_columns(mask_full, seed=123)

    # A: exact fixed-column split properties.
    assert torch.count_nonzero(split.mask_cond * split.mask_hold) == 0
    assert torch.equal(split.mask_cond + split.mask_hold, mask_full)
    assert torch.all(split.mask_cond[..., :, ACS_START:ACS_STOP] == 1)
    hold_columns = torch.all(split.mask_hold.bool()[0, 0], dim=0)
    assert int(hold_columns.sum().item()) == 11
    assert int(split.mask_hold.sum().item()) == 11 * 384

    maps = torch.ones(1, 1, 384, 384, dtype=torch.complex64)
    truth = torch.complex(
        torch.randn(1, 1, 384, 384),
        torch.randn(1, 1, 384, 384),
    )
    measurement = mask_full * sense_forward_unmasked(truth, maps)
    prediction = torch.zeros_like(truth)

    # B: changing held-out y while keeping y_cond fixed cannot alter CG output.
    measurement_cond = split.mask_cond * measurement
    z_reference = proximal_cg_sense(
        prediction, measurement_cond, maps, split.mask_cond, iterations=5
    )
    changed_full_measurement = measurement + split.mask_hold * (2 + 3j)
    unchanged_cond = split.mask_cond * changed_full_measurement
    assert torch.equal(unchanged_cond, measurement_cond)
    z_changed_hold = proximal_cg_sense(
        prediction, unchanged_cond, maps, split.mask_cond, iterations=5
    )
    assert torch.equal(z_reference, z_changed_hold)

    # C: L_hold reaches D and network parameters, and one Adam step changes output.
    net = TinyPatchPrior()
    adapter = HoldoutScanTTAAdapter(net)
    optimizer = adapter.new_subject_optimizer()
    x_fixed = torch.complex(
        torch.randn(1, 1, 512, 512),
        torch.randn(1, 1, 512, 512),
    )
    positions = torch.zeros(1, 2, 512, 512)
    spaced = np.linspace(0, 384, 7, dtype=int)
    seed = deterministic_refinement_seed(123, 1, 1, stream=0)
    x_noisy = x_fixed + 0.5 * local_complex_randn_like(x_fixed, seed=seed)
    indices = get_local_patch_indices(
        spaced, 7, 64, 64, random.Random(321)
    )
    adapter.unfreeze()
    optimizer.zero_grad(set_to_none=True)
    output_before = patch_prediction(net, x_noisy, positions, indices)
    output_before.retain_grad()
    z_cond = proximal_cg_sense(
        output_before,
        measurement_cond,
        maps,
        split.mask_cond,
        iterations=5,
    )
    loss = heldout_kspace_loss(z_cond, measurement, maps, split.mask_hold)
    loss.backward()
    assert output_before.grad is not None
    assert torch.count_nonzero(output_before.grad).item() > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in net.parameters()
    )
    output_value_before = output_before.detach().clone()
    optimizer.step()
    output_after = patch_prediction(net, x_noisy, positions, indices).detach()
    assert not torch.equal(output_value_before, output_after)

    # D: a complete five-refinement event preserves global Python/Torch RNG.
    adapter.reset()
    event_optimizer = adapter.new_subject_optimizer()
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    adapter.refine_event(
        fixed_x=x_fixed,
        sigma=torch.tensor(0.5),
        latents_pos=positions,
        measurement=measurement,
        inverseop=SimpleNamespace(mask=mask_full, maps=maps),
        optimizer=event_optimizer,
        spaced=spaced,
        patches=7,
        pad=64,
        psize=64,
        event=1,
        outer_step=39,
        diffusion_index=40,
        subject_seed=123,
    )
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)

    # E: exact reset and formal schedule/budget.
    assert not adapter.is_exactly_reset()
    adapter.reset()
    assert adapter.is_exactly_reset()
    events = formal_event_indices(78)
    assert events == [40, 30, 20, 10]
    assert len(events) * REFINEMENT_ITERS == 20
    assert len(events) * REFINEMENT_ITERS * CG_ITERS == 100

    print("A split: PASS (11/54 columns, center24 conditional)")
    print("B no direct conditional-CG leakage: PASS")
    print("C holdout gradient and optimizer update: PASS")
    print("D global Python/Torch RNG preservation: PASS")
    print("E reset/events/budget: PASS ([40,30,20,10], 20, 100)")
    print("Holdout Scan-TTA smoke test: PASS")


if __name__ == "__main__":
    main()
