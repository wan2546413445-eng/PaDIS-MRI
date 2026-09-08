"""Full-network cross-mask Scan-TTA for one formal PaDIS-MRI scan."""

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

try:
    from .cg_sense import heldout_kspace_loss, proximal_cg_sense
    from .kspace_split import (
        EXPECTED_HOLDOUT_LINES,
        deterministic_refinement_seed,
        split_mask7_columns,
    )
except ImportError:
    from cg_sense import heldout_kspace_loss, proximal_cg_sense
    from kspace_split import (
        EXPECTED_HOLDOUT_LINES,
        deterministic_refinement_seed,
        split_mask7_columns,
    )


TTA_INTERVAL = 10
TTA_MAX_DIFFUSION = 40
REFINEMENT_ITERS = 5
CG_ITERS = 5
CG_GAMMA = 1.0
TTA_LR = 1e-5


def formal_event_indices(num_steps: int = 78) -> List[int]:
    return [
        index
        for index in range(num_steps, 0, -1)
        if index <= TTA_MAX_DIFFUSION and index % TTA_INTERVAL == 0
    ]


def get_local_patch_indices(
    spaced: Sequence[int],
    patches: int,
    pad: int,
    psize: int,
    rng: random.Random,
) -> List[List[int]]:
    """Generate a patch partition using only a caller-owned Python RNG."""
    row_offset = rng.randint(0, pad - 1) if pad > 0 else 0
    column_offset = rng.randint(0, pad - 1) if pad > 0 else 0
    return [
        [
            int(spaced[row] + row_offset),
            int(spaced[row] + row_offset + psize),
            int(spaced[column] + column_offset),
            int(spaced[column] + column_offset + psize),
        ]
        for row in range(patches)
        for column in range(patches)
    ]


def local_complex_randn_like(reference: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Exactly match complex ``randn_like`` semantics without global RNG use."""
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def denoised_from_patches_holdout(
    net: torch.nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    latents_pos: torch.Tensor,
    indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Use the baseline patch path without its zero-scaled global RNG draw."""
    x_hat = x.clone()
    _, channels, size, _ = x_hat.shape
    patch_count = len(indices)
    grid_width = math.isqrt(patch_count)
    psize = int(indices[0][1] - indices[0][0])
    pad = int(size - grid_width * psize)

    x_input = torch.zeros(
        patch_count, channels, psize, psize,
        device=x.device, dtype=x.dtype,
    )
    pos_input = torch.zeros(
        patch_count, 2, psize, psize,
        device=latents_pos.device, dtype=latents_pos.dtype,
    )
    for patch_index, coordinates in enumerate(indices):
        r0, r1, c0, c1 = coordinates
        x_input[patch_index] = x_hat[0, :, r0:r1, c0:c1]
        pos_input[patch_index] = latents_pos[0, :, r0:r1, c0:c1]

    patch_batch = int(os.environ.get("PADIS_PATCH_BATCH", patch_count))
    outputs = []
    for start in range(0, patch_count, patch_batch):
        stop = min(start + patch_batch, patch_count)
        outputs.append(
            net(x_input[start:stop], sigma, pos_input[start:stop], None).float()
        )
    denoised_patches = torch.cat(outputs, dim=0)

    residual = torch.zeros_like(x_hat)
    for patch_index, coordinates in enumerate(indices):
        r0, r1, c0, c1 = coordinates
        residual[0, :, r0:r1, c0:c1] += (
            denoised_patches[patch_index] - x_hat[0, :, r0:r1, c0:c1]
        )
    denoised = x_hat + residual
    result = torch.zeros_like(denoised)
    result[:, :, pad:size - pad, pad:size - pad] = denoised[
        :, :, pad:size - pad, pad:size - pad
    ]
    return result


@dataclass
class RefinementSummary:
    rows: List[Dict]
    denoiser_calls: int
    backprops: int
    cg_iterations: int


class HoldoutScanTTAAdapter:
    """CPU theta0 plus subject-specific full-network Adam state."""

    def __init__(self, net: torch.nn.Module) -> None:
        self.net = net
        self.theta0 = {
            name: tensor.detach().cpu().clone()
            for name, tensor in net.state_dict().items()
        }
        self.parameter_count = sum(parameter.numel() for parameter in net.parameters())
        self.freeze()

    def freeze(self) -> None:
        self.net.requires_grad_(False)
        for parameter in self.net.parameters():
            parameter.grad = None

    def unfreeze(self) -> None:
        self.net.requires_grad_(True)

    def reset(self) -> None:
        self.net.load_state_dict(self.theta0, strict=True)
        self.freeze()

    def is_exactly_reset(self) -> bool:
        return all(
            torch.equal(value.detach().cpu(), self.theta0[name])
            for name, value in self.net.state_dict().items()
        )

    def new_subject_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.net.parameters(), lr=TTA_LR, weight_decay=0.0
        )

    def refine_event(
        self,
        *,
        fixed_x: torch.Tensor,
        sigma: torch.Tensor,
        latents_pos: torch.Tensor,
        measurement: torch.Tensor,
        inverseop,
        optimizer: torch.optim.Adam,
        spaced: Sequence[int],
        patches: int,
        pad: int,
        psize: int,
        event: int,
        outer_step: int,
        diffusion_index: int,
        subject_seed: int,
    ) -> RefinementSummary:
        x_fixed = fixed_x.detach()
        image_size = x_fixed.shape[-1] - 2 * pad
        rows: List[Dict] = []
        self.unfreeze()
        try:
            for refinement in range(1, REFINEMENT_ITERS + 1):
                optimizer.zero_grad(set_to_none=True)
                noise_seed = deterministic_refinement_seed(
                    subject_seed, event, refinement, stream=0
                )
                patch_seed = deterministic_refinement_seed(
                    subject_seed, event, refinement, stream=1
                )
                split_seed = deterministic_refinement_seed(
                    subject_seed, event, refinement, stream=2
                )

                noise = local_complex_randn_like(x_fixed, seed=noise_seed)
                x_noisy = x_fixed + sigma * noise
                x_real = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
                indices = get_local_patch_indices(
                    spaced, patches, pad, psize, random.Random(patch_seed)
                )
                d_real = denoised_from_patches_holdout(
                    self.net, x_real, sigma, latents_pos, indices
                )
                d_crop = d_real[:, :, pad:pad + image_size, pad:pad + image_size]
                prediction = torch.complex(d_crop[:, 0], d_crop[:, 1]).unsqueeze(1)

                split = split_mask7_columns(inverseop.mask, seed=split_seed)
                measurement_cond = split.mask_cond * measurement
                z_cond = proximal_cg_sense(
                    prediction,
                    measurement_cond,
                    inverseop.maps,
                    split.mask_cond,
                    gamma=CG_GAMMA,
                    iterations=CG_ITERS,
                )
                loss = heldout_kspace_loss(
                    z_cond,
                    measurement,
                    inverseop.maps,
                    split.mask_hold,
                )
                loss_value = float(loss.detach().item())
                loss.backward()
                optimizer.step()

                rows.append({
                    "event": int(event),
                    "refinement": int(refinement),
                    "outer_step": int(outer_step),
                    "diffusion_index": int(diffusion_index),
                    "sigma": float(sigma.detach().item()),
                    "holdout_loss": loss_value,
                    "num_holdout_lines": int(len(split.holdout_columns)),
                })
                del (
                    noise,
                    x_noisy,
                    x_real,
                    d_real,
                    d_crop,
                    prediction,
                    measurement_cond,
                    z_cond,
                    loss,
                )
        finally:
            optimizer.zero_grad(set_to_none=True)
            self.freeze()

        return RefinementSummary(
            rows=rows,
            denoiser_calls=REFINEMENT_ITERS,
            backprops=REFINEMENT_ITERS,
            cg_iterations=REFINEMENT_ITERS * CG_ITERS,
        )


assert EXPECTED_HOLDOUT_LINES == 11
