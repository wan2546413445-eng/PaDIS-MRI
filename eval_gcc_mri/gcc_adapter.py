"""Holdout cross-mask full-network adaptation for GCC-PaDIS."""

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from eval_tta_mri_holdout.scan_tta_holdout import (
    denoised_from_patches_holdout,
    get_local_patch_indices,
    local_complex_randn_like,
)

try:
    from .cg_sense import acquired_measurement_mse, cg_data_consistency
    from .kspace_split import (
        deterministic_refinement_seed,
        split_mask7_columns,
    )
except ImportError:
    from cg_sense import acquired_measurement_mse, cg_data_consistency
    from kspace_split import (
        deterministic_refinement_seed,
        split_mask7_columns,
    )


@dataclass
class AdaptationResult:
    rows: List[Dict[str, object]]
    denoiser_calls: int
    backprops: int
    cg_calls: int
    cg_iterations: int


class GCCScanAdapter:
    """Own the pretrained state and one subject-specific Adam trajectory."""

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 1e-5,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.net = net
        self.lr = float(lr)
        self.theta0 = {
            name: value.detach().cpu().clone()
            for name, value in net.state_dict().items()
        }
        self.parameter_count = sum(
            parameter.numel() for parameter in net.parameters()
        )
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
            self.net.parameters(),
            lr=self.lr,
            weight_decay=0.0,
        )

    def refine(
        self,
        *,
        current_x: torch.Tensor,
        sigma: torch.Tensor,
        latents_pos: torch.Tensor,
        measurement: torch.Tensor,
        maps: torch.Tensor,
        full_mask: torch.Tensor,
        optimizer: torch.optim.Adam,
        spaced: Sequence[int],
        patches: int,
        pad: int,
        psize: int,
        refinement_iters: int,
        cg_iters: int,
        event: int,
        subject_seed: int,
    ) -> AdaptationResult:
        """Cross-mask TTA with hard conditional CG on Omega_cond."""
        if refinement_iters < 1 or cg_iters < 1:
            raise ValueError("refinement_iters and cg_iters must be positive")

        x_fixed = current_x.detach()
        image_size = x_fixed.shape[-1] - 2 * pad
        if image_size != 384:
            raise ValueError("Formal GCC-PaDIS TTA requires image_size=384")

        rows: List[Dict[str, object]] = []
        self.unfreeze()
        try:
            for refinement in range(1, refinement_iters + 1):
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

                noise = local_complex_randn_like(
                    x_fixed, seed=noise_seed
                )
                x_tta_noisy = x_fixed + sigma * noise
                x_real = torch.view_as_real(
                    x_tta_noisy.squeeze(1)
                ).permute(0, 3, 1, 2)

                indices = get_local_patch_indices(
                    spaced,
                    patches,
                    pad,
                    psize,
                    random.Random(patch_seed),
                )
                denoised_real = denoised_from_patches_holdout(
                    self.net, x_real, sigma, latents_pos, indices
                )
                denoised_crop = denoised_real[
                    :, :, pad:pad + image_size, pad:pad + image_size
                ]
                prediction = torch.complex(
                    denoised_crop[:, 0], denoised_crop[:, 1]
                ).unsqueeze(1)

                split = split_mask7_columns(
                    full_mask, seed=split_seed
                )
                measurement_cond = split.mask_cond * measurement

                conditional, _ = cg_data_consistency(
                    prediction,
                    measurement_cond,
                    maps,
                    split.mask_cond,
                    iterations=cg_iters,
                    differentiable=True,
                )

                holdout_loss = acquired_measurement_mse(
                    conditional,
                    measurement,
                    maps.detach(),
                    split.mask_hold.detach(),
                )
                holdout_loss.backward()
                optimizer.step()

                rows.append({
                    "event": int(event),
                    "refinement": int(refinement),
                    "sigma": float(sigma.detach().item()),
                    "holdout_loss": float(holdout_loss.detach().item()),
                })
        finally:
            optimizer.zero_grad(set_to_none=True)
            self.freeze()

        return AdaptationResult(
            rows=rows,
            denoiser_calls=int(refinement_iters),
            backprops=int(refinement_iters),
            cg_calls=int(refinement_iters),
            cg_iterations=int(refinement_iters * cg_iters),
        )
