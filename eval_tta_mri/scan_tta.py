"""Full-network differentiable proximal self-supervision for one MRI scan."""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

try:
    from .cg_sense import (
        cg_sense_prox_differentiable,
        measurement_residual_mse,
    )
except ImportError:
    from cg_sense import (
        cg_sense_prox_differentiable,
        measurement_residual_mse,
    )


def refinement_diffusion_indices(num_steps: int, interval: int) -> List[int]:
    """Return descending 1-based diffusion indices selected for TTA."""
    if num_steps < 1 or interval < 1:
        raise ValueError("num_steps and interval must be positive")
    return [
        index
        for index in range(num_steps, 0, -1)
        if index % interval == 0
    ]


def get_tta_indices(
    spaced: Sequence[int],
    patches: int,
    pad: int,
    psize: int,
    rng: random.Random,
) -> List[List[int]]:
    """Generate a patch partition without touching Python's global RNG."""
    a = rng.randint(0, pad - 1) if pad > 0 else 0
    b = rng.randint(0, pad - 1) if pad > 0 else 0
    return [
        [
            int(spaced[p] + a),
            int(spaced[p] + a + psize),
            int(spaced[q] + b),
            int(spaced[q] + b + psize),
        ]
        for p in range(patches)
        for q in range(patches)
    ]


def denoised_from_patches_tta(
    net: torch.nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    latents_pos: torch.Tensor,
    indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Baseline-equivalent patch denoising without its zero-scaled RNG draw."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(
            f"Expected TTA input [1,C,H,W], got {tuple(x.shape)}"
        )
    if not indices:
        raise ValueError("indices must not be empty")

    x_hat = x.clone()
    _, channels, size, _ = x_hat.shape
    psize = int(indices[0][1] - indices[0][0])
    patch_count = len(indices)
    grid_width = int(round(np.sqrt(patch_count)))
    if grid_width * grid_width != patch_count:
        raise ValueError("Patch partition must be a square grid")
    pad = int(size - grid_width * psize)

    x_input = torch.zeros(
        patch_count,
        channels,
        psize,
        psize,
        device=x.device,
        dtype=x.dtype,
    )
    pos_input = torch.zeros(
        patch_count,
        2,
        psize,
        psize,
        device=latents_pos.device,
        dtype=latents_pos.dtype,
    )
    for patch_index, coords in enumerate(indices):
        r0, r1, c0, c1 = coords
        x_input[patch_index] = x_hat[0, :, r0:r1, c0:c1]
        pos_input[patch_index] = latents_pos[0, :, r0:r1, c0:c1]

    patch_batch = int(
        os.environ.get("PADIS_PATCH_BATCH", patch_count)
    )
    if patch_batch < 1:
        raise ValueError("PADIS_PATCH_BATCH must be positive")

    outputs = []
    for start in range(0, patch_count, patch_batch):
        end = min(start + patch_batch, patch_count)
        outputs.append(
            net(
                x_input[start:end],
                sigma,
                pos_input[start:end],
                None,
            ).float()
        )
    bigout = torch.cat(outputs, dim=0)

    residual = torch.zeros_like(x_hat)
    for patch_index, coords in enumerate(indices):
        r0, r1, c0, c1 = coords
        residual[0, :, r0:r1, c0:c1] += (
            bigout[patch_index]
            - x_hat[0, :, r0:r1, c0:c1]
        )

    denoised = x_hat + residual
    result = torch.zeros_like(denoised)
    result[:, :, pad:size - pad, pad:size - pad] = denoised[
        :, :, pad:size - pad, pad:size - pad
    ]
    return result


def complex_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Historical helper retained for compatibility with old diagnostics."""
    return torch.mean(torch.abs(prediction - target).square())


@dataclass
class RefinementResult:
    row: Dict[str, object]
    denoiser_calls: int
    backprops: int
    cg_iterations: int


class ScanTTAAdapter:
    """Own pretrained CPU state and subject-specific full-network Adam state."""

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 1e-5,
        cg_gamma: float = 1.0,
    ) -> None:
        if cg_gamma <= 0:
            raise ValueError("cg_gamma must be positive")
        self.net = net
        self.lr = float(lr)
        self.cg_gamma = float(cg_gamma)
        self.theta0 = {
            name: tensor.detach().cpu().clone()
            for name, tensor in net.state_dict().items()
        }
        self.parameter_count = sum(
            parameter.numel()
            for parameter in net.parameters()
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
            torch.equal(
                tensor.detach().cpu(),
                self.theta0[name],
            )
            for name, tensor in self.net.state_dict().items()
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
        refinement_iters: int,
        cg_iters: int,
        event: int,
        outer_step: int,
        diffusion_index: int,
        local_seed: int,
    ) -> RefinementResult:
        """Refine theta through a differentiable conditional CG denoiser.

        For each refinement iteration:

          D_theta(x_t)
              -> differentiable proximal CG
              -> D_theta(x_t | y)
              -> ||A D_theta(x_t | y) - y||^2
              -> backpropagate through CG into theta

        The normal PaDIS reconstruction loop remains unchanged.
        """
        if refinement_iters < 1:
            raise ValueError("refinement_iters must be positive")

        x_t = fixed_x.detach()
        x_real = torch.view_as_real(
            x_t.squeeze(1)
        ).permute(0, 3, 1, 2)
        local_rng = random.Random(int(local_seed))
        image_size = x_t.shape[-1] - 2 * pad

        self.unfreeze()

        first_loss = None
        last_loss = None
        cg_before = None
        cg_after = None
        prior_mse_first = None
        prior_mse_last = None

        try:
            for refine_index in range(refinement_iters):
                optimizer.zero_grad(set_to_none=True)

                indices = get_tta_indices(
                    spaced,
                    patches,
                    pad,
                    psize,
                    local_rng,
                )
                d_real = denoised_from_patches_tta(
                    self.net,
                    x_real,
                    sigma,
                    latents_pos,
                    indices,
                )

                if pad == 0:
                    d_crop = d_real
                else:
                    d_crop = d_real[
                        :,
                        :,
                        pad:pad + image_size,
                        pad:pad + image_size,
                    ]

                prediction = torch.complex(
                    d_crop[:, 0],
                    d_crop[:, 1],
                ).unsqueeze(1)

                want_info = (
                    refine_index == 0
                    or refine_index == refinement_iters - 1
                )

                # Do NOT detach prediction. The finite CG graph is part of the
                # TTA computation graph, so the measurement loss reaches theta.
                conditional, cg_info = (
                    cg_sense_prox_differentiable(
                        prediction,
                        measurement,
                        inverseop.maps,
                        inverseop.mask,
                        iterations=cg_iters,
                        gamma=self.cg_gamma,
                        return_info=want_info,
                    )
                )

                loss = measurement_residual_mse(
                    conditional,
                    measurement,
                    inverseop.maps.detach(),
                    inverseop.mask.detach(),
                )

                if refine_index == 0:
                    first_loss = float(loss.detach().item())
                    cg_before = (
                        cg_info.measurement_residual_before
                    )
                    prior_mse_first = cg_info.prior_mse_after

                if refine_index == refinement_iters - 1:
                    last_loss = float(loss.detach().item())
                    cg_after = (
                        cg_info.measurement_residual_after
                    )
                    prior_mse_last = cg_info.prior_mse_after

                loss.backward()
                optimizer.step()

        finally:
            optimizer.zero_grad(set_to_none=True)
            self.freeze()

        row = {
            "event": int(event),
            "outer_step": int(outer_step),
            "diffusion_index": int(diffusion_index),
            "sigma": float(sigma.detach().item()),
            "tta_objective": "differentiable_prox_measurement",
            "cg_gamma": float(self.cg_gamma),
            "first_tta_loss": float(first_loss),
            "last_tta_loss": float(last_loss),
            "cg_residual_before": float(cg_before),
            "cg_residual_after": float(cg_after),
            "conditional_prior_mse_first": float(prior_mse_first),
            "conditional_prior_mse_last": float(prior_mse_last),
        }

        return RefinementResult(
            row=row,
            denoiser_calls=int(refinement_iters),
            backprops=int(refinement_iters),
            cg_iterations=int(
                refinement_iters * cg_iters
            ),
        )
