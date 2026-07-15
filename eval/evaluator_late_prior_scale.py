"""Evaluator adapter for PaDIS-MRI late-stage prior scaling."""

import numpy as np
import torch

from evaluator import DPSHyperEvaluator
from recon import dps_uncond
from recon_late_prior_scale import dps2_late_prior_scale


class DPSHyperEvaluatorLatePriorScale(DPSHyperEvaluator):
    def __init__(
        self,
        *args,
        late_prior_steps: int = 0,
        late_prior_scale: float = 1.0,
        save_scale_diagnostics: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.late_prior_steps = int(late_prior_steps)
        self.late_prior_scale = float(late_prior_scale)
        self.save_scale_diagnostics = bool(save_scale_diagnostics)

    def dps2_wrapper(
        self,
        inverse_op,
        measurement,
        clean,
        zeta,
        pad,
        psize,
        num_steps,
        save_dir=None,
        tag=None,
        save_intermediate: bool = False,
        intermediate_every: int = 10,
        inner_loops: int = 10,
    ):
        if measurement is None:
            recon = dps_uncond(
                net=self.model,
                batch_size=1,
                resolution=self.image_size,
                psize=psize,
                pad=pad,
                num_steps=num_steps,
                sigma_min=0.003,
                sigma_max=10.0,
                rho=7,
                device=self.device,
                randn_like=torch.randn_like,
            )
            recon_cpu = recon.cpu()
            mag = torch.abs(recon_cpu.squeeze(0).squeeze(0)).numpy()
            return np.clip(mag, 0, 1), 0, 0, 0, 0, 0, 0

        return dps2_late_prior_scale(
            net=self.model,
            latents=self.latents,
            latents_pos=self.latents_pos,
            inverseop=inverse_op,
            measurement=measurement,
            clean=clean,
            pad=pad,
            psize=psize,
            zeta=zeta,
            num_steps=num_steps,
            inner_loops=inner_loops,
            save_dir=save_dir,
            tag=tag,
            save_intermediate=save_intermediate,
            intermediate_every=intermediate_every,
            late_prior_steps=self.late_prior_steps,
            late_prior_scale=self.late_prior_scale,
            save_scale_diagnostics=self.save_scale_diagnostics,
        )
