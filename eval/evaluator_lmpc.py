"""
LMPC evaluator adapter.

It subclasses the repository's original DPSHyperEvaluator so the existing
sample loading, reconstruction saving, metric collection, plotting, and
post-normalization pipeline remain unchanged.
"""

import numpy as np
import torch

from evaluator import DPSHyperEvaluator
from recon import dps_uncond
from recon_lmpc import dps2_lmpc


class DPSHyperEvaluatorLMPC(DPSHyperEvaluator):
    def __init__(
        self,
        *args,
        late_consensus_steps: int = 0,
        late_consensus_k: int = 1,
        late_consensus_offset_mode: str = "random_complementary",
        save_consensus_diagnostics: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.late_consensus_steps = int(late_consensus_steps)
        self.late_consensus_k = int(late_consensus_k)
        self.late_consensus_offset_mode = late_consensus_offset_mode
        self.save_consensus_diagnostics = bool(
            save_consensus_diagnostics
        )

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
            mag = torch.abs(
                recon_cpu.squeeze(0).squeeze(0)
            ).numpy()
            mag_clip = np.clip(mag, 0, 1)
            return mag_clip, 0, 0, 0, 0, 0, 0

        return dps2_lmpc(
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
            late_consensus_steps=self.late_consensus_steps,
            late_consensus_k=self.late_consensus_k,
            late_consensus_offset_mode=(
                self.late_consensus_offset_mode
            ),
            save_consensus_diagnostics=(
                self.save_consensus_diagnostics
            ),
        )
