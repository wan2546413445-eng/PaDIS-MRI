"""Minimal evaluator adapter for the LGFC reconstruction entrypoint."""

import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EVAL_DIR = os.path.join(_REPO_ROOT, "eval")
for _path in (_REPO_ROOT, _EVAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluator import DPSHyperEvaluator
from recon import dps_uncond
from eval_lgfc.frequency_correction import LGFCConfig
from eval_lgfc.recon_lgfc import dps2_lgfc


class LgfcDPSHyperEvaluator(DPSHyperEvaluator):
    """Reuse the baseline evaluator while replacing only ``dps2_wrapper``."""

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
            magnitude = torch.abs(recon_cpu.squeeze(0).squeeze(0)).numpy()
            return np.clip(magnitude, 0, 1), 0, 0, 0, 0, 0, 0

        config = getattr(self, "lgfc_config", LGFCConfig(enabled=False))
        return dps2_lgfc(
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
            lgfc_config=config,
            save_lgfc_diagnostics=getattr(self, "save_lgfc_diagnostics", False),
            profile_memory=getattr(self, "profile_memory", False),
            device=measurement.device,
        )
