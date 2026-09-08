"""Evaluator adapter for the 320-ROI / 384-measurement LGFC experiment."""

import os
import sys
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EVAL_DIR = os.path.join(_REPO_ROOT, "eval")
for _path in (_REPO_ROOT, _EVAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluator import DPSHyperEvaluator
from eval_lgfc.frequency_correction import LGFCConfig
from eval_lgfc_320.recon_lgfc_320 import dps2_lgfc_320


class Lgfc320DPSHyperEvaluator(DPSHyperEvaluator):
    """Keep original 384 samples/operators while using a 320 PaDIS target."""

    def _build_latents_and_pos(self, pad_override: int = None) -> None:
        pad = self.pad if pad_override is None else pad_override
        if self.image_size != 320:
            raise ValueError("Lgfc320DPSHyperEvaluator requires image_size=320")
        if pad != 64:
            raise ValueError("LGFC-320 requires pad=64")

        # Only defines target size / 6x6 patch grid.
        self.latents = torch.randn(
            [1, 1, 320, 320],
            device=self.device,
        )

        # Keep the exact 384-training positional coordinate system.
        ref_resolution = 384 + 2 * 64
        coord = torch.linspace(-1, 1, ref_resolution, device=self.device)
        x_pos = coord.view(1, -1).repeat(ref_resolution, 1)
        y_pos = coord.view(-1, 1).repeat(1, ref_resolution)
        pos_512 = torch.stack([x_pos, y_pos], dim=0).unsqueeze(0)

        # New 448 canvas == central [32:480] crop of original 512 canvas.
        self.latents_pos = pos_512[:, :, 32:480, 32:480].contiguous()

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
            raise ValueError("LGFC-320 is a measured-MRI experiment only")

        config = getattr(self, "lgfc_config", LGFCConfig(enabled=False))
        return dps2_lgfc_320(
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
            device=measurement.device,
        )
