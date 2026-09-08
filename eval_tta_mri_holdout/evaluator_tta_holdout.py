"""Subject-isolated evaluator for formal k-space holdout Scan-TTA."""

import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from inverse_operators import MRI_utils
from utils import fftmod, post_eval_normalize

try:
    from .recon_tta_holdout import dps2_tta_holdout
    from .scan_tta_holdout import (
        CG_GAMMA,
        CG_ITERS,
        HoldoutScanTTAAdapter,
        REFINEMENT_ITERS,
        TTA_INTERVAL,
        TTA_LR,
        TTA_MAX_DIFFUSION,
        formal_event_indices,
    )
except ImportError:
    from recon_tta_holdout import dps2_tta_holdout
    from scan_tta_holdout import (
        CG_GAMMA,
        CG_ITERS,
        HoldoutScanTTAAdapter,
        REFINEMENT_ITERS,
        TTA_INTERVAL,
        TTA_LR,
        TTA_MAX_DIFFUSION,
        formal_event_indices,
    )


class HoldoutScanTTAEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        val_dir: str,
        sample_indices: Iterable[int],
        device: torch.device,
        seed: int = 123,
    ) -> None:
        self.device = device
        self.model = model.to(device).eval()
        self.val_dir = Path(val_dir)
        self.val_indices = sorted(set(int(index) for index in sample_indices))
        self.seed = int(seed)
        self.image_size = 384
        self.pad = 64
        self.psize = 64
        self.mask_select = 7
        if not self.val_dir.is_dir():
            raise FileNotFoundError(f"val_dir not found: {self.val_dir}")
        missing = [
            index for index in self.val_indices
            if not (self.val_dir / f"sample_{index}.pt").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing validation samples: {missing}")

        self._reset_rng()
        resolution = self.image_size + 2 * self.pad
        axis = torch.linspace(-1, 1, resolution, device=device)
        x_pos = axis.view(1, -1).repeat(resolution, 1)
        y_pos = axis.view(-1, 1).repeat(1, resolution)
        self.latents_pos = torch.stack([x_pos, y_pos], dim=0).unsqueeze(0)
        self._reset_sample_latent()
        self.adapter = HoldoutScanTTAAdapter(self.model)

    def _reset_rng(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _reset_sample_latent(self) -> None:
        self.latents = torch.randn(
            [1, 1, self.image_size, self.image_size], device=self.device
        )

    def _load_measurement(self, index: int):
        data = torch.load(
            self.val_dir / f"sample_{index}.pt",
            map_location=self.device,
            weights_only=False,
        )
        maps = fftmod(data["s_map"])[None, ...].to(self.device)
        full_kspace = fftmod(data["ksp"])[None, ...].to(self.device)
        mask = data["mask_7"][None, ...].to(self.device)
        measurement = mask * full_kspace
        return measurement, MRI_utils(mask=mask, maps=maps)

    @staticmethod
    def _write_rows(path: Path, rows) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def evaluate(self, save_dir: str) -> Dict:
        root = Path(save_dir)
        recons_dir = root / "recons"
        diagnostics_dir = root / "tta_diagnostics"
        recons_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        runtime_rows = []
        for order, index in enumerate(self.val_indices, start=1):
            self._reset_rng()
            self._reset_sample_latent()
            optimizer = self.adapter.new_subject_optimizer()
            measurement, inverseop = self._load_measurement(index)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            try:
                reconstruction, diagnostics, counters = dps2_tta_holdout(
                    net=self.model,
                    latents=self.latents,
                    latents_pos=self.latents_pos,
                    inverseop=inverseop,
                    measurement=measurement,
                    adapter=self.adapter,
                    optimizer=optimizer,
                    num_steps=78,
                    inner_loops=10,
                    zeta=3.0,
                    pad=64,
                    psize=64,
                    subject_seed=self.seed,
                    device=str(self.device),
                )
                torch.cuda.synchronize(self.device)
                runtime_seconds = time.perf_counter() - start
                peak_bytes = int(torch.cuda.max_memory_allocated(self.device))
                np.save(
                    recons_dir / f"recon_patch_{index}.npy",
                    reconstruction.cpu().numpy(),
                )
                self._write_rows(
                    diagnostics_dir / f"sample_{index}.csv", diagnostics
                )
                runtime_rows.append({
                    "sample": int(index),
                    "baseline_denoiser_calls": counters["baseline_denoiser_calls"],
                    "tta_denoiser_calls": counters["tta_denoiser_calls"],
                    "tta_backprops": counters["tta_backprops"],
                    "cg_iterations_total": counters["cg_iterations_total"],
                    "runtime_seconds": float(runtime_seconds),
                    "peak_gpu_memory_bytes": peak_bytes,
                })
            finally:
                del optimizer
                self.adapter.reset()

            print(
                f"[Holdout Scan-TTA] sample {order}/{len(self.val_indices)} "
                f"idx={index} complete"
            )

        self._write_rows(root / "sample_runtime.csv", runtime_rows)
        with (root / "tta_holdout_config.json").open("w") as handle:
            json.dump({
                "scope": "full_network",
                "optimizer": "adam",
                "lr": TTA_LR,
                "weight_decay": 0.0,
                "gamma": CG_GAMMA,
                "cg_iters": CG_ITERS,
                "refinement_iters": REFINEMENT_ITERS,
                "tta_interval": TTA_INTERVAL,
                "tta_max_diffusion": TTA_MAX_DIFFUSION,
                "event_diffusion_indices": formal_event_indices(78),
                "holdout_fraction": 0.20,
                "holdout_lines": 11,
                "full_network_parameters": self.adapter.parameter_count,
            }, handle, indent=2)

        metrics = post_eval_normalize(
            recon_dir=str(recons_dir),
            val_dir=str(self.val_dir),
            plot_dir=str(root / "comp_plots"),
            json_basename="results_tta_mri_holdout",
            mask_select=7,
        )
        return {"metrics": metrics, "runtime": runtime_rows}
