"""Single-GPU subject-isolated evaluator for GCC-PaDIS."""

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
    from .gcc_adapter import GCCScanAdapter
    from .recon_gcc import GCC_TTA_EVENTS, reconstruct_gcc
except ImportError:
    from gcc_adapter import GCCScanAdapter
    from recon_gcc import GCC_TTA_EVENTS, reconstruct_gcc


class GCCEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        val_dir: str,
        image_size: int = 384,
        pad: int = 64,
        psize: int = 64,
        mask_select: int = 7,
        val_count: int = 32,
        seed: int = 123,
        sample_indices: Optional[Iterable[int]] = None,
        tta_lr: float = 1e-5,
        gamma: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        if image_size != 384 or pad != 64 or psize != 64 or mask_select != 7:
            raise ValueError(
                "Formal GCC-PaDIS requires image=384, pad=64, patch=64, mask7"
            )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device).eval()
        self.val_dir = Path(val_dir)
        self.image_size = int(image_size)
        self.pad = int(pad)
        self.psize = int(psize)
        self.mask_select = int(mask_select)
        self.seed = int(seed)

        if not self.val_dir.is_dir():
            raise FileNotFoundError(f"val_dir not found: {self.val_dir}")
        available = sorted(
            int(path.stem.split("_")[-1])
            for path in self.val_dir.glob("sample_*.pt")
        )
        if sample_indices:
            chosen = sorted(set(int(index) for index in sample_indices))
            missing = sorted(set(chosen) - set(available))
            if missing:
                raise FileNotFoundError(f"Missing validation samples: {missing}")
            self.val_indices = chosen
        else:
            random.seed(self.seed)
            self.val_indices = sorted(
                random.sample(available, min(int(val_count), len(available)))
            )

        self._reset_rng()
        resolution = self.image_size + 2 * self.pad
        axis = torch.linspace(-1, 1, resolution, device=self.device)
        x_pos = axis.view(1, -1).repeat(resolution, 1)
        y_pos = axis.view(-1, 1).repeat(1, resolution)
        self.latents_pos = torch.stack(
            [x_pos, y_pos], dim=0
        ).unsqueeze(0)
        self._reset_sample_latent()

        self.adapter = GCCScanAdapter(
            self.model, lr=tta_lr, gamma=gamma
        )

    def _reset_rng(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _reset_sample_latent(self) -> None:
        self.latents = torch.randn(
            [1, 1, self.image_size, self.image_size],
            device=self.device,
        )

    def _load_measurement(self, index: int):
        data = torch.load(
            self.val_dir / f"sample_{index}.pt",
            map_location=self.device,
            weights_only=False,
        )
        maps = fftmod(data["s_map"])[None, ...].to(self.device)
        full_kspace = fftmod(data["ksp"])[None, ...].to(self.device)
        mask = data[f"mask_{self.mask_select}"][None, ...].to(self.device)
        measurement = mask * full_kspace
        return measurement, MRI_utils(mask=mask, maps=maps)

    @staticmethod
    def _write_rows(path: Path, rows) -> None:
        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def evaluate(
        self,
        save_dir: str,
        *,
        num_steps: int = 78,
        inner_loops: int = 10,
        refinement_iters: int = 5,
        cg_iters: int = 5,
        gamma: float = 1.0,
        fixed_seed_per_sample: bool = False,
    ) -> Dict:
        root = Path(save_dir)
        recons_dir = root / "recons"
        diagnostics_dir = root / "tta_diagnostics"
        recons_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        if fixed_seed_per_sample and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        runtime_rows = []
        for order, index in enumerate(self.val_indices, start=1):
            if fixed_seed_per_sample:
                self._reset_rng()
                self._reset_sample_latent()

            self.adapter.reset()
            optimizer = self.adapter.new_subject_optimizer()
            measurement, inverseop = self._load_measurement(index)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()

            try:
                reconstruction, diagnostics, counters = reconstruct_gcc(
                    net=self.model,
                    latents=self.latents,
                    latents_pos=self.latents_pos,
                    inverseop=inverseop,
                    measurement=measurement,
                    adapter=self.adapter,
                    optimizer=optimizer,
                    num_steps=num_steps,
                    inner_loops=inner_loops,
                    pad=self.pad,
                    psize=self.psize,
                    tta_events=GCC_TTA_EVENTS,
                    refinement_iters=refinement_iters,
                    cg_iters=cg_iters,
                    gamma=gamma,
                    subject_seed=self.seed,
                    device=str(self.device),
                )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                runtime_seconds = time.perf_counter() - start
                peak_bytes = (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else 0
                )

                np.save(
                    recons_dir / f"recon_patch_{index}.npy",
                    reconstruction.cpu().numpy(),
                )
                self._write_rows(
                    diagnostics_dir / f"sample_{index}.csv", diagnostics
                )
                runtime_rows.append({
                    "sample": int(index),
                    "runtime_seconds": float(runtime_seconds),
                    "peak_gpu_memory_bytes": int(peak_bytes),
                    "denoiser_calls": counters["denoiser_calls"],
                    "tta_backprops": counters["tta_backprops"],
                    "full_cg_calls": counters["full_cg_calls"],
                    "total_cg_iterations": counters["total_cg_iterations"],
                })
            finally:
                del optimizer
                self.adapter.reset()

            print(
                f"[GCC-PaDIS] sample {order}/{len(self.val_indices)} "
                f"idx={index} complete"
            )

        self._write_rows(root / "sample_runtime.csv", runtime_rows)
        with (root / "gcc_config.json").open("w") as handle:
            json.dump({
                "method": "GCC-PaDIS",
                "tta_objective": "kspace_holdout_cross_mask",
                "tta_scope": "full_network",
                "optimizer": "Adam",
                "tta_lr": self.adapter.lr,
                "weight_decay": 0.0,
                "tta_events": list(GCC_TTA_EVENTS),
                "refinement_iters": int(refinement_iters),
                "holdout_columns": 11,
                "acquired_columns": 54,
                "acs_columns": 24,
                "conditional_estimator": "proximal_cg",
                "gamma": float(gamma),
                "cg_iters": int(cg_iters),
                "num_steps": int(num_steps),
                "inner_loops": int(inner_loops),
                "full_network_parameters": self.adapter.parameter_count,
            }, handle, indent=2)

        metrics = post_eval_normalize(
            recon_dir=str(recons_dir),
            val_dir=str(self.val_dir),
            plot_dir=str(root / "comp_plots"),
            json_basename="results_gcc_padis",
            mask_select=self.mask_select,
        )
        return {"metrics": metrics, "runtime": runtime_rows}
