"""Single-GPU, subject-isolated evaluator for MCPP Stage-1."""

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
    from .prior_calibration import MCPPHeadAdapter
    from .recon_mcpp import dps2_mcpp
except ImportError:
    from prior_calibration import MCPPHeadAdapter
    from recon_mcpp import dps2_mcpp


class MCPPEvaluator:
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
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            count = min(int(val_count), len(available))
            self.val_indices = sorted(random.sample(available, count))

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        resolution = self.image_size + 2 * self.pad
        coord_x = torch.linspace(-1, 1, resolution, device=self.device)
        coord_y = torch.linspace(-1, 1, resolution, device=self.device)
        x_pos = coord_x.view(1, -1).repeat(resolution, 1)
        y_pos = coord_y.view(-1, 1).repeat(1, resolution)
        self.latents_pos = torch.stack([x_pos, y_pos], dim=0).unsqueeze(0)
        self.latents = torch.randn(
            [1, 1, self.image_size, self.image_size], device=self.device
        )

        self.adapter = MCPPHeadAdapter(self.model)

    def _reset_rng(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _reset_sample_latent(self) -> None:
        """Match the existing fixed-per-sample evaluator RNG stream exactly.

        ``dps2`` only uses the latent shape, but the project baseline regenerates this
        tensor after reseeding. Consuming the same draw keeps subsequent patch/noise
        randomness paired with the formal baseline.
        """
        self.latents = torch.randn(
            [1, 1, self.image_size, self.image_size],
            device=self.device,
        )

    def _load_measurement(self, idx: int):
        data = torch.load(
            self.val_dir / f"sample_{idx}.pt",
            map_location=self.device,
            weights_only=False,
        )
        maps = fftmod(data["s_map"])[None, ...].to(self.device)
        # 数据文件保存 fully sampled k-space，但 MCPP 只把 mask 后的 measurement 交给重建/适配。
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
        zeta: float = 3.0,
        num_steps: int = 78,
        inner_loops: int = 10,
        mcpp_lr: float = 1e-4,
        save_mcpp_diagnostics: bool = False,
        fixed_seed_per_sample: bool = False,
    ) -> Dict:
        root = Path(save_dir)
        recons_dir = root / "recons"
        diagnostics_dir = root / "mcpp_diagnostics"
        plots_dir = root / "comp_plots"
        recons_dir.mkdir(parents=True, exist_ok=True)
        if save_mcpp_diagnostics:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

        if fixed_seed_per_sample and torch.cuda.is_available():
            # 与项目现有 fixed-per-sample evaluator 的复现实验设置保持一致。
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        sample_rows = []
        for order, idx in enumerate(self.val_indices, start=1):
            if fixed_seed_per_sample:
                self._reset_rng()
                # Formal baseline also regenerates x_T here; keep the RNG stream paired.
                self._reset_sample_latent()

            self.adapter.reset()
            if not self.adapter.is_exactly_reset():
                raise RuntimeError("Failed to reset phi to pretrained phi0 before subject reconstruction")

            measurement, inverseop = self._load_measurement(idx)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)

            start = time.perf_counter()
            try:
                recon, diagnostics, metadata = dps2_mcpp(
                    net=self.model,
                    latents=self.latents,
                    latents_pos=self.latents_pos,
                    inverseop=inverseop,
                    measurement=measurement,
                    num_steps=num_steps,
                    inner_loops=inner_loops,
                    zeta=zeta,
                    pad=self.pad,
                    psize=self.psize,
                    mcpp_lr=mcpp_lr,
                    device=str(self.device),
                    adapter=self.adapter,
                    collect_diagnostics=save_mcpp_diagnostics,
                )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                runtime = time.perf_counter() - start
                peak_vram = (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda" else 0
                )

                np.save(recons_dir / f"recon_patch_{idx}.npy", recon.cpu().numpy())
                if save_mcpp_diagnostics:
                    self._write_rows(diagnostics_dir / f"sample_{idx}.csv", diagnostics)

                sample_rows.append({
                    "sample": idx,
                    "runtime_seconds": runtime,
                    "peak_gpu_memory_bytes": peak_vram,
                    "updates": metadata["updates"],
                    "denoiser_calls": metadata["denoiser_calls"],
                })
            finally:
                # 每个 subject 的在线 prior 只属于该 subject，下一样本必须从 phi0 重启。
                self.adapter.reset()
                if not self.adapter.is_exactly_reset():
                    raise RuntimeError("Failed to reset phi after subject reconstruction")

            print(f"[MCPP] sample {order}/{len(self.val_indices)} idx={idx} complete")

        self._write_rows(root / "sample_runtime.csv", sample_rows)
        with (root / "head_info.json").open("w") as handle:
            json.dump({
                "norm_name": self.adapter.info.norm_name,
                "conv_name": self.adapter.info.conv_name,
                "parameter_shapes": {
                    name: list(shape)
                    for name, shape in self.adapter.info.parameter_shapes.items()
                },
                "adaptable_parameters": self.adapter.info.adaptable_parameters,
                "total_parameters": self.adapter.info.total_parameters,
                "adaptable_ratio": self.adapter.info.adaptable_ratio,
            }, handle, indent=2)

        # GT / fully sampled k-space only enter the existing post-hoc metric stage, never phi adaptation.
        metrics = post_eval_normalize(
            recon_dir=str(recons_dir),
            val_dir=str(self.val_dir),
            plot_dir=str(plots_dir),
            json_basename="results_mcpp",
            mask_select=self.mask_select,
        )
        return {"metrics": metrics, "runtime": sample_rows}
