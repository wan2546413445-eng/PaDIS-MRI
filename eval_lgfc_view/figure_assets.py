"""Detached CPU snapshots and display-only exports for one actual LGFC step."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


class LGFCFigureAssets:
    def __init__(self, output_dir, outer_step, metadata):
        self.output_dir = Path(output_dir)
        self.outer_step = outer_step
        self.metadata = dict(metadata)
        self.tensors = {}

    def capture(self, **values):
        # Clone even on CPU: no autograd graph, GPU storage, or live state alias.
        for name, value in values.items():
            self.tensors[name] = None if value is None else value.detach().cpu().clone()

    def set_step_metadata(self, outer_step, sigma, inner_j, steps, inner_loops,
                          config, stats):
        self.metadata.update(
            outer_step=outer_step, sigma=sigma,
            inner_j_python_index=inner_j, inner_j_paper_index=inner_j + 1,
            steps=steps, inner_loops=inner_loops,
            reference_mode=config.reference_mode, ema_beta=config.ema_beta,
            global_start_outer=config.start_outer, global_every=config.correction_every,
            unsampled_weight=config.unsampled_weight,
            max_relative_update=config.max_relative_update,
            dc_growth_limit=config.dc_growth_limit,
            backtrack_factor=config.backtrack_factor, max_backtracks=config.max_backtracks,
            eps=config.eps, lgfc_config=asdict(config),
            correction_accepted=bool(stats["correction_accepted"].cpu().item()),
            effective_scale=float(stats["effective_scale"].cpu().item()),
            backtrack_count=int(stats["backtrack_count"].cpu().item()),
        )

    @staticmethod
    def _pooled_vmax(arrays):
        return float(np.percentile(np.concatenate([a.ravel() for a in arrays]), 99.5))

    def _png(self, filename, array, vmax):
        # This zero-range case preserves a genuinely zero correction as black.
        scaled = np.zeros_like(array) if vmax == 0 else np.clip(array / vmax, 0, 1)
        Image.fromarray(np.rint(scaled * 255).astype(np.uint8)).save(
            self.output_dir / filename
        )

    def _check_provenance(self):
        # These checks use only the saved CPU snapshots, never a forward or FFT.
        t, m = self.tensors, self.metadata
        before = t["reference_before_j"]
        expected = t["d_fov_j"]
        if before is not None and m["reference_mode"] == "ema":
            expected = m["ema_beta"] * before + (1 - m["ema_beta"]) * expected
        checks = {
            "reference_after_j_matches_same_inner_d_fov": torch.allclose(
                t["reference_after_j"], expected, rtol=1e-5, atol=1e-7),
            "reference_final_is_correction_reference": torch.equal(
                t["reference_final"], t["reference"]),
            "x_post_inner_is_pre_correction_x_fov": torch.equal(
                t["x_post_inner"], t["x_fov"]),
            "delta_k_matches_actual_masked_discrepancy": torch.allclose(
                t["delta_k"], t["unsampled_mask"] * t["k_difference"],
                rtol=1e-5, atol=1e-7),
            "x_corrected_is_returned_x_out": torch.equal(t["x_corrected"], t["x_out"]),
            "x_corrected_is_written_back_state": torch.equal(
                t["x_corrected"], t["x_corrected_written_back"]),
        }
        self.metadata["provenance_checks"] = checks
        if not all(checks.values()):
            raise RuntimeError(f"LGFC Figure provenance check failed: {checks}")

    def write(self):
        t, m = self.tensors, self.metadata
        if "x_corrected" not in t:
            raise RuntimeError("The requested Figure outer step did not execute LGFC")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Display-only derived values. The saved delta_k is the actual GPU result.
        t["unsampled_mask"] = 1.0 - t["mask4"]
        t["k_difference"] = t["k_reference"] - t["k_x"]
        t["final_update"] = t["x_out"] - t["x_fov"]
        torch.save(t, self.output_dir / "lgfc_figure_tensors.pt")
        self._check_provenance()

        mri_files = {
            "x_before_inner_j": "01_x_before_inner_j.png",
            "d_fov_j": "02_denoised_fov_j.png",
            "reference_before_j": "03_reference_before_j.png",
            "reference_after_j": "04_reference_after_j.png",
            "x_post_inner": "05_x_post_inner.png",
            "reference_final": "06_reference_final.png",
            "x_corrected": "15_x_corrected.png",
        }
        mri = {name: np.abs(t[name].numpy())[0, 0]
               for name in mri_files if t[name] is not None}
        # The shared range is pooled over precisely the six specified MRI tensors.
        shared_names = [name for name in mri_files if name != "reference_before_j"]
        mri_vmax = self._pooled_vmax([mri[name] for name in shared_names])
        for name, filename in mri_files.items():
            array = mri.get(name, np.zeros_like(mri["d_fov_j"]))
            self._png(filename, array, mri_vmax)
        m["reference_before_j_is_none"] = t["reference_before_j"] is None
        m["reference_before_j_png_is_empty_placeholder"] = t["reference_before_j"] is None

        update_files = {
            "raw_update": "12_raw_update_mag.png",
            "clipped_update": "13_clipped_update_mag.png",
            "final_update": "14_final_update_mag.png",
        }
        updates = {name: np.abs(t[name].numpy())[0, 0] for name in update_files}
        update_vmax = self._pooled_vmax(list(updates.values()))
        for name, filename in update_files.items():
            self._png(filename, updates[name], update_vmax)

        k_files = {
            "k_x": "07_k_current.png", "k_reference": "08_k_reference.png",
            "k_difference": "09_k_difference.png", "delta_k": "11_delta_k.png",
        }
        k_vis = {
            name: np.log1p(np.sqrt(np.sum(np.abs(t[name].numpy()) ** 2, axis=1)))[0]
            for name in k_files
        }
        k_vmax = self._pooled_vmax([k_vis["k_x"], k_vis["k_reference"]])
        discrepancy_vmax = self._pooled_vmax([k_vis["k_difference"], k_vis["delta_k"]])
        for name, filename in k_files.items():
            vmax = k_vmax if name in ("k_x", "k_reference") else discrepancy_vmax
            self._png(filename, k_vis[name], vmax)
        self._png("10_unsampled_mask.png", t["unsampled_mask"].numpy()[0, 0], 1.0)

        m.update(
            mri_shared_display_vmin=0.0, mri_shared_display_vmax=mri_vmax,
            update_shared_display_vmin=0.0, update_shared_display_vmax=update_vmax,
            kspace_display_ranges={
                "current_and_reference": [0.0, k_vmax],
                "difference_and_delta": [0.0, discrepancy_vmax],
            },
            display_percentile=99.5, kspace_display="log1p(multicoil_RSS)",
            fftshift_applied=False, png_orientation="native tensor H,W",
            tensor_shapes={name: None if value is None else list(value.shape)
                           for name, value in t.items()},
            tensor_dtypes={name: None if value is None else str(value.dtype)
                           for name, value in t.items()},
            display_only_derived_tensors=["unsampled_mask", "k_difference", "final_update"],
            original_source_commit="2c8c72a7a7387cbda28f8ee8824d136b9c527793",
        )
        with (self.output_dir / "00_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(m, handle, indent=2, ensure_ascii=False)
        print(
            "\n[LGFC Figure Assets]\n"
            f"sample       : {m['sample_index']}\n"
            f"outer step   : {m['outer_step']}\n"
            f"sigma        : {m['sigma']:.8g}\n"
            f"inner j      : {m['inner_j_paper_index']} / {m['inner_loops']} "
            f"(Python index {m['inner_j_python_index']})\n"
            f"reference    : {m['reference_mode']}\n"
            f"accepted     : {m['correction_accepted']}\n"
            f"output       : {self.output_dir}\n"
            "checks       : passed"
        )
