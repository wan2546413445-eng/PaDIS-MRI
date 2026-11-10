"""Compare FH correction vectors and the transition into the DPS tail."""

import json
import sys
from pathlib import Path

import numpy as np


def _cpu_image(tensor):
    return tensor.detach().squeeze().cpu().numpy().copy()


def _difference(reference, other, foreground):
    results = {}
    for name, a, b in (
        ("full_fov", reference.ravel(), other.ravel()),
        ("gt_foreground", reference[foreground], other[foreground]),
    ):
        difference = float(np.linalg.norm(b - a))
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        cosine = float(np.vdot(a, b).real) / (a_norm * b_norm) if a_norm and b_norm else None
        results[name] = {
            "rms_difference": difference / np.sqrt(a.size),
            "relative_l2_difference": difference / a_norm if a_norm else None,
            "cosine": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
        }
    return results


class FHDiagnostics:
    def __init__(self, cg_steps):
        self.cg_steps = cg_steps
        self.steps = {step for step in (20, 32, cg_steps) if step <= cg_steps}
        self.corrections = {}
        self.transition = None

    def record_corrections(self, step, sigma, isotropic, dct, fh):
        self.corrections[step] = (
            sigma, _cpu_image(isotropic), _cpu_image(dct), _cpu_image(fh),
        )

    def record_transition(self, state, last_prediction):
        self.transition = {
            "state_entering_dps": _cpu_image(state),
            "last_cg_denoised_prediction": _cpu_image(last_prediction),
        }

    def save(self, output, index, clean, final_reconstruction):
        # The evaluation reference is used only after sampling, to define a
        # foreground proxy shared by both arms. It is not a brain segmentation.
        magnitude = np.abs(_cpu_image(clean))
        threshold = 0.1 * float(np.quantile(magnitude, 0.99))
        foreground = magnitude >= threshold
        comparisons = {
            str(step): {
                "sigma": sigma,
                "dct_vs_isotropic": _difference(iso, dct, foreground),
                "full_fh_vs_dct": _difference(dct, fh, foreground),
                "full_fh_vs_isotropic": _difference(iso, fh, foreground),
            }
            for step, (sigma, iso, dct, fh) in self.corrections.items()
        }
        np.savez_compressed(
            output / f"fh_transition_{index}.npz", gt_foreground=foreground, **self.transition,
        )
        magnitude_rmse = {}
        for key, prediction in (
            ("last_cg_denoised_prediction", self.transition["last_cg_denoised_prediction"]),
            ("final_reconstruction", final_reconstruction.squeeze()),
        ):
            error = np.abs(prediction) - magnitude
            magnitude_rmse[key] = {
                "full_fov": float(np.sqrt(np.mean(error**2))),
                "gt_foreground": float(np.sqrt(np.mean(error[foreground]**2))),
            }
        report = {
            "transition_after_step": self.cg_steps,
            "foreground_definition": "GT magnitude >= 10% of its 99th percentile",
            "foreground_pixels": int(foreground.sum()),
            "same_state_cg_corrections_first_inner_iteration": comparisons,
            "magnitude_rmse_to_gt": magnitude_rmse,
        }
        (output / f"fh_diagnostics_{index}.json").write_text(json.dumps(report, indent=2) + "\n")


def compare_pair(root, index):
    iso, fh = (root / group / f"sample_{index}" / "evaluate" for group in ("isotropic", "fh"))
    iso_report = json.loads((iso / f"fh_diagnostics_{index}.json").read_text())
    report = json.loads((fh / f"fh_diagnostics_{index}.json").read_text())
    report["magnitude_rmse_to_gt"] = {
        "isotropic": iso_report["magnitude_rmse_to_gt"],
        "fh": report["magnitude_rmse_to_gt"],
    }
    iso_transition = iso / f"fh_transition_{index}.npz"
    fh_transition = fh / f"fh_transition_{index}.npz"
    with np.load(iso_transition) as a, np.load(fh_transition) as b:
        foreground = a["gt_foreground"]
        report["trajectory_difference_fh_vs_isotropic"] = {
            key: _difference(a[key], b[key], foreground)
            for key in ("state_entering_dps", "last_cg_denoised_prediction")
        }
    final_a, final_b = (np.load(p / "recons" / f"recon_patch_{index}.npy").squeeze() for p in (iso, fh))
    report["trajectory_difference_fh_vs_isotropic"]["final_reconstruction"] = _difference(
        final_a, final_b, foreground,
    )
    path = root / f"fh_diagnostic_pair_{index}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"FH paired diagnostic saved to {path}")


if __name__ == "__main__":
    compare_pair(Path(sys.argv[1]), int(sys.argv[2]))
