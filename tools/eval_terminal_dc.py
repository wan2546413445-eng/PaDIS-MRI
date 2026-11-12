"""Compare terminal multicoil data consistency on two saved v5 reconstructions."""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from eval_fast.recon_fast import _cg_correction, _channels


SAMPLES = (19, 26)
METRICS = ("psnr", "ssim", "nrmse")
BASELINE = {19: (34.5573, 0.886104, 0.109431), 26: (35.4200, 0.843731, 0.083592)}
BASELINE_TOL = (1e-4, 1e-6, 1e-6)
SOFT_LAMBDA, SOFT_ITERS = 0.1, 5
LS_NORMAL_RTOL = 1e-10
METHODS = ("original", "soft_cg", "unregularized_ls_cg30", "unregularized_ls_cg60")


def _encode(operator, image):
    return operator.forward(_channels(image))


def _norm_sq(value):
    vector = value.flatten()
    return torch.vdot(vector, vector).real


@torch.no_grad()
def unregularized_ls_dc(operator, measurement, original, adjoint_y_norm, max_iters):
    """CGLS from m on the full masked multicoil operator, in complex128."""
    image = original.to(torch.complex128)
    data_residual = measurement - _encode(operator, image)
    normal_residual = operator.adjoint(data_residual)
    direction = normal_residual
    gamma = _norm_sq(normal_residual)
    target_sq = (LS_NORMAL_RTOL * adjoint_y_norm) ** 2

    for iterations in range(max_iters + 1):
        gamma_value = gamma.item()
        if not math.isfinite(gamma_value):
            return image, iterations, "nonfinite_normal_residual"
        if gamma_value <= target_sq:
            return image, iterations, "normal_residual_tolerance"
        if iterations == max_iters:
            return image, iterations, "max_iterations"

        projected = _encode(operator, direction)
        curvature = _norm_sq(projected).item()
        if not math.isfinite(curvature) or curvature <= 0:
            return image, iterations, "invalid_curvature"
        step = gamma_value / curvature
        next_image = image + step * direction
        if not torch.isfinite(next_image).all().item():
            return image, iterations, "nonfinite_update"
        if torch.equal(next_image, image):
            return image, iterations, "stagnated"

        image = next_image
        data_residual = data_residual - step * projected
        normal_residual = operator.adjoint(data_residual)
        next_gamma = _norm_sq(normal_residual)
        direction = normal_residual + (next_gamma / gamma) * direction
        gamma = next_gamma


def _recons_dir(output, method):
    return output / method / "evaluate" / "recons"


def _evaluate(output, method, val_dir):
    from utils import post_eval_normalize

    recons = _recons_dir(output, method)
    plots = recons.parent / "comp_plots"
    post_eval_normalize(
        recon_dir=str(recons), val_dir=val_dir,
        plot_dir=str(plots), mask_select=7,
    )
    return json.loads((plots / "results.json").read_text())["per_image"]


def _verify_original(results):
    for sample, expected in BASELINE.items():
        metrics = results.get(str(sample), {})
        for name, target, tol in zip(METRICS, expected, BASELINE_TOL):
            actual = metrics.get(name)
            if actual is None or not math.isclose(actual, target, rel_tol=0, abs_tol=tol):
                raise ValueError(
                    f"Original sample {sample}: {name}={actual}, expected {target} "
                    f"(absolute tolerance {tol}). Check v5 files and evaluation settings. "
                    "Terminal DC was not run."
                )


def _timed_dc(solve):
    torch.cuda.synchronize()
    start = perf_counter()
    result = solve()
    torch.cuda.synchronize()
    return result, perf_counter() - start


def _measure(operator, measurement, image, measurement_norm, adjoint_y_norm):
    residual = _encode(operator, image.to(torch.complex128)) - measurement
    return {
        "relative_measurement_residual": _norm_sq(residual).sqrt().item() / measurement_norm,
        "relative_normal_residual": _norm_sq(operator.adjoint(residual)).sqrt().item() / adjoint_y_norm,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, required=True, help="v5 run containing evaluate/config.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory outside the v5 run")
    args = parser.parse_args()
    source, output = args.input_run.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error("--output-dir must be outside the original v5 run.")
    config = json.loads((source / "evaluate" / "config.json").read_text())
    expected = {"mode": "mri_recurrent_hybrid", "mask_select": 7, "seed": 123, "image_size": 384}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Expected a v5 configuration with {expected}; check {source}/evaluate/config.json.")
    if not set(SAMPLES).issubset(config.get("sample_indices", [])):
        raise ValueError("The v5 configuration must include both samples 19 and 26.")

    inputs = {sample: source / "evaluate" / "recons" / f"recon_patch_{sample}.npy" for sample in SAMPLES}
    originals = {sample: np.load(path, allow_pickle=False) for sample, path in inputs.items()}
    shape = (1, config["image_size"], config["image_size"])
    for sample, image in originals.items():
        if image.shape != shape or image.dtype not in (np.complex64, np.complex128) or not np.isfinite(image).all():
            raise ValueError(f"{inputs[sample]} must contain a finite native complex v5 FOV of shape {shape}.")

    if not torch.cuda.is_available():
        raise RuntimeError("The original PaDIS-MRI measurement loader requires CUDA.")
    torch.cuda.set_device(0)
    from evaluator_fixed_seed123 import DPSHyperEvaluator

    evaluator = DPSHyperEvaluator(
        model=None, sample_indices=SAMPLES,
        **{key: config[key] for key in ("mask_select", "val_dir", "image_size", "pad", "psize", "seed")},
    )
    output.mkdir(parents=True, exist_ok=False)
    recons = _recons_dir(output, "original")
    recons.mkdir(parents=True)
    for path in inputs.values():
        shutil.copyfile(path, recons / path.name)
    baseline = _evaluate(output, "original", config["val_dir"])
    _verify_original(baseline)

    for method in METHODS[1:]:
        _recons_dir(output, method).mkdir(parents=True)

    rows = []
    for sample in SAMPLES:
        measurement, _, operator = evaluator._load_sample(sample)
        original = torch.from_numpy(originals[sample]).unsqueeze(1).to(measurement.device)
        measurement_double = measurement.to(torch.complex128)
        measurement_norm = _norm_sq(measurement_double).sqrt().item()
        adjoint_y_norm = _norm_sq(operator.adjoint(measurement_double)).sqrt().item()
        if not all(math.isfinite(norm) and norm > 0 for norm in (measurement_norm, adjoint_y_norm)):
            raise ValueError(f"Sample {sample} has an invalid measurement or adjoint norm for relative residuals.")

        soft, soft_seconds = _timed_dc(lambda: original + _cg_correction(
            operator, operator.adjoint(measurement), original, SOFT_LAMBDA, SOFT_ITERS,
        ))
        candidates = [
            ("original", "original", original, 0, 0, "no_dc", 0.0),
            ("soft_cg", "regularized_increment", soft, SOFT_ITERS, SOFT_ITERS, "fixed_iterations", soft_seconds),
        ]
        for max_iters, method in zip((30, 60), METHODS[2:]):
            (image, actual_iters, stop), seconds = _timed_dc(lambda: unregularized_ls_dc(
                operator, measurement_double, original, adjoint_y_norm, max_iters,
            ))
            candidates.append((method, "unregularized_ls_dc", image, max_iters, actual_iters, stop, seconds))

        for method, dc_type, image, budget, iterations, stop, seconds in candidates:
            image = image.to(original.dtype)
            if method != "original":
                np.save(_recons_dir(output, method) / inputs[sample].name, image.squeeze(1).cpu().numpy())
            rows.append({
                "sample": sample, "method": method, "dc_type": dc_type,
                "cg_max_iterations": budget, "cg_iterations": iterations, "cg_stop_reason": stop,
                "terminal_dc_seconds": seconds,
                **_measure(operator, measurement_double, image, measurement_norm, adjoint_y_norm),
            })

    results = {"original": baseline}
    for method in METHODS[1:]:
        results[method] = _evaluate(output, method, config["val_dir"])
    for row in rows:
        sample = str(row["sample"])
        for name in METRICS:
            value = results[row["method"]][sample][name]
            row[name] = value
            row[f"delta_{name}"] = value - baseline[sample][name]

    summary = output / "paired_summary.json"
    summary.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(f"Saved paired terminal DC comparisons to {summary}")


if __name__ == "__main__":
    main()
