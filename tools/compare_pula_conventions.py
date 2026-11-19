"""Compare the saved pULA handoff, final reconstruction and metrics of two runs."""

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate_dir(path):
    return path / "evaluate" if (path / "evaluate").is_dir() else path


def reconstruction_path(directory):
    paths = sorted([*directory.glob("recons/recon_patch_*.npy"), *directory.glob("recon_patch_*.npy")])
    if len(paths) != 1:
        raise ValueError(f"Expected one final reconstruction in {directory}, found {len(paths)}.")
    return paths[0]


def compare_arrays(label, real_path, complex_path):
    real = np.load(real_path, allow_pickle=False)
    complex_ = np.load(complex_path, allow_pickle=False)
    print(f"{label}: shape={real.shape} / {complex_.shape}, dtype={real.dtype} / {complex_.dtype}")
    same_shape = real.shape == complex_.shape
    close = same_shape and np.allclose(real, complex_, atol=1e-5, rtol=1e-4)
    if same_shape:
        reference = real.astype(np.complex128)
        difference = reference - complex_
        magnitude = np.abs(difference)
        error = np.linalg.norm(difference.ravel())
        reference_norm = np.linalg.norm(reference.ravel())
        relative = error / reference_norm if reference_norm else (float("inf") if error else 0.0)
        print(f"  max_abs_difference={magnitude.max():.6e}")
        print(f"  mean_abs_difference={magnitude.mean():.6e}")
        print(f"  relative_l2_difference={relative:.6e}")
    else:
        print("  Shape mismatch; difference statistics are unavailable.")
    print(f"  allclose(atol=1e-5, rtol=1e-4)={close}")
    return close


def sample_metrics(directory, index):
    candidates = (directory / "comp_plots" / "results.json", directory / "results.json")
    for path in candidates:
        if path.is_file():
            data = json.loads(path.read_text())
            return data.get("per_image", {index: data}).get(index, {})
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padis_real", type=Path, required=True)
    parser.add_argument("--paper_complex", type=Path, required=True)
    args = parser.parse_args()
    real_dir, complex_dir = map(evaluate_dir, (args.padis_real, args.paper_complex))
    print("Column order: padis_real / paper_complex")
    try:
        real_recon, complex_recon = map(reconstruction_path, (real_dir, complex_dir))
        if real_recon.name != complex_recon.name:
            raise ValueError(f"Sample files differ: {real_recon.name} and {complex_recon.name}.")
        handoff_close = compare_arrays("pULA handoff", real_dir / "pula_handoff.npy",
                                       complex_dir / "pula_handoff.npy")
        final_close = compare_arrays("Final reconstruction", real_recon, complex_recon)
        index = real_recon.stem.removeprefix("recon_patch_")
        real_metrics = sample_metrics(real_dir, index)
        complex_metrics = sample_metrics(complex_dir, index)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    for key in ("psnr", "ssim", "nrmse"):
        real, complex_ = real_metrics.get(key), complex_metrics.get(key)
        delta = ""
        if real is not None and complex_ is not None:
            delta = f" (paper_complex - padis_real = {complex_ - real:+.6e})"
        print(f"{key.upper()}: {real if real is not None else 'unavailable'} / "
              f"{complex_ if complex_ is not None else 'unavailable'}{delta}")

    if not handoff_close:
        print("WARNING: The first saved mismatch is already present at the pULA handoff, before DPS.")
    elif not final_close:
        print("WARNING: The first saved mismatch is in the final reconstruction after the pULA handoff.")
    return 0 if handoff_close and final_close else 1


if __name__ == "__main__":
    raise SystemExit(main())
