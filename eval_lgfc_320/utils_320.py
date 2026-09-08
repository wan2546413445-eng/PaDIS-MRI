"""Author-style post evaluation on the common central 320 ROI."""

import json
import os
import re
from pathlib import Path
import sys

import numpy as np
import sigpy as sp
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EVAL_DIR = os.path.join(_REPO_ROOT, "eval")
for _path in (_REPO_ROOT, _EVAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils import fftmod


def _normalization_const_from_ksp(fs_ksp: torch.Tensor) -> float:
    ksp_numpy = fs_ksp.squeeze(0).cpu().numpy()
    acs_size = 24
    num_coils, _, _ = ksp_numpy.shape
    ksp_acs_only = sp.resize(
        sp.resize(ksp_numpy, (num_coils, acs_size, acs_size)),
        ksp_numpy.shape,
    )
    acs_img = sp.rss(
        sp.ifft(ksp_acs_only, axes=(-2, -1)),
        axes=(0,),
    )
    return float(np.percentile(np.abs(acs_img), 99))


def _nrmse(gt: np.ndarray, recon: np.ndarray) -> float:
    denom = np.linalg.norm(gt.ravel())
    return float(
        np.linalg.norm((gt - recon).ravel())
        / max(denom, 1e-12)
    )


def _center320(array: np.ndarray) -> np.ndarray:
    if array.shape[-2:] == (320, 320):
        return array
    if array.shape[-2:] == (384, 384):
        return array[..., 32:352, 32:352]
    raise ValueError(
        f"Expected 320x320 or 384x384 reconstruction, got {array.shape[-2:]}"
    )


def post_eval_normalize_320(
    recon_dir: str,
    val_dir: str,
    output_dir: str,
    json_basename: str = "results_320roi",
):
    """Evaluate reconstruction and GT on the identical central 320 ROI.

    Normalization is still derived from original fully-sampled 384 k-space.
    A 384 baseline reconstruction can also be passed; it is center-cropped
    to the same 320 ROI before metric computation.
    """
    recon_dir = Path(recon_dir)
    val_dir = Path(val_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recon_files = sorted(recon_dir.glob("recon_patch_*.npy"))
    if not recon_files:
        recon_files = sorted(recon_dir.glob("recon_*.npy"))
    if not recon_files:
        print(f"[post_eval_normalize_320] No recon files in {recon_dir}")
        return {"per_image": {}, "summary": {}}

    results = {"per_image": {}}
    eps = 1e-9

    for path in recon_files:
        match = re.search(r"_(\d+)\.npy$", path.name)
        if match is None:
            continue

        idx = int(match.group(1))
        data = torch.load(
            val_dir / f"sample_{idx}.pt",
            map_location="cpu",
            weights_only=False,
        )

        # ORIGINAL 384 full k-space normalization.
        fs_ksp = fftmod(data["ksp"])[None, ...]
        norm_val = _normalization_const_from_ksp(fs_ksp)

        gt = torch.abs(data["gt"]).cpu().numpy()
        gt = _center320(gt) / (norm_val + eps)

        recon = np.load(path)
        recon = np.squeeze(recon)
        recon = np.abs(_center320(recon)) / (norm_val + eps)

        data_range = max(float(gt.max() - gt.min()), eps)
        psnr = float(
            peak_signal_noise_ratio(
                gt,
                recon,
                data_range=data_range,
            )
        )
        ssim = float(
            structural_similarity(
                gt,
                recon,
                data_range=data_range,
            )
        )
        nrmse = _nrmse(gt, recon)

        results["per_image"][str(idx)] = {
            "psnr": psnr,
            "ssim": ssim,
            "nrmse": nrmse,
        }

        print(
            f"[320 ROI] sample {idx}: "
            f"PSNR={psnr:.3f}, SSIM={ssim:.4f}, NRMSE={nrmse:.4f}"
        )

    values = list(results["per_image"].values())
    if values:
        results["summary"] = {
            "psnr_mean": float(np.mean([v["psnr"] for v in values])),
            "psnr_std": float(np.std([v["psnr"] for v in values])),
            "ssim_mean": float(np.mean([v["ssim"] for v in values])),
            "ssim_std": float(np.std([v["ssim"] for v in values])),
            "nrmse_mean": float(np.mean([v["nrmse"] for v in values])),
            "nrmse_std": float(np.std([v["nrmse"] for v in values])),
        }
    else:
        results["summary"] = {}

    json_path = output_dir / (
        json_basename
        if json_basename.endswith(".json")
        else f"{json_basename}.json"
    )
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[post_eval_normalize_320] Metrics -> {json_path}")
    return results
