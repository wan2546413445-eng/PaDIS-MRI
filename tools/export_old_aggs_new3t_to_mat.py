#!/usr/bin/env python3
"""
Export existing old AGGs+OC / AGGs+OC+LGFC new-3T reconstructions to .mat
using the SAME ACS-99 normalization and full-image metric definitions as
PaDIS-MRI eval/utils.py::post_eval_normalize().

This script does NOT rerun reconstruction and does NOT modify source files.

Input root:
  /mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/
    agg_overlap_simply_ckpt010006_78x10_serial/

Outputs are added under the existing MATLAB evaluation root:
  /mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-matlab/brain_roi_mat/

Method names:
  04_old_AGGs_OC
  05_old_AGGs_OC_LGFC

Dataset names:
  new3T_T1FLAIR_all32_R7_s78x10
  new3T_T2_all50_R7_s78x10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import sigpy as sp
import torch
from skimage.metrics import structural_similarity as ssim


DEFAULT_RECON_ROOT = Path(
    "/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/"
    "agg_overlap_simply_ckpt010006_78x10_serial"
)
DEFAULT_MAT_ROOT = Path(
    "/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-matlab/brain_roi_mat"
)

JOBS = [
    {
        "source_dir": "new3t_t1flair_agg_only",
        "method": "04_old_AGGs_OC",
        "dataset": "new3T_T1FLAIR_all32_R7_s78x10",
        "expected_n": 32,
    },
    {
        "source_dir": "new3t_t1flair_agg_lgfc",
        "method": "05_old_AGGs_OC_LGFC",
        "dataset": "new3T_T1FLAIR_all32_R7_s78x10",
        "expected_n": 32,
    },
    {
        "source_dir": "new3t_t2_agg_only",
        "method": "04_old_AGGs_OC",
        "dataset": "new3T_T2_all50_R7_s78x10",
        "expected_n": 50,
    },
    {
        "source_dir": "new3t_t2_agg_lgfc",
        "method": "05_old_AGGs_OC_LGFC",
        "dataset": "new3T_T2_all50_R7_s78x10",
        "expected_n": 50,
    },
]


def fftmod(x: torch.Tensor) -> torch.Tensor:
    """Match eval/utils.py fftmod exactly, without mutating source data."""
    x = x.clone()
    x[..., ::2, :] *= -1
    x[..., :, ::2] *= -1
    return x


def normalization_const_from_ksp(fs_ksp: torch.Tensor) -> float:
    """Match eval/utils.py ACS-99 normalization exactly."""
    ksp_numpy = fs_ksp.squeeze(0).cpu().numpy()
    acs_size = 24
    num_coils, h, w = ksp_numpy.shape
    ksp_acs_only = sp.resize(
        sp.resize(ksp_numpy, (num_coils, acs_size, acs_size)),
        ksp_numpy.shape,
    )
    acs_img = sp.rss(
        sp.ifft(ksp_acs_only, axes=(-2, -1)),
        axes=(0,),
    )
    value = float(np.percentile(np.abs(acs_img), 99))
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid ACS-99 normalization constant: {value}")
    return value


def nrmse_np(gt: np.ndarray, x: np.ndarray) -> float:
    denom = np.linalg.norm(gt)
    if denom <= 0:
        raise RuntimeError("GT norm is zero")
    return float(np.linalg.norm(gt - x) / denom)


def psnr_np(gt: np.ndarray, x: np.ndarray, data_range: float) -> float:
    mse = float(np.mean((gt - x) ** 2))
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def discover_samples(source_dir: Path):
    pairs = []
    for sample_dir in source_dir.glob("sample_*"):
        if not sample_dir.is_dir():
            continue
        m = re.fullmatch(r"sample_(\d+)", sample_dir.name)
        if not m:
            continue
        idx = int(m.group(1))
        recon = sample_dir / "evaluate" / "recons" / f"recon_patch_{idx}.npy"
        config = sample_dir / "config.json"
        if recon.is_file() and config.is_file():
            pairs.append((idx, sample_dir, recon, config))
    return sorted(pairs, key=lambda z: z[0])


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_2d_magnitude(arr) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    arr = np.abs(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D image after squeeze, got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def process_sample(idx: int, recon_path: Path, config_path: Path, out_path: Path):
    cfg = load_config(config_path)

    val_dir = Path(cfg["val_dir"])
    sample_indices = cfg.get("sample_indices", [])
    if sample_indices and idx not in [int(v) for v in sample_indices]:
        raise RuntimeError(
            f"{config_path}: sample index {idx} not present in sample_indices={sample_indices}"
        )

    sample_pt = val_dir / f"sample_{idx}.pt"
    if not sample_pt.is_file():
        raise FileNotFoundError(sample_pt)

    recon_raw = np.load(recon_path)
    data = torch.load(sample_pt, map_location="cpu", weights_only=False)

    for key in ("gt", "ksp"):
        if key not in data:
            raise KeyError(f"{sample_pt} missing key {key!r}")

    gt_raw = data["gt"]
    fs_ksp = fftmod(data["ksp"])[None, ...]
    norm_val = normalization_const_from_ksp(fs_ksp)

    epsilon = 1e-9
    gt_norm = as_2d_magnitude(gt_raw) / (norm_val + epsilon)
    recon_norm = as_2d_magnitude(recon_raw) / (norm_val + epsilon)

    if gt_norm.shape != recon_norm.shape:
        raise RuntimeError(
            f"sample_{idx}: GT/recon shape mismatch "
            f"{gt_norm.shape} vs {recon_norm.shape}"
        )

    data_range = float(gt_norm.max() - gt_norm.min())
    if not np.isfinite(data_range) or data_range <= 0:
        data_range = float(gt_norm.max() + epsilon)
    if data_range <= 0:
        raise RuntimeError(f"sample_{idx}: invalid data_range={data_range}")

    full_psnr = psnr_np(gt_norm, recon_norm, data_range)
    full_ssim = float(ssim(gt_norm, recon_norm, data_range=data_range))
    full_nrmse = nrmse_np(gt_norm, recon_norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        out_path,
        {
            "sample_index": np.array([[idx]], dtype=np.int32),
            "gt_norm": gt_norm,
            "recon_norm": recon_norm,
            "data_range": np.array([[data_range]], dtype=np.float64),
            "full_psnr": np.array([[full_psnr]], dtype=np.float64),
            "full_ssim": np.array([[full_ssim]], dtype=np.float64),
            "full_nrmse": np.array([[full_nrmse]], dtype=np.float64),
            "acs99_norm": np.array([[norm_val]], dtype=np.float64),
            "source_recon": str(recon_path),
            "source_val_dir": str(val_dir),
            "source_config": str(config_path),
        },
        do_compression=True,
    )

    return {
        "sample_index": idx,
        "full_psnr": full_psnr,
        "full_ssim": full_ssim,
        "full_nrmse": full_nrmse,
        "acs99_norm": norm_val,
        "val_dir": str(val_dir),
        "recon_path": str(recon_path),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_index",
        "full_psnr",
        "full_ssim",
        "full_nrmse",
        "acs99_norm",
        "val_dir",
        "recon_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def finite_summary(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon-root", type=Path, default=DEFAULT_RECON_ROOT)
    parser.add_argument("--mat-root", type=Path, default=DEFAULT_MAT_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already exported sample_*.mat files.",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("PaDIS-MRI old AGGs new-3T -> MATLAB ACS-99 export")
    print(f"Recon root : {args.recon_root}")
    print(f"MAT root   : {args.mat_root}")
    print("=" * 88)

    for job in JOBS:
        source_dir = args.recon_root / job["source_dir"]
        out_dir = args.mat_root / job["method"] / job["dataset"]

        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {source_dir}")

        samples = discover_samples(source_dir)
        indices = [x[0] for x in samples]

        print()
        print("-" * 88)
        print(f"Source  : {source_dir}")
        print(f"Method  : {job['method']}")
        print(f"Dataset : {job['dataset']}")
        print(f"Found   : {len(samples)} samples")
        print(f"Indices : {indices[:8]}{' ...' if len(indices) > 8 else ''}")

        if len(samples) != job["expected_n"]:
            raise RuntimeError(
                f"{job['source_dir']}: expected {job['expected_n']} samples, "
                f"found {len(samples)}"
            )

        expected_indices = list(range(job["expected_n"]))
        if indices != expected_indices:
            raise RuntimeError(
                f"{job['source_dir']}: sample indices are not exactly "
                f"0..{job['expected_n'] - 1}"
            )

        rows = []
        for ii, (idx, sample_dir, recon_path, config_path) in enumerate(samples, 1):
            out_path = out_dir / f"sample_{idx}.mat"
            if out_path.exists() and not args.overwrite:
                # Recompute metrics for the audit CSV only if needed.
                d = sio.loadmat(
                    out_path,
                    variable_names=[
                        "full_psnr",
                        "full_ssim",
                        "full_nrmse",
                        "acs99_norm",
                        "source_val_dir",
                        "source_recon",
                    ],
                )
                rows.append(
                    {
                        "sample_index": idx,
                        "full_psnr": float(np.squeeze(d["full_psnr"])),
                        "full_ssim": float(np.squeeze(d["full_ssim"])),
                        "full_nrmse": float(np.squeeze(d["full_nrmse"])),
                        "acs99_norm": float(np.squeeze(d["acs99_norm"])),
                        "val_dir": str(np.squeeze(d.get("source_val_dir", ""))),
                        "recon_path": str(np.squeeze(d.get("source_recon", ""))),
                    }
                )
                print(f"[{ii:02d}/{len(samples):02d}] sample_{idx}: exists, kept")
                continue

            row = process_sample(idx, recon_path, config_path, out_path)
            rows.append(row)
            print(
                f"[{ii:02d}/{len(samples):02d}] sample_{idx}: "
                f"PSNR={row['full_psnr']:.4f}, "
                f"SSIM={row['full_ssim']:.5f}, "
                f"NRMSE={row['full_nrmse']:.5f}"
            )

        write_csv(out_dir / "full_metrics_audit.csv", rows)

        p_m, p_s = finite_summary([r["full_psnr"] for r in rows])
        s_m, s_s = finite_summary([r["full_ssim"] for r in rows])
        n_m, n_s = finite_summary([r["full_nrmse"] for r in rows])

        summary = {
            "method": job["method"],
            "dataset": job["dataset"],
            "N": len(rows),
            "full_PSNR_mean": p_m,
            "full_PSNR_std": p_s,
            "full_SSIM_mean": s_m,
            "full_SSIM_std": s_s,
            "full_NRMSE_mean": n_m,
            "full_NRMSE_std": n_s,
        }
        with (out_dir / "full_metrics_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(
            f"Summary : PSNR {p_m:.4f} ± {p_s:.4f} | "
            f"SSIM {s_m:.5f} ± {s_s:.5f} | "
            f"NRMSE {n_m:.5f} ± {n_s:.5f}"
        )
        print(f"Output  : {out_dir}")

    print()
    print("All four exports completed.")


if __name__ == "__main__":
    main()
