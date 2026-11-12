import os
import re
import json
import numpy as np
import torch
import random
import sigpy as sp
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Union

from inverse_operators import *
from dnnlib.util import configure_bart

configure_bart()
from bart import bart

torch.manual_seed(123)
np.random.seed(123)

def fftmod(x: torch.Tensor) -> torch.Tensor:
    x[...,::2,:] *= -1
    x[...,:,::2] *= -1
    return x

def _to_mag_np(x):
    if isinstance(x, np.ndarray):
        arr = x
    else:
        x = x.detach().cpu()
        arr = x.numpy()
    arr = np.squeeze(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    else:
        arr = np.abs(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        arr = arr[..., -2:, :].squeeze()
    return arr.astype(np.float64, copy=False)

def _nrmse_np(gt: np.ndarray, x: np.ndarray) -> float:
    return np.linalg.norm(gt - x) / np.linalg.norm(gt)

def _psnr_np(gt, x, data_range):
    mse = np.mean((gt - x) ** 2)

    if not np.isfinite(mse):
        return float('nan')

    if mse == 0:
        return float('inf')

    return 20 * np.log10(data_range / np.sqrt(mse))
def makeFigures(noisy2, 
                denoised2, 
                orig2, 
                i, 
                out_dir='.', 
                tag="recon", 
                imsize=256, 
                plot=True,
                vmax_percentile=99.5):
    
    gt_np = _to_mag_np(orig2)
    recon_np = _to_mag_np(noisy2) 
    denoise_np = _to_mag_np(denoised2)
    
    epsilon = 1e-9
    norm_val = 1.0 
    gt_np = gt_np / (norm_val + epsilon)
    recon_np = recon_np / (norm_val + epsilon)
    denoise_np = denoise_np / (norm_val + epsilon)
    
    data_range = gt_np.max() - gt_np.min()
    if data_range <= 0: data_range = float(gt_np.max() + epsilon)

    noisypsnr = _psnr_np(gt_np, recon_np, data_range=data_range)
    denoisedpsnr = _psnr_np(gt_np, denoise_np, data_range=data_range)

    noisyssim = ssim(gt_np, recon_np, data_range=data_range)
    denoisedssim = ssim(gt_np, denoise_np, data_range=data_range)
    
    noisynrmse = _nrmse_np(gt_np, recon_np)
    denoisednrmse = _nrmse_np(gt_np, denoise_np)

    if plot:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        vmax = np.percentile(gt_np, vmax_percentile)
        vmin = 0.0 

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(recon_np, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(f"Recon\nPSNR: {noisypsnr:.2f} dB")

        plt.subplot(1, 3, 2)
        plt.imshow(denoise_np, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(f"Diffusion Recon\nPSNR: {denoisedpsnr:.2f} dB")

        plt.subplot(1, 3, 3)
        plt.imshow(gt_np, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title("Ground Truth")

        plt.tight_layout()
        save_path = out_dir_path / f"{i}_{tag}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close('all')

    return noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse



def _espirit_foreground_mask(fs_ksp: torch.Tensor) -> np.ndarray:
    """
    Compute the pULA-style foreground mask from fully sampled multi-coil
    k-space with BART: ecalib -r16 -A -m1 followed by RSS across coils.

    Unlike the saved sensitivity maps, this call keeps ESPIRiT's default
    eigenvalue crop (0.8), which is what turns the sensitivity support into
    a foreground mask.
    """
    ksp = fs_ksp.detach().cpu().numpy() if torch.is_tensor(fs_ksp) else np.asarray(fs_ksp)
    ksp = np.squeeze(ksp)
    if ksp.ndim != 3:
        raise ValueError(f"Expected fully sampled k-space [coil,H,W], got {ksp.shape}")

    num_coils = ksp.shape[0]
    ksp_bart = np.transpose(ksp, (1, 2, 0))[:, :, None, :]
    sensitivity = np.asarray(
        bart(1, "ecalib -r16 -A -m1", ksp_bart)
    ).squeeze()

    coil_axes = [axis for axis, size in enumerate(sensitivity.shape) if size == num_coils]
    if sensitivity.ndim != 3 or not coil_axes:
        raise ValueError(
            f"Unexpected BART sensitivity-map shape {sensitivity.shape}; "
            f"expected one coil axis of length {num_coils}"
        )

    foreground = np.sqrt(np.sum(np.abs(sensitivity) ** 2, axis=coil_axes[-1]))
    foreground = np.nan_to_num(foreground, nan=0.0, posinf=0.0, neginf=0.0)
    if foreground.shape != ksp.shape[-2:]:
        raise ValueError(
            f"ESPIRiT foreground shape {foreground.shape} does not match "
            f"k-space image shape {ksp.shape[-2:]}"
        )
    if not np.any(foreground > 0):
        raise ValueError("ESPIRiT foreground mask is empty")
    return foreground.astype(np.float64, copy=False)

def _metric_triplet(gt: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    data_range = float(gt.max() - gt.min()) if gt.size else 1.0
    if not np.isfinite(data_range) or data_range <= 0:
        data_range = float(np.max(np.abs(gt))) if gt.size else 1.0
    data_range = max(data_range, 1e-9)
    return {
        "psnr": float(_psnr_np(gt, x, data_range=data_range)),
        "ssim": float(ssim(gt, x, data_range=data_range)),
        "nrmse": float(_nrmse_np(gt, x)),
    }


def _save_image(image: np.ndarray, path: Path, cmap: str, vmin: float, vmax: float) -> None:
    """Save without axes, title, margins, or another image normalization."""
    plt.imsave(path, image, cmap=cmap, vmin=vmin, vmax=max(float(vmax), float(vmin) + 1e-9))


def post_eval_normalize(
    recon_dir: str,
    val_dir: str,
    plot_dir: str,
    json_basename: str = "results",
    mask_select: Optional[int] = None,
) -> Dict:
    """
    Keep the existing full-image metrics and add pULA-style ESPIRiT-masked
    metrics. For the latter, GT and reconstruction are multiplied by the same
    soft foreground mask before PSNR, SSIM, and NRMSE are computed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recon_dir = Path(recon_dir)
    val_dir = Path(val_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = plot_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    json_path = plot_dir / (
        json_basename if json_basename.endswith(".json") else f"{json_basename}.json"
    )

    patterns = [
        "recon_admm_*.npy",
        "recon_patch_*.npy",
        "recon_whole_*.npy",
        "recon_*.npy",
    ]
    recon_files: List[Path] = []
    pattern_used = None
    for pat in patterns:
        files = sorted(recon_dir.glob(pat))
        if files:
            recon_files = files
            pattern_used = pat
            print(f"[post_eval_normalize] Found {len(recon_files)} files via '{pat}'")
            break
    if not recon_files:
        print(f"[post_eval_normalize] No reconstruction files in {recon_dir}. Tried: {patterns}")
        return {"per_image": {}, "summary": {}}

    if "admm_mask" in (pattern_used or ""):
        regex = r"^recon_admm_mask(?P<mask>\d+)_(?P<idx>\d+)\.npy$"
    elif "admm" in (pattern_used or ""):
        regex = r"^recon_admm_(?P<idx>\d+)\.npy$"
    elif "patch" in (pattern_used or ""):
        regex = r"^recon_patch_(?P<idx>\d+)\.npy$"
    elif "whole" in (pattern_used or ""):
        regex = r"^recon_whole_(?P<idx>\d+)\.npy$"
    else:
        regex = r"^recon_(?P<idx>\d+)\.npy$"

    def normalization_const_from_ksp(fs_ksp: torch.Tensor) -> float:
        ksp_numpy = fs_ksp.squeeze(0).cpu().numpy()
        acs_size = 24
        num_coils, _, _ = ksp_numpy.shape
        ksp_acs_only = sp.resize(
            sp.resize(ksp_numpy, (num_coils, acs_size, acs_size)),
            ksp_numpy.shape,
        )
        acs_img = sp.rss(sp.ifft(ksp_acs_only, axes=(-2, -1)), axes=(0,))
        return float(np.percentile(np.abs(acs_img), 99))

    results: Dict = {
        "metric_protocol": {
            "full_image": "Existing PaDIS-MRI full-image metrics",
            "espirit_masked": (
                "GT and reconstruction multiplied by the same soft ESPIRiT "
                "foreground mask before metric computation"
            ),
            "espirit_mask_source": "BART ecalib -r16 -A -m1 followed by coil RSS",
        },
        "per_image": {},
    }
    epsilon = 1e-9

    for rf in recon_files:
        match = re.fullmatch(regex, rf.name)
        if not match:
            print(f"[post_eval_normalize] Skip unexpected filename: {rf.name}")
            continue

        if "mask" in match.groupdict():
            file_mask = int(match.group("mask"))
            if mask_select is not None and file_mask != mask_select:
                continue

        sample_idx = int(match.group("idx"))

        try:
            recon = torch.from_numpy(np.load(rf)[None, ...]).to(device)
            sample_pt = val_dir / f"sample_{sample_idx}.pt"
            data = torch.load(sample_pt, map_location=device, weights_only=False)
            gt = data["gt"][None, None, ...].to(device)
            raw_fs_ksp = data["ksp"]
            fs_ksp = fftmod(raw_fs_ksp.clone())[None, ...].to(device)
        except FileNotFoundError:
            print(f"[post_eval_normalize] Missing GT sample for idx {sample_idx}; skipping.")
            continue
        except KeyError as exc:
            print(f"[post_eval_normalize] Missing key {exc} for idx {sample_idx}; skipping.")
            continue

        norm_val = normalization_const_from_ksp(fs_ksp)
        gt2 = (torch.abs(gt) / (norm_val + epsilon)).squeeze().detach().cpu().numpy()
        rc2 = (torch.abs(recon) / (norm_val + epsilon)).squeeze().detach().cpu().numpy()

        full_metrics = _metric_triplet(gt2, rc2)
        espirit_mask = _espirit_foreground_mask(raw_fs_ksp)
        if espirit_mask.shape != gt2.shape:
            raise ValueError(
                f"ESPIRiT mask shape {espirit_mask.shape} does not match image shape {gt2.shape}"
            )

        gt_espirit = gt2 * espirit_mask
        rc_espirit = rc2 * espirit_mask
        espirit_metrics = _metric_triplet(gt_espirit, rc_espirit)

        support_threshold = float(espirit_mask.max()) * 1e-6
        support_fraction = float(np.mean(espirit_mask > support_threshold))
        mask_weight_mean = float(np.mean(espirit_mask))

        results["per_image"][str(sample_idx)] = {
            "psnr": full_metrics["psnr"],
            "ssim": full_metrics["ssim"],
            "nrmse": full_metrics["nrmse"],
            "espirit_masked_psnr": espirit_metrics["psnr"],
            "espirit_masked_ssim": espirit_metrics["ssim"],
            "espirit_masked_nrmse": espirit_metrics["nrmse"],
            "espirit_mask_support_fraction": support_fraction,
            "espirit_mask_weight_mean": mask_weight_mean,
        }

        vmax = float(np.percentile(gt2, 99.5)) if np.isfinite(gt2).all() else float(gt2.max())
        vmax = max(vmax, epsilon)
        error = np.abs(gt2 - rc2)
        error_vmax = float(np.percentile(error, 99.5)) if np.any(error > 0) else 1.0
        mask_vmax = max(float(espirit_mask.max()), epsilon)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(gt2, cmap="gray", vmin=0.0, vmax=vmax)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        axes[1].imshow(rc2, cmap="gray", vmin=0.0, vmax=vmax)
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")
        fig.suptitle(
            (
                f"Sample {sample_idx} | Full: "
                f"PSNR {full_metrics['psnr']:.2f} dB, "
                f"SSIM {full_metrics['ssim']:.4f}, "
                f"NRMSE {full_metrics['nrmse']:.4f}\n"
                f"ESPIRiT-masked: "
                f"PSNR {espirit_metrics['psnr']:.2f} dB, "
                f"SSIM {espirit_metrics['ssim']:.4f}, "
                f"NRMSE {espirit_metrics['nrmse']:.4f}"
            ),
            fontsize=13,
        )
        plt.tight_layout()
        fig.savefig(plot_dir / f"comparison_sample_{sample_idx}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        _save_image(gt2, individual_dir / f"gt_sample_{sample_idx}.png", "gray", 0.0, vmax)
        _save_image(rc2, individual_dir / f"recon_sample_{sample_idx}.png", "gray", 0.0, vmax)
        _save_image(error, individual_dir / f"error_sample_{sample_idx}.png", "inferno", 0.0, error_vmax)
        _save_image(
            espirit_mask,
            individual_dir / f"espirit_mask_sample_{sample_idx}.png",
            "gray",
            0.0,
            mask_vmax,
        )
        _save_image(
            rc_espirit,
            individual_dir / f"recon_espirit_masked_sample_{sample_idx}.png",
            "gray",
            0.0,
            vmax,
        )

    metric_names = (
        "psnr",
        "ssim",
        "nrmse",
        "espirit_masked_psnr",
        "espirit_masked_ssim",
        "espirit_masked_nrmse",
        "espirit_mask_support_fraction",
        "espirit_mask_weight_mean",
    )
    if results["per_image"]:
        summary = {}
        for name in metric_names:
            values = [item[name] for item in results["per_image"].values()]
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_std"] = float(np.std(values))
        results["summary"] = summary

    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2, default=lambda value: float(value))

    print(f"[post_eval_normalize] Metrics → {json_path}")
    print(f"[post_eval_normalize] Plots   → {plot_dir}")

    if "summary" in results:
        summary = results["summary"]
        print(
            f"[post_eval_normalize] Full image     | "
            f"PSNR {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB | "
            f"SSIM {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f} | "
            f"NRMSE {summary['nrmse_mean']:.4f} ± {summary['nrmse_std']:.4f}"
        )
        print(
            f"[post_eval_normalize] ESPIRiT-masked | "
            f"PSNR {summary['espirit_masked_psnr_mean']:.2f} ± "
            f"{summary['espirit_masked_psnr_std']:.2f} dB | "
            f"SSIM {summary['espirit_masked_ssim_mean']:.4f} ± "
            f"{summary['espirit_masked_ssim_std']:.4f} | "
            f"NRMSE {summary['espirit_masked_nrmse_mean']:.4f} ± "
            f"{summary['espirit_masked_nrmse_std']:.4f}"
        )

    return results
