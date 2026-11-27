"""CPU magnitude-image comparison, sample 26. Run with --help for overrides."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

ROOT = Path("/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/创新")
VAL = Path("/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB/sample_26.pt")


def recon_path(run, model):
    return ROOT / run / "dps40x6/T1FLAIR/R7" / model / "sample_26/evaluate/recons/recon_patch_26.npy"


def magnitude(path):
    if path.suffix == ".pt":
        import torch
        try:
            sample = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            sample = torch.load(path, map_location="cpu")
        a = np.asarray(sample["gt"])
    else:
        a = np.load(path, allow_pickle=False)
    a = a.squeeze()
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"Expected a finite 2D real or complex image: {path}, {a.shape}")
    return np.abs(a).astype(np.float64)


def haar(image, width):
    """Undecimated orthonormal Haar atoms at every valid top-left position."""
    half = width // 2
    sums = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    boxes = sums[half:, half:] - sums[:-half, half:] - sums[half:, :-half] + sums[:-half, :-half]
    a, b = boxes[:-half, :-half], boxes[:-half, half:]
    c, d = boxes[half:, :-half], boxes[half:, half:]
    return np.stack((a-b+c-d, a+b-c-d, a-b-c+d)) / width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0", type=Path, default=recon_path("E0_10k_pure_DPS_smax1", "E0_10k"))
    parser.add_argument("--opa", type=Path, default=recon_path("OPA_10k_pure_DPS_smax1", "OPA1p25_10k"))
    parser.add_argument("--gt", type=Path, default=VAL, help="validation sample_26.pt or matched raw GT .npy")
    parser.add_argument("--out", type=Path, default=ROOT / "R7_wavelet_E0_vs_OPA1p25_10k/sample_26")
    args = parser.parse_args()
    paths = [args.gt, args.e0, args.opa]
    images = [magnitude(p) for p in paths]
    gt = images[0]
    if any(a.shape != gt.shape for a in images):
        raise ValueError(f"Unmatched FOV/shapes: {[a.shape for a in images]}")
    scale = np.percentile(gt, 99)
    if scale <= 0:
        raise ValueError("GT p99 must be positive")
    images = [a / scale for a in images]
    gt = images[0]
    # Approximate foreground only; optional anatomical interpretation requires inspection.
    labels, count = ndi.label(ndi.binary_closing(gt > 0.12, iterations=2))
    if not count:
        raise ValueError("Empty GT foreground")
    foreground = ndi.binary_fill_holes(labels == (1 + np.bincount(labels.ravel())[1:].argmax()))
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    for ax, img, name in zip(axes[0], images + [foreground], ["GT", "E0", "OPA 1.25", "GT foreground"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
    for row, width in enumerate((2, 4), 1):
        coeff = [haar(img, width) for img in images]
        g = coeff[0]
        valid = ndi.minimum_filter(foreground.astype("uint8"), size=width)[width//2:gt.shape[0]-(width-1)//2, width//2:gt.shape[1]-(width-1)//2].astype(bool)
        strength = np.sqrt(np.sum(g*g, axis=0))
        lo, mid, hi = np.percentile(strength[valid], [20, 40, 70])
        masks = {"flat": valid & (strength <= lo), "weak_detail_proxy": valid & (strength > mid) & (strength <= hi)}
        error_maps = [np.sqrt(np.mean((v-g)**2, axis=0)) for v in coeff[1:]]
        vmax = max(np.percentile(np.stack(error_maps)[:,valid], 98), 1e-12)
        for ax, (name, mask) in zip(axes[row,:2], masks.items()):
            ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{width}px {name}")
            if not mask.any():
                raise ValueError(f"Empty GT stratum {name}; inspect input and thresholds")
            ref = g[:,mask]
            denom = (ref**2).sum()
            for model, v in zip(("E0", "OPA1p25"), coeff[1:]):
                pred = v[:,mask]
                gain = (pred*ref).sum()/denom if denom > 0 and name != "flat" else float("nan")
                rows.append(dict(width_px=width, region=name, model=model, n_positions=int(mask.sum()),
                                 detail_rmse=np.sqrt(np.mean((pred-ref)**2)), signed_gain=gain))
        for ax, err, model in zip(axes[row,2:], error_maps, ("E0", "OPA")):
            im = ax.imshow(np.where(valid, err, np.nan), cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"{width}px {model} detail error")
            fig.colorbar(im, ax=ax, shrink=.65)
    for ax in axes.flat:
        ax.set_axis_off()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "wavelet_detail.png", dpi=170)
    plt.close(fig)
    with (args.out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("GT, E0, OPA:", *paths, sep="\n")
    print(f"Shape {gt.shape}; common raw GT p99 = {scale:g}")
    for r in rows:
        print("{width_px}px {region:18s} {model:8s} RMSE={detail_rmse:.5f} gain={signed_gain:.3f}".format(**r))
    print(args.out)


if __name__ == "__main__":
    main()
