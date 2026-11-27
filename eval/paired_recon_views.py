import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np


def foreground_results(recon_dir, val_dir, output, mask_select, json_name):
    """Run the installed ESPIRiT protocol once for this reconstruction directory."""
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("post_eval_espirit.py")
    with TemporaryDirectory(dir=output) as temp:
        path = Path(temp)
        cmd = [sys.executable, str(script), "--recon-dir", str(recon_dir.resolve()),
               "--val-dir", str(val_dir.resolve()), "--plot-dir", temp,
               "--json-basename", "results"]
        if mask_select is not None:
            cmd.extend(["--mask-select", str(mask_select)])
        subprocess.run(cmd, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg"))
        result = json.loads((path / "results.json").read_text())
        if not result["per_image"]:
            raise ValueError("ESPIRiT returned no evaluated images")
        for index in result["per_image"]:
            name = f"espirit_mask_sample_{index}.npy"
            shutil.copyfile(path / "individual" / name, output / name)
        shutil.copyfile(path / "results.json", output / json_name)
    return result["per_image"]


def save_views(recon, gt, index, metrics, full_dir, foreground_dir, foreground):
    mask = np.load(foreground_dir / f"espirit_mask_sample_{index}.npy").squeeze()
    if mask.shape != gt.shape or not np.isfinite(mask).all():
        raise ValueError(f"Invalid ESPIRiT mask for sample {index}")
    masked = foreground[str(index)]
    vmax = float(np.percentile(gt, 99.5))
    for directory, image, reference, scores, prefix in [
        (full_dir, recon, gt, metrics, ""),
        (foreground_dir, recon * mask, gt * mask, masked, "espirit_masked_"),
    ]:
        display = dict(cmap="gray", vmin=0, vmax=vmax)
        plt.imsave(directory / f"recon_sample_{index}.png", image, **display)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, array, title in zip(axes, (image, reference), ("Recon", "GT")):
            ax.imshow(array, interpolation="nearest", **display)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Sample {index} | PSNR {scores[prefix+'psnr']:.2f} dB | "
                     f"SSIM {scores[prefix+'ssim']:.4f} | NRMSE {scores[prefix+'nrmse']:.4f}")
        fig.tight_layout()
        fig.savefig(directory / f"comparison_sample_{index}.png")
        plt.close(fig)
