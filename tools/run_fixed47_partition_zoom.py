"""Visualize fixed47 baseline pULA/DPS/hybrid with the same 47-layer grid."""
from pathlib import Path
from unittest.mock import patch
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "train/padis-mri")]
from eval_pula import run

ROI = np.s_[144:240, 144:240]


def save_layer(folder, stage, layer, sigma, records, gt, pad, vmax):
    count = len(records)
    cols = 3  # Five predictions plus one matched GT.
    groups = (count + 1 + cols - 1) // cols
    fig, axes = plt.subplots(
        2 * groups, cols, squeeze=False, figsize=(4 * cols, 5.5 * groups),
        gridspec_kw={"height_ratios": [2, 1] * groups},
        constrained_layout=True,
    )
    for ax in axes.flat:
        ax.axis("off")

    panels = records + [(np.pad(abs(gt), pad), None, "GT reference")]
    for k, (image, boxes, label) in enumerate(panels):
        group, col = divmod(k, cols)
        full, zoom = axes[2 * group, col], axes[2 * group + 1, col]
        h, w = image.shape
        full.imshow(image, cmap="gray", vmin=0, vmax=vmax,
                    extent=(0, w, h, 0), interpolation="nearest")

        if boxes is not None:
            ys = np.unique(boxes[:, :2])
            xs = np.unique(boxes[:, 2:])
            full.hlines(ys, xs.min(), xs.max(), color="turquoise", lw=0.55)
            full.vlines(xs, ys.min(), ys.max(), color="turquoise", lw=0.55)

        full.add_patch(Rectangle(
            (pad, pad), gt.shape[1], gt.shape[0],
            fill=False, edgecolor="red", lw=1,
        ))
        full.set(xlim=(0, w), ylim=(h, 0), title=label)

        fov = image[pad:pad + gt.shape[0], pad:pad + gt.shape[1]]
        zoom.imshow(fov[ROI], cmap="gray", vmin=0, vmax=vmax,
                    interpolation="nearest")

    fig.suptitle(
        f"Layer {layer:02d}/47   {stage}   sigma={sigma:.6g}\n"
        "Full padded image and actual grid / central 96 x 96 ROI without grid"
    )
    dest = folder / f"layer_{layer:03d}_sigma_{sigma:.6g}.png"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    print(f"[panel] {dest}", flush=True)


def main():
    args = run.parse_args()
    if args.dry_run:
        return run.main()

    import torch
    from eval_pula import recon_pula

    sigmas, tail, levels = run.sampling_schedule(args)
    if (args.fixed47_switch not in (0, 23, 47)
            or args.langevin_steps != 5 or args.dps_inner_loops != 5
            or len(sigmas) != 48 or levels + len(tail) != 47
            or args.fixed47_patch_policy != "shared_cycle_k"
            or args.fixed47_skip_pula):
        raise ValueError("Use fixed47 switch 0, 23 or 47, K=5, shared_cycle_k.")
    if (args.score_mode != "patch" or args.grid_mode != "cycle_k"
            or args.pula_variant != "baseline" or args.pula_dc_guide != "none"
            or args.diagnose_patch_score or args.diagnose_dps_jacobian
            or args.terminal_denoise or args.consensus_mode != "off"
            or args.num_trajectories is not None):
        raise ValueError("Use a single baseline trajectory without extra diagnostics.")

    index, = args.sample_indices
    sample = torch.load(Path(args.val_dir) / f"sample_{index}.pt",
                        map_location="cpu", weights_only=False)
    gt = sample["gt"].squeeze().numpy()
    if gt.shape != (384, 384):
        raise ValueError(f"Expected 384 x 384 GT, got {gt.shape}")
    vmax = float(np.percentile(abs(gt), 99))
    folder = Path(args.out_root) / "partition_panels_zoom"
    original_reconstruct = recon_pula.reconstruct

    def reconstruct_with_panels(system, oracle, *params, **kwargs):
        original_denoise = oracle.denoise_patches
        records = []
        calls = 0

        def capture(*a, **kw):
            nonlocal calls
            result = original_denoise(*a, **kw)
            # Read the actual padded prediction and actual partition; retain the graph.
            image_ri = result[0].detach().float().cpu().numpy()
            pred = image_ri[0] + 1j * image_ri[1]
            boxes = np.asarray(a[5]).copy()
            sigma = float(a[2])
            p = oracle.pad
            fov = pred[p:p + gt.shape[0], p:p + gt.shape[1]]
            err = np.sqrt(np.mean(abs(fov[ROI] - gt[ROI]) ** 2)) / vmax

            layer, inner = divmod(calls, 5)
            stage = "pULA" if layer < levels else "DPS"
            expected_sigma = sigmas[layer + 1]
            if not np.isclose(sigma, expected_sigma, rtol=1e-6, atol=1e-9):
                raise RuntimeError(f"Layer {layer+1}: observed sigma {sigma}, expected {expected_sigma}")
            dy, dx = boxes[0, [0, 2]]
            label = f"Inner {inner+1}   ({dy},{dx})   RMSE {err:.5f}"
            records.append((abs(pred), boxes, label))
            calls += 1
            if inner == 4:
                # Use the common double precision grid for identical filenames in all cases.
                save_layer(folder, stage, layer + 1, expected_sigma, records, gt, p, vmax)
                records.clear()
            return result

        with patch.object(oracle, "denoise_patches", capture):
            output = original_reconstruct(system, oracle, *params, **kwargs)
        if calls != 235:
            raise RuntimeError(f"Expected 235 predictions, observed {calls}")
        return output

    with patch.object(recon_pula, "reconstruct", reconstruct_with_panels):
        run.main()

    if len(list(folder.glob("layer_*.png"))) != 47:
        raise RuntimeError("Missing panel pages. Use a fresh --out_root if reconstruction was skipped.")


if __name__ == "__main__":
    main()
