"""Visualize one baseline pure pULA trajectory (40 levels x 6 updates)."""
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


def save_layer(folder, tag, level, total, sigma, records, gt, pad, scale):
    # 与混合版相同：每个预测的全图和无网格 ROI 上下对应。
    panels = records + [(np.pad(abs(gt), pad), None, "GT reference")]
    fig, axes = plt.subplots(
        4, 4, figsize=(16, 12), squeeze=False,
        gridspec_kw={"height_ratios": [2, 1, 2, 1]},
        constrained_layout=True,
    )
    for ax in axes.flat:
        ax.axis("off")

    for k, (image, boxes, title) in enumerate(panels):
        group, col = divmod(k, 4)
        full = axes[2 * group, col]
        zoom = axes[2 * group + 1, col]

        h, w = image.shape
        full.imshow(image, cmap="gray", vmin=0, vmax=scale,
                    extent=(0, w, h, 0), interpolation="nearest")
        if boxes is not None:
            ys = np.unique(boxes[:, :2])
            xs = np.unique(boxes[:, 2:])
            full.hlines(ys, xs.min(), xs.max(), color="turquoise", linewidth=.55)
            full.vlines(xs, ys.min(), ys.max(), color="turquoise", linewidth=.55)

        full.add_patch(Rectangle(
            (pad, pad), gt.shape[1], gt.shape[0],
            fill=False, edgecolor="red", linewidth=1,
        ))
        full.set(xlim=(0, w), ylim=(h, 0), title=title)

        fov = image[pad:pad + gt.shape[0], pad:pad + gt.shape[1]]
        zoom.imshow(fov[ROI], cmap="gray", vmin=0, vmax=scale,
                    interpolation="nearest")

    fig.suptitle(
        f"{tag}   pULA layer {level:02d}/{total}   sigma={sigma:.6g}\n"
        "Actual partition / central 96 x 96 ROI without overlay"
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"layer_{level:03d}_pULA_sigma_{sigma:.6g}_inner_01-06.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"[panel] {path}", flush=True)


def main():
    args = run.parse_args()
    if args.dry_run:
        return run.main()

    import torch
    from eval_pula import recon_pula

    _, tail, levels = run.sampling_schedule(args)
    if (levels, args.langevin_steps, len(tail)) != (40, 6, 0):
        raise ValueError("This viewer expects 40 pULA levels x 6 and no DPS.")
    if (args.score_mode != "patch" or args.grid_mode != "random_step"
            or args.pula_variant != "baseline" or args.pula_dc_guide != "none"
            or args.diagnose_patch_score or args.diagnose_dps_jacobian
            or args.terminal_denoise or args.consensus_mode != "off"
            or args.num_trajectories is not None):
        raise ValueError("Use the single baseline pure pULA trajectory without extra diagnostics.")

    index, = args.sample_indices
    sample = torch.load(Path(args.val_dir) / f"sample_{index}.pt",
                        map_location="cpu", weights_only=False)
    gt = sample["gt"].squeeze().numpy()
    if gt.shape != (384, 384):
        raise ValueError(f"Expected 384 x 384 GT, got {gt.shape}")
    vmax = float(np.percentile(abs(gt), 99))
    folder = (Path(args.out_root) / args.grid_mode / f"sample_{index}"
              / "evaluate" / "partition_panels_zoom")
    original_reconstruct = recon_pula.reconstruct

    def reconstruct_with_panels(system, oracle, *params, **kwargs):
        original_denoise = oracle.denoise_patches
        records = []
        calls = 0

        def capture(*a, **kw):
            nonlocal calls
            result = original_denoise(*a, **kw)
            # Keep all plotting on a detached copy; return the original tensor unchanged.
            image_ri = result[0].detach().float().cpu().numpy()
            pred = image_ri[0] + 1j * image_ri[1]
            boxes = np.asarray(a[5]).copy()
            sigma = float(a[2])
            p = oracle.pad
            fov = pred[p:p + gt.shape[0], p:p + gt.shape[1]]
            err = np.sqrt(np.mean(abs(fov[ROI] - gt[ROI]) ** 2)) / vmax

            layer, inner = divmod(calls, args.langevin_steps)

            dy, dx = boxes[0, [0, 2]]
            label = f"Inner {inner+1}   ({dy},{dx})   RMSE {err:.5f}"
            records.append((abs(pred), boxes, label))
            calls += 1
            if inner + 1 == args.langevin_steps:
                save_layer(folder, f"Sample {index}", layer + 1, levels, sigma, records, gt, p, vmax)
                records.clear()
            return result

        with patch.object(oracle, "denoise_patches", capture):
            output = original_reconstruct(system, oracle, *params, **kwargs)
        expected = levels * args.langevin_steps
        if calls != expected:
            raise RuntimeError(f"Expected {expected} predictions, observed {calls}")
        return output

    with patch.object(recon_pula, "reconstruct", reconstruct_with_panels):
        run.main()

    if len(list(folder.glob("layer_*.png"))) != levels:
        raise RuntimeError("Missing panel pages. Use a fresh --out_root if reconstruction was skipped.")


if __name__ == "__main__":
    main()
