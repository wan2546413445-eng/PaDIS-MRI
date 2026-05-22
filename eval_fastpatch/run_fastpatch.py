import os
import sys
import csv
import time
import atexit
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
EVAL_DIR = os.path.join(REPO_ROOT, "eval")

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, REPO_ROOT)
import json
import argparse
import pickle
import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dnnlib

from evaluator_fast import DPSHyperEvaluator
from eval.utils import post_eval_normalize

def parse_args():
    p = argparse.ArgumentParser(description="DPS/EDM/ADMM experiment runner")

    p.add_argument("--model_path", type=str, help="Path to network-snapshot .pkl (required for padis/edm)")
    p.add_argument("--val_dir", type=str, required=True, help="Validation directory with .pt samples")
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--psize", type=int, default=64)
    p.add_argument(
        "--patch_schedule",
        type=str,
        default="fixed",
        choices=["fixed", "train_random", "coarse_to_fine", "sigma_c2f"],
        help="Patch sampling schedule for PaDIS posterior inference."
    )
    p.add_argument(
        "--sigma_switch",
        type=float,
        default=0.1,
        help="Sigma threshold for sigma_c2f patch schedule. Use large patch when sigma > threshold, small patch otherwise."
    )

    p.add_argument(
        "--multiscale_patch_sizes",
        type=str,
        default="",
        help="Comma-separated patch sizes used when patch_schedule is not fixed. "
             "If omitted: sigma_c2f uses 32,64; train_random/coarse_to_fine use 16,32,64."
    )
    p.add_argument(
        "--multiscale_patch_probs",
        type=str,
        default="",
        help="Comma-separated patch probabilities for train_random schedule. "
             "If omitted, train_random uses 0.2,0.3,0.5."
    )
    p.add_argument("--mask_select", type=int, default=7)
    p.add_argument("--val_count", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Comma-separated validation sample indices to evaluate exactly, e.g. 23,7,26,21. "
             "If provided, --val_count random sampling is ignored."
    )

    # algo selection
    p.add_argument("--algo", type=str, required=True, choices=["padis", "edm", "admm"], help="Choose reconstruction algo: padis, edm, or admm")

    # hyperparams
    p.add_argument("--zeta", type=float, default=3.0, help="Chosen zeta value (required for padis/edm calls)")
    p.add_argument("--steps", type=int, default=104, help="Number of steps (or ADMM iters)")
    p.add_argument(
        "--inner_loops",
        type=int,
        default=10,
        help="Number of posterior update loops per outer diffusion step for PaDIS dps2."
    )
    p.add_argument("--save_dir", type=str, required=True, help="Where to write outputs")
    p.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU ids (e.g. --gpus 0 1)")
    p.add_argument("--report_every", type=int, default=1)
    p.add_argument("--lam", type=float, default=1e-4, help="Lambda for TV regularization in ADMM")
    p.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate PaDIS reconstruction figures and author-style metrics during posterior sampling."
    )

    p.add_argument(
        "--intermediate_every",
        type=int,
        default=10,
        help="Save intermediate diagnostics every N outer diffusion steps; step 1 and final step are always saved."
    )

    # uncertainty quantification
    p.add_argument("--run_evaluate_uncertainty", action="store_true")
    p.add_argument("--uncertainty_mask_list", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated seeds for uncertainty (interpreted as seeds for padis/edm, mask ids for admm)")

    # patch size sweep
    p.add_argument("--run_sweep_patch_sizes", action="store_true")
    p.add_argument("--patch_sizes", type=str, default="96", help="Comma-separated patch sizes")

    # hyperparam search
    p.add_argument("--run_hparam_search", action="store_true")
    p.add_argument("--zeta_min", type=float, default=1.0)
    p.add_argument("--zeta_max", type=float, default=10.0)
    p.add_argument("--grid_points", type=int, default=5)
    p.add_argument("--random_samples", type=int, default=5)
    p.add_argument("--subset_size", type=int, default=3)

    # evaluate
    p.add_argument("--run_evaluate", action="store_true")

    # mask sweep
    p.add_argument("--run_sweep_masks", action="store_true")
    p.add_argument("--mask_list", type=str, default="2,4,6,8,10", help="Comma-separated mask IDs (or seeds)")

    # unconditional samples
    p.add_argument("--run_uncond", action="store_true")
    p.add_argument("--uncond_model_paths", type=str, default="", help="Comma-separated .pkl paths for unconditional sampling")
    p.add_argument("--num_samples_per_model", type=int, default=3)

    p.add_argument(
        "--resume_enable",
        action="store_true",
        help="Enable pseudo Noise&Resume posterior perturbation."
    )
    p.add_argument(
        "--resume_step",
        type=int,
        default=52,
        help="Outer diffusion step after which pseudo Noise&Resume injects noise."
    )
    p.add_argument(
        "--resume_noise_std",
        type=float,
        default=0.05,
        help="Noise std injected into current posterior state for pseudo Noise&Resume."
    )

    return p.parse_args()


def parse_list(csv_str, cast=int):
    if csv_str is None or csv_str == "":
        return []
    return [cast(x.strip()) for x in csv_str.split(",") if x.strip() != ""]


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    model = None
    if args.algo in ("padis", "edm"):
        if not args.model_path:
            raise ValueError("--model_path is required for algo padis/edm")
        print(f'Loading network from "{args.model_path}"...')
        with dnnlib.util.open_url(args.model_path, verbose=False) as f:
            model = pickle.load(f)['ema']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()

    sample_indices = parse_list(args.sample_indices, cast=int)
    multiscale_patch_sizes = parse_list(args.multiscale_patch_sizes, cast=int)
    multiscale_patch_probs = parse_list(args.multiscale_patch_probs, cast=float)

    # Schedule-specific defaults.
    # Keep fixed mode untouched. Only fill defaults when a multiscale schedule needs them.
    if args.patch_schedule == "sigma_c2f":
        if not multiscale_patch_sizes:
            multiscale_patch_sizes = [32, 64]
        if len(multiscale_patch_sizes) != 2:
            raise ValueError(
                "--patch_schedule sigma_c2f requires exactly two patch sizes, e.g. "
                "--multiscale_patch_sizes 32,64"
            )

    elif args.patch_schedule == "train_random":
        if not multiscale_patch_sizes:
            multiscale_patch_sizes = [16, 32, 64]
        if not multiscale_patch_probs:
            multiscale_patch_probs = [0.2, 0.3, 0.5]
        if len(multiscale_patch_sizes) != len(multiscale_patch_probs):
            raise ValueError(
                "train_random requires multiscale_patch_sizes and multiscale_patch_probs "
                "to have the same length."
            )

    elif args.patch_schedule == "coarse_to_fine":
        if not multiscale_patch_sizes:
            multiscale_patch_sizes = [16, 32, 64]

    if args.resume_enable:
        print(
            f"[Noise&Resume] enabled: "
            f"resume_step={args.resume_step}, "
            f"resume_noise_std={args.resume_noise_std}"
        )

    opt = DPSHyperEvaluator(
        model=model,
        mask_select=args.mask_select,
        val_dir=args.val_dir,
        image_size=args.image_size,
        pad=args.pad,
        psize=args.psize,
        val_count=args.val_count,
        seed=args.seed,
        sample_indices=sample_indices if sample_indices else None,

    )

    tag = ("whole" if args.algo == "edm" else "patch") if args.algo != "admm" else "admm"

    if args.run_hparam_search:
        best_zeta = opt.hyperparam_search(
            zeta_min=args.zeta_min,
            zeta_max=args.zeta_max,
            grid_points=args.grid_points,
            random_samples=args.random_samples,
            default_steps=args.steps,
            subset_size=args.subset_size,
        )
        print(f"[HParam] Best zeta is {best_zeta:.3f}")
        if args.zeta is None:
            args.zeta = float(best_zeta)

    if args.algo in ("padis", "edm") and args.zeta is None:
        raise ValueError("--zeta is required for padis/edm runs (or run --run_hparam_search)")

    if args.run_evaluate_uncertainty:
        mask_list = parse_list(args.uncertainty_mask_list, cast=int)
        opt.evaluate_uncertainty(
            mask_list=mask_list,
            zeta=args.zeta if args.algo in ("padis", "edm") else 0.0,
            num_steps=args.steps,
            pad=args.pad,
            psize=args.psize,
            algo=args.algo,
            save_dir=os.path.join(args.save_dir, "uncertainty_run"),
            gpus=args.gpus,
            report_every=args.report_every,
            tag=tag,
            lam=args.lam
        )

    if args.run_sweep_patch_sizes:
        patch_sizes = parse_list(args.patch_sizes, cast=int)
        opt.sweep_patch_sizes(
            num_trials=args.val_count,
            patch_sizes=patch_sizes,
            zeta=args.zeta if args.algo in ("padis", "edm") else 0.0,
            num_steps=args.steps,
            save_dir=os.path.join(args.save_dir, "patch_sweep"),
            algo=args.algo,
            gpus=args.gpus,
            tag="patch_sweep",
            report_every=args.report_every,
        )
    if args.algo == "padis":
        print(
            f"[PaDIS Budget] steps={args.steps}, "
            f"inner_loops={args.inner_loops}, "
            f"total_updates={args.steps * args.inner_loops}"
        )
        print(f"[Patch Schedule] {args.patch_schedule}")
        if args.patch_schedule == "train_random":
            print(f"[Patch Schedule] sizes={multiscale_patch_sizes}")
            print(f"[Patch Schedule] probs={multiscale_patch_probs}")
        elif args.patch_schedule == "coarse_to_fine":
            print(f"[Patch Schedule] sizes={multiscale_patch_sizes}")
        elif args.patch_schedule == "sigma_c2f":
            print(f"[Patch Schedule] sizes={multiscale_patch_sizes}")
            print(f"[Patch Schedule] sigma_switch={args.sigma_switch}")
    if args.run_evaluate:
        metrics = opt.evaluate(
            zeta=args.zeta if args.algo in ("padis", "edm") else 0.0,
            num_steps=args.steps,
            inner_loops=args.inner_loops,
            pad=args.pad,
            psize=args.psize,
            algo=args.algo,
            save_dir=os.path.join(args.save_dir, "evaluate"),
            tag=tag,
            gpus=args.gpus,
            report_every=args.report_every,
            lam=args.lam,
            save_intermediate=args.save_intermediate,
            intermediate_every=args.intermediate_every,
            patch_schedule=args.patch_schedule,
            multiscale_patch_sizes=multiscale_patch_sizes,
            multiscale_patch_probs=multiscale_patch_probs,
            sigma_switch=args.sigma_switch,
            resume_enable=args.resume_enable,
            resume_step=args.resume_step,
            resume_noise_std=args.resume_noise_std,
        )
        s = metrics['summary']
        print(f"PSNR:  {s['psnr_mean']:.2f} ± {s['psnr_std']:.2f}")
        print(f"SSIM:  {s['ssim_mean']:.4f} ± {s['ssim_std']:.4f}")
        print(f"NRMSE: {s['nrmse_mean']:.4f} ± {s['nrmse_std']:.4f}")

    if args.run_sweep_masks:
        mask_list = parse_list(args.mask_list, cast=int)
        opt.sweep_masks(
            num_trials=args.val_count,
            mask_list=mask_list,
            zeta=args.zeta if args.algo in ("padis", "edm") else 0.0,
            num_steps=args.steps,
            save_dir=os.path.join(args.save_dir, "mask_sweep"),
            algo=args.algo,
            gpus=args.gpus,
            tag=tag,
            report_every=args.report_every,
            lam=args.lam,
        )

    if args.run_uncond:
        paths = [p for p in args.uncond_model_paths.split(",") if p.strip()]
        if not paths:
            raise ValueError("--run_uncond requires --uncond_model_paths with at least one .pkl path")
        opt.generate_unconditional_samples(
            model_paths=paths,
            output_root=os.path.join(args.save_dir, "uncond"),
            num_samples_per_model=args.num_samples_per_model,
            algo=args.algo,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )


    # Compute final metrics and plots.
    recon_dir = os.path.join(args.save_dir, "evaluate", "recons")
    plot_dir = os.path.join(args.save_dir, "evaluate", "comp_plots")
    try:
        post_eval_normalize(
            recon_dir=recon_dir,
            val_dir=args.val_dir,
            plot_dir=plot_dir,
            json_basename="results",
            mask_select=args.mask_select,
        )
    except Exception as e:
        print(f"[post_eval_normalize] Skipped due to error: {e}")

if __name__ == "__main__":
    main()