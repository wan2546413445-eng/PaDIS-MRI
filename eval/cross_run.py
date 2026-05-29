import os
import sys
import json
import argparse
import pickle
import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dnnlib
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(repo_root, 'train', 'padis-mri'))
from torch_utils import persistence

@persistence.import_hook
def _fix_cross_checkpoint_source(meta):
    if hasattr(meta, "module_src") and meta.module_src is not None:
        meta.module_src = meta.module_src.replace(
            "from .networks import Linear, UNetBlock, SongUNet",
            "from training.networks import Linear, UNetBlock, SongUNet",
        )
        meta.module_src = meta.module_src.replace(
            "isinstance(block, UNetBlock)",
            "block.__class__.__name__ == 'UNetBlock'",
        )
    return meta

from cross_evaluator import CrossDPSHyperEvaluator
from utils import post_eval_normalize

def parse_args():
    p = argparse.ArgumentParser(description="DPS/EDM/ADMM experiment runner")

    p.add_argument("--model_path", type=str, help="Path to network-snapshot .pkl (required for padis/edm)")
    p.add_argument("--val_dir", type=str, required=True, help="Validation directory with .pt samples")
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--psize", type=int, default=64)
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
    p.add_argument("--algo", type=str, required=True, choices=["cross_padis", "edm", "admm"], help="Choose reconstruction algo: padis, edm, or admm")

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
    p.add_argument("--cp_k", type=int, default=8)
    p.add_argument("--cp_local_k", type=int, default=3)
    p.add_argument("--cp_global_k", type=int, default=4)
    p.add_argument("--cp_eval_batch_size", type=int, default=2)
    p.add_argument("--cp_debug", action="store_true")
    p.add_argument("--memory_safe_eval", action="store_true")

    # mask sweep
    p.add_argument("--run_sweep_masks", action="store_true")
    p.add_argument("--mask_list", type=str, default="2,4,6,8,10", help="Comma-separated mask IDs (or seeds)")

    # unconditional samples
    p.add_argument("--run_uncond", action="store_true")
    p.add_argument("--uncond_model_paths", type=str, default="", help="Comma-separated .pkl paths for unconditional sampling")
    p.add_argument("--num_samples_per_model", type=int, default=3)

    return p.parse_args()


def parse_list(csv_str, cast=int):
    if csv_str is None or csv_str == "":
        return []
    return [cast(x.strip()) for x in csv_str.split(",") if x.strip() != ""]


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    model = None
    if args.algo in ("cross_padis", "edm"):
        if not args.model_path:
            raise ValueError("--model_path is required for algo padis/edm")
        print(f'Loading network from "{args.model_path}"...')
        with dnnlib.util.open_url(args.model_path, verbose=False) as f:
            model = pickle.load(f)['ema']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()

    sample_indices = parse_list(args.sample_indices, cast=int)

    opt = CrossDPSHyperEvaluator(cp_k=args.cp_k, cp_local_k=args.cp_local_k, cp_global_k=args.cp_global_k, cp_eval_batch_size=args.cp_eval_batch_size, cp_debug=args.cp_debug,
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

    if args.algo in ("cross_padis", "edm") and args.zeta is None:
        raise ValueError("--zeta is required for padis/edm runs (or run --run_hparam_search)")

    if args.run_evaluate_uncertainty:
        mask_list = parse_list(args.uncertainty_mask_list, cast=int)
        opt.evaluate_uncertainty(
            mask_list=mask_list,
            zeta=args.zeta if args.algo in ("cross_padis", "edm") else 0.0,
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
            zeta=args.zeta if args.algo in ("cross_padis", "edm") else 0.0,
            num_steps=args.steps,
            save_dir=os.path.join(args.save_dir, "patch_sweep"),
            algo=args.algo,
            gpus=args.gpus,
            tag="patch_sweep",
            report_every=args.report_every,
        )
    if args.algo == "cross_padis":
        print(
            f"[PaDIS Budget] steps={args.steps}, "
            f"inner_loops={args.inner_loops}, "
            f"total_updates={args.steps * args.inner_loops}"
        )
    if args.run_evaluate:
        metrics = opt.evaluate(
            zeta=args.zeta if args.algo in ("cross_padis", "edm") else 0.0,
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
            memory_safe_eval=args.memory_safe_eval,
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
            zeta=args.zeta if args.algo in ("cross_padis", "edm") else 0.0,
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
