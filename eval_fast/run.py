import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from time import perf_counter


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PaDIS DPS and image-space CG")
    parser.add_argument(
        "--mode", required=True, choices=["dps", "cg_each", "cg_layer", "hybrid", "ct_cg_ddim"],
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--sample_indices", default="26", help="Comma-separated sample IDs")
    parser.add_argument("--mask_select", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--psize", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, help="Noise levels (default: 200 for ct_cg_ddim, 78 otherwise).")
    parser.add_argument("--inner_loops", type=int, help="Sweeps per level (default: 2 for ct_cg_ddim, 10 otherwise).")
    parser.add_argument(
        "--ddim_eta", type=float, default=argparse.SUPPRESS,
        help="Cross-level noise mixing for ct_cg_ddim in [0, 1] (default: 0.8).",
    )
    parser.add_argument(
        "--profile_ct", action="store_true", default=argparse.SUPPRESS,
        help="Record patch denoising and CG CUDA intervals for ct_cg_ddim.",
    )
    parser.add_argument("--zeta", type=float, default=3.0)
    parser.add_argument("--cg_lambda", type=float, default=0.1)
    parser.add_argument("--cg_iters", type=int, default=5)
    parser.add_argument("--cg_covariance", choices=("isotropic", "fh"), default="isotropic")
    parser.add_argument("--fh_cov_path", help="NPZ from tools/build_fh_covariance.py.")
    parser.add_argument(
        "--fh_noise_var", type=float,
        help="Independent calibration override in evaluator k-space units per real component. "
             "Defaults to the training cache. Use training or external calibration only.",
    )
    parser.add_argument("--fh_max_rank", type=int, default=16)
    parser.add_argument("--profile_fh", action="store_true",
                        help="Record CUDA intervals for the hybrid CG phase.")
    parser.add_argument("--diagnose_fh", action="store_true",
                        help="Save same-state CG comparisons and the transition into DPS.")
    parser.add_argument(
        "--cg_gain", type=float, default=None,
        help="Scale the CG increment (default: 0.1 for hybrid, 1.0 for other CG modes).",
    )
    parser.add_argument(
        "--dps_tail_steps", type=int, default=20,
        help="Number of final noise levels that use DPS in hybrid mode.",
    )
    args = parser.parse_args()
    is_ct = args.mode == "ct_cg_ddim"
    if args.steps is None:
        args.steps = 200 if is_ct else 78
    if args.inner_loops is None:
        args.inner_loops = 2 if is_ct else 10
    if is_ct:
        args.ddim_eta = getattr(args, "ddim_eta", 0.8)
        args.profile_ct = getattr(args, "profile_ct", False)
        if not 0 <= args.ddim_eta <= 1:
            parser.error("--ddim_eta must be in [0, 1].")
        if args.image_size < 1 or args.psize < 1 or args.pad < 1:
            parser.error("ct_cg_ddim needs positive image_size, psize, and pad.")
        grid_width = (args.image_size // args.psize + 1) * args.psize
        if not args.image_size + args.pad <= grid_width <= args.image_size + args.pad + 1:
            parser.error("ct_cg_ddim needs a patch grid that covers the FOV at every offset and fits in the padding.")
    elif hasattr(args, "ddim_eta") or hasattr(args, "profile_ct"):
        parser.error("--ddim_eta and --profile_ct require --mode ct_cg_ddim.")
    if args.cg_gain is None:
        args.cg_gain = 0.1 if args.mode == "hybrid" else 1.0
    if args.steps < 2 or args.inner_loops < 1:
        parser.error("Use at least two noise levels and one inner sweep.")
    if args.mode != "dps" and (
        args.cg_iters < 1 or not math.isfinite(args.cg_lambda) or args.cg_lambda <= 0
    ):
        parser.error("CG needs at least one iteration and a positive, finite cg_lambda.")
    if args.mode != "dps" and (not math.isfinite(args.cg_gain) or args.cg_gain <= 0):
        parser.error("--cg_gain must be positive and finite.")
    if args.mode == "hybrid" and not 0 < args.dps_tail_steps < args.steps:
        parser.error("Hybrid needs 0 < --dps_tail_steps < --steps.")
    if args.cg_covariance == "fh":
        if args.mode != "hybrid":
            parser.error("--cg_covariance fh requires --mode hybrid.")
        if not args.fh_cov_path:
            parser.error("FH requires --fh_cov_path with a v3 training cache.")
        if args.fh_noise_var is not None and (
            not math.isfinite(args.fh_noise_var) or args.fh_noise_var <= 0
        ):
            parser.error("--fh_noise_var must be positive and finite.")
        if args.fh_max_rank < 1:
            parser.error("--fh_max_rank must be at least 1.")
    if args.profile_fh and args.mode != "hybrid":
        parser.error("--profile_fh requires --mode hybrid.")
    if args.diagnose_fh and args.mode != "hybrid":
        parser.error("--diagnose_fh requires --mode hybrid.")
    try:
        args.sample_indices = [int(index) for index in args.sample_indices.split(",")]
    except ValueError:
        parser.error("--sample_indices must be comma-separated integers.")
    return args


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "eval")]

    import numpy as np
    import torch

    import dnnlib
    from evaluator_fixed_seed123 import DPSHyperEvaluator
    from eval_fast.fh_covariance import FHProfile, load_fh_cache
    from eval_fast.fh_diagnostics import FHDiagnostics
    from eval_fast.recon_fast import reconstruct, reconstruct_ct_cg_ddim
    from utils import post_eval_normalize

    if not torch.cuda.is_available():
        raise RuntimeError("The PaDIS-MRI evaluation pipeline requires CUDA.")
    torch.cuda.set_device(0)
    fh_variance = None
    fh_config = {}
    if args.cg_covariance == "fh":
        variance, training_noise_var, training_files = load_fh_cache(args.fh_cov_path, args.image_size)
        fh_variance = variance.cuda()
        fh_config = {
            "fh_training_files": training_files,
            "fh_training_noise_var": training_noise_var,
            "fh_noise_var_source": "training_cache" if args.fh_noise_var is None else "explicit",
            "fh_noise_var": training_noise_var if args.fh_noise_var is None else args.fh_noise_var,
        }
    with dnnlib.util.open_url(args.model_path, verbose=False) as file:
        model = pickle.load(file)["ema"].cuda().eval()
    evaluator = DPSHyperEvaluator(
        model=model, mask_select=args.mask_select, val_dir=args.val_dir,
        image_size=args.image_size, pad=args.pad, psize=args.psize,
        seed=args.seed, sample_indices=args.sample_indices,
    )

    output = Path(args.save_dir) / "evaluate"
    recons = output / "recons"
    recons.mkdir(parents=True, exist_ok=False)
    updates = args.steps * args.inner_loops
    dps_steps = {"dps": args.steps, "hybrid": args.dps_tail_steps}.get(args.mode, 0)
    cg_solves = {
        "dps": 0,
        "cg_each": updates,
        "cg_layer": args.steps,
        "hybrid": (args.steps - dps_steps) * args.inner_loops,
        "ct_cg_ddim": args.steps,
    }
    config = {
        **vars(args),
        **fh_config,
        "dps_tail_steps": dps_steps if args.mode == "hybrid" else 0,
        "rng_policy": "fixed_per_sample",
        "sample_indices": evaluator.val_indices,
        "gpu": torch.cuda.get_device_name(0),
        "patch_batch": os.getenv("PADIS_PATCH_BATCH"),
        "patch_checkpoint": os.getenv("PADIS_PATCH_CHECKPOINT", "0"),
        "patch_sweeps_per_sample": updates,
        "dps_backprops_per_sample": dps_steps * args.inner_loops,
        "cg_solves_per_sample": cg_solves[args.mode],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    elapsed = {}
    operations = {}
    gpu_profiles = {}
    profile_ct = args.mode == "ct_cg_ddim" and args.profile_ct
    for index in evaluator.val_indices:
        evaluator._set_sample_seed(idx=index, gpu_id=0)
        evaluator._reset_sample_latent(gpu_id=0)
        measurement, clean, inverseop = evaluator._load_sample(index)
        counts = dict.fromkeys((
            "main_denoise", "probe_denoise", "dps_backprops", "cg_solves",
            "cg_iterations", "precision_applies", "covariance_transfers",
            "bfgs_accepted", "bfgs_skipped",
        ), 0)
        profile = None
        if args.profile_fh:
            profile = FHProfile()
        elif profile_ct:
            profile = FHProfile(sections=("denoise", "cg_total"))
        diagnostics = FHDiagnostics(args.steps - dps_steps) if args.diagnose_fh else None
        torch.cuda.synchronize()
        start = perf_counter()
        if args.mode == "dps":
            recon = evaluator.dps2_wrapper(
                inverse_op=inverseop, measurement=measurement, clean=clean,
                zeta=args.zeta, pad=args.pad, psize=args.psize,
                num_steps=args.steps, inner_loops=args.inner_loops,
            )[0]
            # The original DPS wrapper performs one sweep and backward per update.
            counts["main_denoise"] = counts["dps_backprops"] = updates
        elif args.mode == "ct_cg_ddim":
            recon = reconstruct_ct_cg_ddim(
                model, evaluator.latents_pos, inverseop, measurement,
                num_steps=args.steps, inner_loops=args.inner_loops,
                pad=args.pad, psize=args.psize, ddim_eta=args.ddim_eta,
                cg_lambda=args.cg_lambda, cg_iters=args.cg_iters, cg_gain=args.cg_gain,
                counts=counts, profile=profile,
            )
        else:
            recon = reconstruct(
                model, evaluator.latents_pos, inverseop, measurement,
                num_steps=args.steps, inner_loops=args.inner_loops,
                pad=args.pad, psize=args.psize, mode=args.mode,
                cg_lambda=args.cg_lambda, cg_iters=args.cg_iters, cg_gain=args.cg_gain,
                dps_tail_steps=dps_steps, zeta=args.zeta,
                fh_variance=fh_variance, fh_noise_var=fh_config.get("fh_noise_var"),
                fh_max_rank=args.fh_max_rank, counts=counts, profile=profile,
                diagnostics=diagnostics,
            )
        torch.cuda.synchronize()
        elapsed[index] = perf_counter() - start
        operations[index] = counts
        if profile:
            gpu_profiles[index] = profile.summary()
        print(f"[{args.mode}] sample {index}: {elapsed[index]:.2f} s", flush=True)
        if fh_config and counts["bfgs_accepted"] == 0:
            print(f"[fh] sample {index}: no BFGS updates accepted. "
                  f"Skip reasons: {counts['bfgs_skip_reasons']}", flush=True)
        recon_array = recon.cpu().numpy()
        np.save(recons / f"recon_patch_{index}.npy", recon_array)
        if diagnostics:
            diagnostics.save(output, index, clean, recon_array)

    timing = {
        "recon_seconds_per_sample": elapsed,
        "mean_recon_seconds": sum(elapsed.values()) / len(elapsed),
    }
    if profile_ct:
        timing["gpu_seconds_per_sample"] = gpu_profiles
    (output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    config["operations_per_sample"] = operations
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.profile_fh:
        (output / "fh_profile.json").write_text(json.dumps({
            "seconds_per_sample": gpu_profiles,
            "cg_precision_is_part_of_cg_total": True,
        }, indent=2) + "\n")
    post_eval_normalize(
        recon_dir=str(recons), val_dir=args.val_dir,
        plot_dir=str(output / "comp_plots"), mask_select=args.mask_select,
    )


if __name__ == "__main__":
    main()
