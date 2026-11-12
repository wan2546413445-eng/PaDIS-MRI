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
        "--mode", required=True,
        choices=[
            "dps", "cg_each", "cg_layer", "hybrid", "ct_cg_ddim",
            "mri_recurrent_cg", "mri_recurrent_hybrid",
        ],
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
    parser.add_argument(
        "--steps", type=int,
        help="Noise levels (default: 104 for MRI recurrent modes, 200 for ct_cg_ddim, 78 otherwise).",
    )
    parser.add_argument(
        "--inner_loops", type=int,
        help="Sweeps per level in legacy modes (default: 2 for ct_cg_ddim, 10 otherwise).",
    )
    parser.add_argument(
        "--ddim_eta", type=float, default=argparse.SUPPRESS,
        help="Cross-level noise mixing for ct_cg_ddim and MRI recurrent modes in [0, 1] (default: 0.8).",
    )
    parser.add_argument(
        "--profile_ct", action="store_true", default=argparse.SUPPRESS,
        help="Record patch denoising and CG CUDA intervals for ct_cg_ddim.",
    )
    parser.add_argument(
        "--diagnose_ct_cg", action="store_true", default=argparse.SUPPRESS,
        help="Save the final denoised fusion immediately before the last CT-style CG solve.",
    )
    parser.add_argument(
        "--recurrent_branches", type=int, default=argparse.SUPPRESS,
        help="Number of MRI recurrent branches (default: 1 for mri_recurrent_hybrid, 2 otherwise).",
    )
    parser.add_argument(
        "--recurrent_depth", type=int, default=argparse.SUPPRESS,
        help="MRI denoising rounds per branch (minimum: 2; default: 3 for mri_recurrent_hybrid, 2 otherwise).",
    )
    parser.add_argument("--profile_mri_recurrent", action="store_true", default=argparse.SUPPRESS,
                        help="Record the four MRI sampling sections with CUDA events.")
    parser.add_argument("--diagnose_mri_recurrent", action="store_true", default=argparse.SUPPRESS,
                        help="Save and evaluate the MRI sampler's final-level predictions.")
    parser.add_argument("--diagnose_mri_recurrent_hybrid", action="store_true", default=argparse.SUPPRESS,
                        help="Save the last recurrent predictions, CG handoff, and final DPS result.")
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
        help="Scale the CG increment (default: 0.1 for hybrid, 1.0 otherwise).",
    )
    parser.add_argument(
        "--dps_tail_steps", type=int,
        help="Final DPS levels (default: 10 for mri_recurrent_hybrid, 20 otherwise).",
    )
    parser.add_argument(
        "--dps_inner_loops", type=int, default=argparse.SUPPRESS,
        help="DPS sweeps per tail level in mri_recurrent_hybrid (default: 10).",
    )
    args = parser.parse_args()
    is_ct = args.mode == "ct_cg_ddim"
    is_mri_hybrid = args.mode == "mri_recurrent_hybrid"
    is_mri = is_mri_hybrid or args.mode == "mri_recurrent_cg"
    if args.dps_tail_steps is None:
        args.dps_tail_steps = 10 if is_mri_hybrid else 20
    if args.steps is None:
        args.steps = {"ct_cg_ddim": 200, "mri_recurrent_cg": 104, "mri_recurrent_hybrid": 104}.get(args.mode, 78)
    if is_mri_hybrid:
        args.dps_inner_loops = getattr(args, "dps_inner_loops", 10)
        args.diagnose_mri_recurrent_hybrid = getattr(args, "diagnose_mri_recurrent_hybrid", False)
        if args.dps_inner_loops < 1:
            parser.error("--dps_inner_loops must be positive.")
        if hasattr(args, "profile_mri_recurrent") or hasattr(args, "diagnose_mri_recurrent"):
            parser.error("--profile_mri_recurrent and --diagnose_mri_recurrent require --mode mri_recurrent_cg.")
    elif hasattr(args, "dps_inner_loops") or hasattr(args, "diagnose_mri_recurrent_hybrid"):
        parser.error("--dps_inner_loops and --diagnose_mri_recurrent_hybrid require --mode mri_recurrent_hybrid.")
    if is_mri:
        args.recurrent_branches = getattr(args, "recurrent_branches", 1 if is_mri_hybrid else 2)
        args.recurrent_depth = getattr(args, "recurrent_depth", 3 if is_mri_hybrid else 2)
        if not is_mri_hybrid:
            args.profile_mri_recurrent = getattr(args, "profile_mri_recurrent", False)
            args.diagnose_mri_recurrent = getattr(args, "diagnose_mri_recurrent", False)
        if args.recurrent_branches < 1:
            parser.error("MRI recurrent CG needs at least one branch.")
        if args.recurrent_depth < 2:
            parser.error("MRI recurrent CG needs at least two rounds per branch.")
        if args.inner_loops is not None:
            parser.error("Use --recurrent_branches and --recurrent_depth for MRI recurrent modes.")
        args.inner_loops = args.recurrent_branches * args.recurrent_depth
    elif any(hasattr(args, name) for name in (
        "recurrent_branches", "recurrent_depth", "profile_mri_recurrent", "diagnose_mri_recurrent",
    )):
        parser.error("MRI recurrent options require --mode mri_recurrent_cg or mri_recurrent_hybrid.")
    elif args.inner_loops is None:
        args.inner_loops = 2 if is_ct else 10
    if is_ct:
        args.profile_ct = getattr(args, "profile_ct", False)
        args.diagnose_ct_cg = getattr(args, "diagnose_ct_cg", False)
    elif hasattr(args, "profile_ct") or hasattr(args, "diagnose_ct_cg"):
        parser.error("--profile_ct and --diagnose_ct_cg require --mode ct_cg_ddim.")
    if is_ct or is_mri:
        args.ddim_eta = getattr(args, "ddim_eta", 0.8)
        if not 0 <= args.ddim_eta <= 1:
            parser.error("--ddim_eta must be in [0, 1].")
        if args.image_size < 1 or args.psize < 1 or args.pad < 1:
            parser.error(f"{args.mode} needs positive image_size, psize, and pad.")
        grid_width = (args.image_size // args.psize + 1) * args.psize
        if not args.image_size + args.pad <= grid_width <= args.image_size + args.pad + 1:
            parser.error(f"{args.mode} needs a patch grid that covers the FOV at every offset and fits in the padding.")
    elif hasattr(args, "ddim_eta"):
        parser.error("--ddim_eta requires --mode ct_cg_ddim, mri_recurrent_cg, or mri_recurrent_hybrid.")
    if args.cg_gain is None:
        args.cg_gain = 0.1 if args.mode == "hybrid" else 1.0
    if args.steps < 2 or args.inner_loops < 1:
        parser.error("Use at least two noise levels and one inner sweep.")
    if args.mode != "dps" and (
        args.cg_iters < 1 or not math.isfinite(args.cg_lambda) or args.cg_lambda <= 0
    ):
        parser.error("CG needs at least one iteration and a positive, finite cg_lambda.")
    if args.mode != "dps" and (
        not math.isfinite(args.cg_gain) or args.cg_gain <= 0
    ):
        parser.error("--cg_gain must be positive and finite.")
    if args.mode == "hybrid" and not 0 < args.dps_tail_steps < args.steps:
        parser.error("Hybrid needs 0 < --dps_tail_steps < --steps.")
    if is_mri_hybrid and not 0 < args.dps_tail_steps < args.steps:
        parser.error("mri_recurrent_hybrid needs 0 < --dps_tail_steps < --steps.")
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
    is_mri_cg = args.mode == "mri_recurrent_cg"
    is_mri_hybrid = args.mode == "mri_recurrent_hybrid"
    is_mri = is_mri_cg or is_mri_hybrid
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "eval")]

    import numpy as np
    import torch

    import dnnlib
    from evaluator_fixed_seed123 import DPSHyperEvaluator
    from eval_fast.fh_covariance import FHProfile, load_fh_cache
    from eval_fast.fh_diagnostics import FHDiagnostics
    from eval_fast.recon_fast import reconstruct, reconstruct_ct_cg_ddim, reconstruct_mri_recurrent_cg
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
    pre_last_cg_recons = None
    if args.mode == "ct_cg_ddim" and args.diagnose_ct_cg:
        pre_last_cg_recons = output / "pre_last_cg_recons"
        pre_last_cg_recons.mkdir()
    mri_diag_root = None
    if (is_mri_cg and args.diagnose_mri_recurrent) or (is_mri_hybrid and args.diagnose_mri_recurrent_hybrid):
        mri_diag_root = output / "mri_recurrent_diagnostics"
        mri_diag_root.mkdir()
    updates = args.steps * args.inner_loops
    dps_steps = {
        "dps": args.steps,
        "hybrid": args.dps_tail_steps,
        "mri_recurrent_hybrid": args.dps_tail_steps,
    }.get(args.mode, 0)
    dps_updates = dps_steps * (args.dps_inner_loops if is_mri_hybrid else args.inner_loops)
    recurrent_layers = args.steps - dps_steps
    if is_mri_hybrid:
        updates = recurrent_layers * args.inner_loops + dps_updates
    cg_solves = {
        "dps": 0,
        "cg_each": updates,
        "cg_layer": args.steps,
        "hybrid": (args.steps - dps_steps) * args.inner_loops,
        "ct_cg_ddim": args.steps,
        "mri_recurrent_cg": args.steps - 1,
        "mri_recurrent_hybrid": recurrent_layers,
    }
    config = {
        **vars(args),
        **fh_config,
        "dps_tail_steps": dps_steps if args.mode == "hybrid" or is_mri_hybrid else 0,
        "rng_policy": "fixed_per_sample",
        "sample_indices": evaluator.val_indices,
        "gpu": torch.cuda.get_device_name(0),
        "patch_batch": os.getenv("PADIS_PATCH_BATCH"),
        "patch_checkpoint": os.getenv("PADIS_PATCH_CHECKPOINT", "0"),
        "patch_sweeps_per_sample": updates,
        "dps_backprops_per_sample": dps_updates,
        "cg_solves_per_sample": cg_solves[args.mode],
    }
    if is_mri:
        config.update(noise_fusion="terminal_mean", final_cg=False)
    if is_mri_hybrid:
        config.update(recurrent_layers=recurrent_layers, handoff="clean_cg_to_original_dps")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    elapsed = {}
    operations = {}
    gpu_profiles = {}
    profile_ct = args.mode == "ct_cg_ddim" and args.profile_ct
    profile_mri = is_mri_cg and args.profile_mri_recurrent
    for index in evaluator.val_indices:
        evaluator._set_sample_seed(idx=index, gpu_id=0)
        evaluator._reset_sample_latent(gpu_id=0)
        measurement, clean, inverseop = evaluator._load_sample(index)
        counts = dict.fromkeys((
            "main_denoise", "probe_denoise", "dps_backprops", "cg_solves",
            "cg_iterations", "precision_applies", "covariance_transfers",
            "bfgs_accepted", "bfgs_skipped",
        ), 0)
        if is_mri:
            counts.update(recurrent_branches=args.recurrent_branches, recurrent_depth=args.recurrent_depth)
        profile = None
        if args.profile_fh:
            profile = FHProfile()
        elif profile_ct:
            profile = FHProfile(sections=("denoise", "cg_total"))
        elif profile_mri:
            profile = FHProfile(sections=("denoise", "recurrent_fusion", "cg_total", "ve_ddim"))
        diagnostics = FHDiagnostics(args.steps - dps_steps) if args.diagnose_fh else None
        mri_diagnostics = {} if mri_diag_root is not None else None
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
            ct_diagnostics = {} if args.diagnose_ct_cg else None
            recon = reconstruct_ct_cg_ddim(
                model, evaluator.latents_pos, inverseop, measurement,
                num_steps=args.steps, inner_loops=args.inner_loops,
                pad=args.pad, psize=args.psize, ddim_eta=args.ddim_eta,
                cg_lambda=args.cg_lambda, cg_iters=args.cg_iters, cg_gain=args.cg_gain,
                counts=counts, profile=profile, diagnostics=ct_diagnostics,
            )
        elif is_mri:
            recon = reconstruct_mri_recurrent_cg(
                model, evaluator.latents_pos, inverseop, measurement,
                num_steps=args.steps, recurrent_branches=args.recurrent_branches,
                recurrent_depth=args.recurrent_depth,
                pad=args.pad, psize=args.psize, ddim_eta=args.ddim_eta,
                cg_lambda=args.cg_lambda, cg_iters=args.cg_iters, cg_gain=args.cg_gain,
                dps_tail_steps=dps_steps, dps_inner_loops=getattr(args, "dps_inner_loops", 10),
                zeta=args.zeta,
                counts=counts, profile=profile, diagnostics=mri_diagnostics,
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
        if pre_last_cg_recons is not None:
            np.save(
                pre_last_cg_recons / f"recon_patch_{index}.npy",
                ct_diagnostics["pre_last_cg"].numpy(),
            )
        if diagnostics:
            diagnostics.save(output, index, clean, recon_array)
        if mri_diagnostics is not None:
            for name, prediction in mri_diagnostics.items():
                stage_recons = mri_diag_root / name / "recons"
                stage_recons.mkdir(parents=True, exist_ok=True)
                np.save(stage_recons / f"recon_patch_{index}.npy", prediction.cpu().numpy())

    timing = {
        "recon_seconds_per_sample": elapsed,
        "mean_recon_seconds": sum(elapsed.values()) / len(elapsed),
    }
    if profile_ct or profile_mri:
        timing["gpu_seconds_per_sample"] = gpu_profiles
    if is_mri:
        timing["operations_per_sample"] = operations
        config["recon_seconds_per_sample"] = elapsed
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
    if pre_last_cg_recons is not None:
        post_eval_normalize(
            recon_dir=str(pre_last_cg_recons), val_dir=args.val_dir,
            plot_dir=str(output / "pre_last_cg_plots"), mask_select=args.mask_select,
        )
    if mri_diag_root is not None:
        for stage in sorted(mri_diag_root.iterdir()):
            post_eval_normalize(
                recon_dir=str(stage / "recons"), val_dir=args.val_dir,
                plot_dir=str(stage / "comp_plots"), mask_select=args.mask_select,
            )


if __name__ == "__main__":
    main()
