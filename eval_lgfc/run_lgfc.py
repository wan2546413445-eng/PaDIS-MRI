"""Stage-1 PaDIS-MRI runner for late-stage global frequency correction."""

import argparse
from dataclasses import asdict
import json
import os
import pickle
import sys
import time

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EVAL_DIR = os.path.join(_REPO_ROOT, "eval")
_TRAIN_DIR = os.path.join(_REPO_ROOT, "train", "padis-mri")
# The EMA checkpoint may pickle classes as ``training.*``. Add the original
# PaDIS training package before unpickling, while keeping eval imports intact.
for _path in (_TRAIN_DIR, _REPO_ROOT, _EVAL_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import dnnlib
from utils import post_eval_normalize
from eval_lgfc.evaluator_lgfc import LgfcDPSHyperEvaluator
from eval_lgfc.frequency_correction import LGFCConfig, validate_lgfc_config
from eval_lgfc.recon_lgfc import get_lgfc_runtime_peaks, reset_lgfc_runtime_peaks


def parse_args():
    parser = argparse.ArgumentParser(description="PaDIS-MRI LGFC stage-1 runner")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--psize", type=int, default=64)
    parser.add_argument("--mask_select", type=int, default=7)
    parser.add_argument("--val_count", type=int, default=1)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--zeta", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=78)
    parser.add_argument("--inner_loops", type=int, default=10)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument("--run_evaluate", action="store_true")
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--intermediate_every", type=int, default=10)

    parser.add_argument("--enable_lgfc", action="store_true")
    parser.add_argument("--reference_mode", choices=["last", "ema"], default="ema")
    parser.add_argument("--ema_beta", type=float, default=0.8)
    parser.add_argument("--global_start_outer", type=int, default=40)
    parser.add_argument("--global_every", type=int, default=1)
    parser.add_argument("--unsampled_weight", type=float, default=0.25)
    parser.add_argument("--max_relative_update", type=float, default=0.05)
    parser.add_argument("--dc_growth_limit", type=float, default=1.02)
    parser.add_argument("--backtrack_factor", type=float, default=0.5)
    parser.add_argument("--max_backtracks", type=int, default=3)
    parser.add_argument("--save_lgfc_diagnostics", action="store_true")
    parser.add_argument("--profile_memory", action="store_true")
    return parser.parse_args()


def parse_indices(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _stage1_validate(args) -> None:
    if args.image_size != 384:
        raise ValueError("LGFC stage 1 fixes --image_size at 384")
    if args.steps != 78:
        raise ValueError("LGFC stage 1 fixes --steps at 78")
    if args.inner_loops != 10:
        raise ValueError("LGFC stage 1 fixes --inner_loops at 10")
    if args.pad != 64 or args.psize != 64:
        raise ValueError("LGFC stage 1 fixes --pad and --psize at 64")
    if args.mask_select != 7:
        raise ValueError("LGFC stage 1 fixes --mask_select at 7")
    if args.zeta != 3.0:
        raise ValueError("LGFC stage 1 fixes --zeta at 3")
    if args.seed != 123:
        raise ValueError("LGFC stage 1 fixes --seed at 123")
    if args.val_count != 1:
        raise ValueError("LGFC stage 1 fixes --val_count at 1")
    if not args.run_evaluate:
        raise ValueError("LGFC stage 1 requires --run_evaluate")
    sample_indices = parse_indices(args.sample_indices)
    if len(sample_indices) != 1:
        raise ValueError("LGFC stage 1 requires exactly one explicit sample index")
    if args.gpus is not None and len(args.gpus) != 1:
        raise ValueError("LGFC stage 1 must run serially on exactly one GPU")
    if args.enable_lgfc and not 1 <= args.global_start_outer <= args.steps:
        raise ValueError("LGFC requires 1 <= --global_start_outer <= --steps")


def main():
    args = parse_args()
    _stage1_validate(args)
    os.makedirs(args.save_dir, exist_ok=True)

    sample_indices = parse_indices(args.sample_indices)
    lgfc_config = validate_lgfc_config(
        LGFCConfig(
            enabled=args.enable_lgfc,
            reference_mode=args.reference_mode,
            ema_beta=args.ema_beta,
            start_outer=args.global_start_outer,
            correction_every=args.global_every,
            unsampled_weight=args.unsampled_weight,
            max_relative_update=args.max_relative_update,
            dc_growth_limit=args.dc_growth_limit,
            backtrack_factor=args.backtrack_factor,
            max_backtracks=args.max_backtracks,
        )
    )
    run_config = {
        "model_path": args.model_path,
        "val_dir": args.val_dir,
        "save_dir": args.save_dir,
        "image_size": args.image_size,
        "pad": args.pad,
        "psize": args.psize,
        "val_count": args.val_count,
        "sample_indices": sample_indices,
        "mask_select": args.mask_select,
        "seed": args.seed,
        "steps": args.steps,
        "inner_loops": args.inner_loops,
        "total_updates": args.steps * args.inner_loops,
        "enable_lgfc": args.enable_lgfc,
        "reference_mode": args.reference_mode,
        "ema_beta": args.ema_beta,
        "global_start_outer": args.global_start_outer,
        "global_every": args.global_every,
        "unsampled_weight": args.unsampled_weight,
        "max_relative_update": args.max_relative_update,
        "dc_growth_limit": args.dc_growth_limit,
        "backtrack_factor": args.backtrack_factor,
        "max_backtracks": args.max_backtracks,
        "lgfc_config": asdict(lgfc_config),
        "save_lgfc_diagnostics": args.save_lgfc_diagnostics,
        "profile_memory": args.profile_memory,
        "run_evaluate": args.run_evaluate,
        "gpus": args.gpus,
    }
    print(json.dumps(run_config, indent=2))
    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as handle:
        model = pickle.load(handle)["ema"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    evaluator = LgfcDPSHyperEvaluator(
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
    evaluator.lgfc_config = lgfc_config
    evaluator.save_lgfc_diagnostics = args.save_lgfc_diagnostics
    evaluator.profile_memory = args.profile_memory

    print(
        f"[PaDIS Budget] steps={args.steps}, inner_loops={args.inner_loops}, "
        f"total_updates={args.steps * args.inner_loops}"
    )
    profile_device = None
    if torch.cuda.is_available():
        gpu_id = args.gpus[0] if args.gpus else torch.cuda.current_device()
        profile_device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.synchronize(profile_device)
        torch.cuda.reset_peak_memory_stats(profile_device)
    reset_lgfc_runtime_peaks()

    metrics = None
    start_time = time.perf_counter()
    try:
        metrics = evaluator.evaluate(
            zeta=args.zeta,
            num_steps=args.steps,
            inner_loops=args.inner_loops,
            pad=args.pad,
            psize=args.psize,
            algo="padis",
            save_dir=os.path.join(args.save_dir, "evaluate"),
            tag="patch",
            gpus=args.gpus,
            report_every=1,
            save_intermediate=args.save_intermediate,
            intermediate_every=args.intermediate_every,
        )
    finally:
        if profile_device is not None:
            torch.cuda.synchronize(profile_device)
            run_peak_allocated_mb = torch.cuda.max_memory_allocated(profile_device) / (
                1024.0 ** 2
            )
            run_peak_reserved_mb = torch.cuda.max_memory_reserved(profile_device) / (
                1024.0 ** 2
            )
            preserved_peaks = get_lgfc_runtime_peaks()
            run_peak_allocated_mb = max(
                run_peak_allocated_mb,
                preserved_peaks["run_peak_memory_allocated_mb"],
            )
            run_peak_reserved_mb = max(
                run_peak_reserved_mb,
                preserved_peaks["run_peak_memory_reserved_mb"],
            )
        else:
            run_peak_allocated_mb = None
            run_peak_reserved_mb = None
        runtime_profile = {
            "total_wall_time_sec": time.perf_counter() - start_time,
            "run_peak_memory_allocated_mb": run_peak_allocated_mb,
            "run_peak_memory_reserved_mb": run_peak_reserved_mb,
            "cuda_available": torch.cuda.is_available(),
            "profile_device": str(profile_device) if profile_device is not None else None,
        }
        with open(
            os.path.join(args.save_dir, "runtime_profile.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(runtime_profile, handle, indent=2)
        print(json.dumps(runtime_profile, indent=2))

    print(json.dumps(metrics.get("summary", {}), indent=2))

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
    except Exception as error:
        print(f"[post_eval_normalize] Skipped due to error: {error}")


if __name__ == "__main__":
    main()
