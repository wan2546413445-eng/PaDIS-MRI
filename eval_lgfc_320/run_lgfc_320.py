"""Runner for 320-ROI PaDIS inference with unchanged 384 MRI measurements."""

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
for _path in (_TRAIN_DIR, _REPO_ROOT, _EVAL_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import dnnlib
from eval_lgfc.frequency_correction import LGFCConfig, validate_lgfc_config
from eval_lgfc_320.evaluator_lgfc_320 import Lgfc320DPSHyperEvaluator
from eval_lgfc_320.utils_320 import post_eval_normalize_320


def parse_args():
    p = argparse.ArgumentParser(
        description="PaDIS-MRI 320-ROI / 384-measurement LGFC experiment"
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=320)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--psize", type=int, default=64)
    p.add_argument("--mask_select", type=int, default=7)
    p.add_argument("--val_count", type=int, default=1)
    p.add_argument("--sample_indices", type=str, required=True)
    p.add_argument("--zeta", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=78)
    p.add_argument("--inner_loops", type=int, default=10)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--gpus", type=int, nargs="+", default=None)

    p.add_argument("--enable_lgfc", action="store_true")
    p.add_argument("--reference_mode", choices=["last", "ema"], default="ema")
    p.add_argument("--ema_beta", type=float, default=0.8)
    p.add_argument("--global_start_outer", type=int, default=40)
    p.add_argument("--global_every", type=int, default=1)
    p.add_argument("--unsampled_weight", type=float, default=0.25)
    p.add_argument("--max_relative_update", type=float, default=0.05)
    p.add_argument("--dc_growth_limit", type=float, default=1.02)
    p.add_argument("--backtrack_factor", type=float, default=0.5)
    p.add_argument("--max_backtracks", type=int, default=3)
    return p.parse_args()


def parse_indices(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def validate_args(args):
    if args.image_size != 320:
        raise ValueError("LGFC-320 fixes the PaDIS target at 320")
    if args.pad != 64 or args.psize != 64:
        raise ValueError("LGFC-320 fixes pad=64 and psize=64")
    if args.steps != 78 or args.inner_loops != 10:
        raise ValueError("LGFC-320 pilot fixes 78x10 posterior updates")
    if args.zeta != 3.0:
        raise ValueError("LGFC-320 pilot fixes zeta=3")
    if args.seed != 123:
        raise ValueError("LGFC-320 pilot fixes seed=123")
    if args.val_count != 1:
        raise ValueError("LGFC-320 pilot runs one explicit sample at a time")
    if args.mask_select not in (4, 7):
        raise ValueError("LGFC-320 supports mask_select 4 or 7")
    indices = parse_indices(args.sample_indices)
    if len(indices) != 1:
        raise ValueError("Provide exactly one --sample_indices value")
    if args.gpus is not None and len(args.gpus) != 1:
        raise ValueError("LGFC-320 must run serially on one GPU")
    return indices


def main():
    args = parse_args()
    sample_indices = validate_args(args)
    os.makedirs(args.save_dir, exist_ok=True)

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

    config = {
        "experiment": "320_roi_with_original_384_measurement",
        "measurement_size": 384,
        "target_size": 320,
        "canvas_size": 448,
        "patches_per_update": 36,
        "baseline_patches_per_update": 49,
        "model_path": args.model_path,
        "val_dir": args.val_dir,
        "sample_indices": sample_indices,
        "mask_select": args.mask_select,
        "steps": args.steps,
        "inner_loops": args.inner_loops,
        "total_updates": args.steps * args.inner_loops,
        "pad": args.pad,
        "psize": args.psize,
        "zeta": args.zeta,
        "seed": args.seed,
        "lgfc": asdict(lgfc_config),
    }
    print(json.dumps(config, indent=2))
    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as handle:
        model = pickle.load(handle)["ema"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    evaluator = Lgfc320DPSHyperEvaluator(
        model=model,
        mask_select=args.mask_select,
        val_dir=args.val_dir,
        image_size=320,
        pad=64,
        psize=64,
        val_count=1,
        seed=123,
        sample_indices=sample_indices,
    )
    evaluator.lgfc_config = lgfc_config

    profile_device = None
    if torch.cuda.is_available():
        gpu_id = args.gpus[0] if args.gpus else torch.cuda.current_device()
        profile_device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.synchronize(profile_device)
        torch.cuda.reset_peak_memory_stats(profile_device)

    start = time.perf_counter()
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
    )

    if profile_device is not None:
        torch.cuda.synchronize(profile_device)
        peak_allocated_mb = torch.cuda.max_memory_allocated(profile_device) / 1024.0 ** 2
        peak_reserved_mb = torch.cuda.max_memory_reserved(profile_device) / 1024.0 ** 2
    else:
        peak_allocated_mb = peak_reserved_mb = None

    runtime = {
        "total_wall_time_sec": time.perf_counter() - start,
        "peak_memory_allocated_mb": peak_allocated_mb,
        "peak_memory_reserved_mb": peak_reserved_mb,
    }
    with open(os.path.join(args.save_dir, "runtime_profile.json"), "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)

    print(json.dumps(metrics.get("summary", {}), indent=2))
    print(json.dumps(runtime, indent=2))

    post_eval_normalize_320(
        recon_dir=os.path.join(args.save_dir, "evaluate", "recons"),
        val_dir=args.val_dir,
        output_dir=os.path.join(args.save_dir, "evaluate", "comp_plots_320roi"),
        json_basename="results_320roi",
    )


if __name__ == "__main__":
    main()
