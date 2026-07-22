#!/usr/bin/env python3
"""Standalone runner for PaDIS-MRI late-stage prior scaling."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train" / "padis-mri"
EVAL_ROOT = REPO_ROOT / "eval"
sys.path.insert(0, str(TRAIN_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_ROOT))

import dnnlib
from evaluator_late_prior_scale import DPSHyperEvaluatorLatePriorScale
from utils import post_eval_normalize


def parse_args():
    p = argparse.ArgumentParser(description="PaDIS-MRI late-stage prior scaling")
    p.add_argument("--run_evaluate", action="store_true")
    p.add_argument("--algo", choices=["padis"], default="padis")
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--psize", type=int, default=64)
    p.add_argument("--mask_select", type=int, default=7)
    p.add_argument("--val_count", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sample_indices", default="")
    p.add_argument("--zeta", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=78)
    p.add_argument("--inner_loops", type=int, default=10)
    p.add_argument("--gpus", type=int, nargs="+", default=None)
    p.add_argument("--report_every", type=int, default=1)
    p.add_argument("--late_prior_steps", type=int, default=10)
    p.add_argument("--late_prior_scale", type=float, default=1.0)
    p.add_argument("--save_scale_diagnostics", action="store_true")
    p.add_argument("--save_intermediate", action="store_true")
    p.add_argument("--intermediate_every", type=int, default=10)
    return p.parse_args()


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()] if value else []


def print_summary(path):
    path = Path(path)
    if not path.is_file():
        print(f"[Late Prior Scale] Missing results: {path}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    s = payload.get("summary", {})
    if not s:
        print(f"[Late Prior Scale] No summary in {path}")
        return
    print("[Late Prior Scale] Author-normalized final metrics")
    print(f"  PSNR: {s['psnr_mean']:.6f} ± {s['psnr_std']:.6f}")
    print(f"  SSIM: {s['ssim_mean']:.6f} ± {s['ssim_std']:.6f}")
    print(f"  NRMSE: {s['nrmse_mean']:.6f} ± {s['nrmse_std']:.6f}")


def main():
    args = parse_args()
    if not args.run_evaluate:
        raise ValueError("Add --run_evaluate")
    if not 0 <= args.late_prior_steps <= args.steps:
        raise ValueError("--late_prior_steps must be between 0 and --steps")
    if args.late_prior_scale < 0:
        raise ValueError("--late_prior_scale must be non-negative")

    os.makedirs(args.save_dir, exist_ok=True)
    print(
        "[Late Prior Scale] "
        f"steps={args.steps}, inner_loops={args.inner_loops}, "
        f"last_outer_steps={args.late_prior_steps}, "
        f"beta={args.late_prior_scale:g}, relative_network_cost=1.000x"
    )

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as f:
        model = pickle.load(f)["ema"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    indices = parse_int_list(args.sample_indices)
    evaluator = DPSHyperEvaluatorLatePriorScale(
        model=model,
        mask_select=args.mask_select,
        val_dir=args.val_dir,
        image_size=args.image_size,
        pad=args.pad,
        psize=args.psize,
        val_count=args.val_count,
        seed=args.seed,
        sample_indices=indices if indices else None,
        late_prior_steps=args.late_prior_steps,
        late_prior_scale=args.late_prior_scale,
        save_scale_diagnostics=args.save_scale_diagnostics,
    )

    evaluator.evaluate(
        zeta=args.zeta,
        num_steps=args.steps,
        inner_loops=args.inner_loops,
        pad=args.pad,
        psize=args.psize,
        algo="padis",
        save_dir=os.path.join(args.save_dir, "evaluate"),
        tag="patch",
        gpus=args.gpus,
        report_every=args.report_every,
        save_intermediate=args.save_intermediate,
        intermediate_every=args.intermediate_every,
    )

    recon_dir = os.path.join(args.save_dir, "evaluate", "recons")
    plot_dir = os.path.join(args.save_dir, "evaluate", "comp_plots")
    post_eval_normalize(
        recon_dir=recon_dir,
        val_dir=args.val_dir,
        plot_dir=plot_dir,
        json_basename="results",
        mask_select=args.mask_select,
    )
    print_summary(Path(plot_dir) / "results.json")


if __name__ == "__main__":
    main()
