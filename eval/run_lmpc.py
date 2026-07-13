#!/usr/bin/env python3
"""
Standalone LMPC evaluation runner.

Run from the PaDIS-MRI repository root:
    python eval/run_lmpc.py ...

The original eval/run.py is not modified.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import dnnlib

from evaluator_lmpc import DPSHyperEvaluatorLMPC
from utils import post_eval_normalize


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "PaDIS-MRI Late-Stage Multi-Partition Consensus evaluation"
        )
    )

    parser.add_argument("--run_evaluate", action="store_true")
    parser.add_argument("--algo", choices=["padis"], default="padis")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--psize", type=int, default=64)
    parser.add_argument("--mask_select", type=int, default=7)
    parser.add_argument("--val_count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Comma-separated exact sample indices, e.g. 3,7,23",
    )

    parser.add_argument("--zeta", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=78)
    parser.add_argument("--inner_loops", type=int, default=10)
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument("--report_every", type=int, default=1)

    parser.add_argument(
        "--late_consensus_steps",
        type=int,
        default=10,
        help="Number of final outer steps using LMPC. 0 disables LMPC.",
    )
    parser.add_argument(
        "--late_consensus_k",
        type=int,
        choices=[1, 2, 4],
        default=4,
        help="Partitions evaluated at the same state in the LMPC region.",
    )
    parser.add_argument(
        "--late_consensus_offset_mode",
        choices=["random_complementary", "fixed"],
        default="random_complementary",
        help=(
            "random_complementary uses a random anchor plus half-patch "
            "shifts; fixed uses deterministic offsets."
        ),
    )
    parser.add_argument(
        "--save_consensus_diagnostics",
        action="store_true",
        help="Save per-step partition-disagreement CSV files.",
    )

    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate reconstruction diagnostics.",
    )
    parser.add_argument(
        "--intermediate_every",
        type=int,
        default=10,
    )

    return parser.parse_args()


def parse_int_list(value: str):
    if not value:
        return []
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def print_budget(args):
    base_forwards = args.steps * args.inner_loops
    active_steps = (
        args.late_consensus_steps
        if args.late_consensus_k > 1
        else 0
    )
    extra_forwards = (
        active_steps
        * args.inner_loops
        * max(args.late_consensus_k - 1, 0)
    )
    total_forwards = base_forwards + extra_forwards
    relative_cost = total_forwards / max(base_forwards, 1)

    print(
        "[PaDIS-LMPC Budget] "
        f"steps={args.steps}, "
        f"inner_loops={args.inner_loops}, "
        f"base_partition_forwards={base_forwards}, "
        f"late_consensus_steps={args.late_consensus_steps}, "
        f"late_consensus_k={args.late_consensus_k}, "
        f"total_partition_forwards={total_forwards}, "
        f"relative_cost={relative_cost:.3f}x"
    )


def print_normalized_summary(results_path: Path):
    if not results_path.is_file():
        print(
            "[LMPC] Normalized results.json was not found at "
            f"{results_path}"
        )
        return

    with results_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    summary = payload.get("summary", {})
    if not summary:
        print(f"[LMPC] No summary section in {results_path}")
        return

    print("[LMPC] Author-normalized final metrics")
    for metric in ("psnr", "ssim", "nrmse"):
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in summary:
            if std_key in summary:
                print(
                    f"  {metric.upper()}: "
                    f"{summary[mean_key]:.6f} ± "
                    f"{summary[std_key]:.6f}"
                )
            else:
                print(
                    f"  {metric.upper()}: "
                    f"{summary[mean_key]:.6f}"
                )


def main():
    args = parse_args()

    if not args.run_evaluate:
        raise ValueError("Add --run_evaluate to start evaluation.")
    if not 0 <= args.late_consensus_steps <= args.steps:
        raise ValueError(
            "--late_consensus_steps must be between 0 and --steps"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    print_budget(args)

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as file:
        model = pickle.load(file)["ema"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()

    sample_indices = parse_int_list(args.sample_indices)

    evaluator = DPSHyperEvaluatorLMPC(
        model=model,
        mask_select=args.mask_select,
        val_dir=args.val_dir,
        image_size=args.image_size,
        pad=args.pad,
        psize=args.psize,
        val_count=args.val_count,
        seed=args.seed,
        sample_indices=sample_indices if sample_indices else None,
        late_consensus_steps=args.late_consensus_steps,
        late_consensus_k=args.late_consensus_k,
        late_consensus_offset_mode=args.late_consensus_offset_mode,
        save_consensus_diagnostics=args.save_consensus_diagnostics,
    )

    evaluator.evaluate(
        zeta=args.zeta,
        num_steps=args.steps,
        inner_loops=args.inner_loops,
        pad=args.pad,
        psize=args.psize,
        algo="padis",
        save_dir=os.path.join(args.save_dir, "evaluate"),
        tag="patch_lmpc",
        gpus=args.gpus,
        report_every=args.report_every,
        save_intermediate=args.save_intermediate,
        intermediate_every=args.intermediate_every,
    )

    recon_dir = os.path.join(
        args.save_dir, "evaluate", "recons"
    )
    plot_dir = os.path.join(
        args.save_dir, "evaluate", "comp_plots"
    )

    try:
        post_eval_normalize(
            recon_dir=recon_dir,
            val_dir=args.val_dir,
            plot_dir=plot_dir,
            json_basename="results",
            mask_select=args.mask_select,
        )
    except Exception as error:
        print(
            "[post_eval_normalize] Failed: "
            f"{type(error).__name__}: {error}"
        )
        raise

    print_normalized_summary(Path(plot_dir) / "results.json")


if __name__ == "__main__":
    main()
