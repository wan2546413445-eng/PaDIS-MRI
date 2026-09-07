"""CLI entry point for MCPP Stage-1 evaluation."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REFERENCE_ROOT = THIS_DIR.parent
# 参考代码可以放在仓库内，也可以通过 PADIS_REPO_ROOT 指向实际 PaDIS-MRI 根目录。
REPO_ROOT = Path(os.environ.get("PADIS_REPO_ROOT", str(REFERENCE_ROOT.parent))).resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

import dnnlib
from evaluator_mcpp import MCPPEvaluator


def _parse_indices(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="MCPP Stage-1 MRI reconstruction")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--psize", type=int, default=64)
    parser.add_argument("--mask_select", type=int, default=7)
    parser.add_argument("--val_count", type=int, default=32)
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--zeta", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=78)
    parser.add_argument("--inner_loops", type=int, default=10)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--fixed_seed_per_sample", action="store_true")
    parser.add_argument("--mcpp-lr", dest="mcpp_lr", type=float, default=1e-4)
    parser.add_argument("--save-mcpp-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 1:
        raise ValueError("MCPP Stage-1 uses exactly one GPU because phi is subject-specific state")
    if not torch.cuda.is_available():
        raise RuntimeError("The formal PaDIS MCPP evaluation requires CUDA")

    torch.cuda.set_device(args.gpus[0])
    device = torch.device(f"cuda:{args.gpus[0]}")

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as handle:
        model = pickle.load(handle)["ema"].to(device).eval()

    sample_indices = _parse_indices(args.sample_indices) or None
    evaluator = MCPPEvaluator(
        model=model,
        val_dir=args.val_dir,
        image_size=args.image_size,
        pad=args.pad,
        psize=args.psize,
        mask_select=args.mask_select,
        val_count=args.val_count,
        seed=args.seed,
        sample_indices=sample_indices,
        device=device,
    )

    rng_policy = "fixed_per_sample" if args.fixed_seed_per_sample else "continuous_stream"
    print(
        f"[MCPP Budget] steps={args.steps}, inner_loops={args.inner_loops}, "
        f"online_updates={args.steps * args.inner_loops}, mcpp_lr={args.mcpp_lr}, "
        f"rng={rng_policy}"
    )
    if sample_indices is not None:
        print(f"[MCPP Samples] {sample_indices}")

    result = evaluator.evaluate(
        save_dir=args.save_dir,
        zeta=args.zeta,
        num_steps=args.steps,
        inner_loops=args.inner_loops,
        mcpp_lr=args.mcpp_lr,
        save_mcpp_diagnostics=args.save_mcpp_diagnostics,
        fixed_seed_per_sample=args.fixed_seed_per_sample,
    )

    summary = result["metrics"].get("summary", {})
    if summary:
        print(f"PSNR:  {summary['psnr_mean']:.2f} +/- {summary['psnr_std']:.2f}")
        print(f"SSIM:  {summary['ssim_mean']:.4f} +/- {summary['ssim_std']:.4f}")
        print(f"NRMSE: {summary['nrmse_mean']:.4f} +/- {summary['nrmse_std']:.4f}")


if __name__ == "__main__":
    main()
