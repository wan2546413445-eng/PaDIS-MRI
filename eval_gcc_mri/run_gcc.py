"""Command-line entry point for Global Cross-Conditioned PaDIS."""

import argparse
import os
import pickle
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("PADIS_REPO_ROOT", str(THIS_DIR.parent))
).resolve()
for import_path in (
    REPO_ROOT,
    REPO_ROOT / "eval",
    REPO_ROOT / "train" / "padis-mri",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import torch
import dnnlib

try:
    from .evaluator_gcc import GCCEvaluator
    from .recon_gcc import GCC_TTA_EVENTS, formal_budget
except ImportError:
    from evaluator_gcc import GCCEvaluator
    from recon_gcc import GCC_TTA_EVENTS, formal_budget


def _parse_indices(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global Cross-Conditioned PaDIS with strong CG data conditioning"
    )
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
    parser.add_argument("--steps", type=int, default=78)
    parser.add_argument("--inner_loops", type=int, default=10)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--fixed_seed_per_sample", action="store_true")
    parser.add_argument("--refinement-iters", type=int, default=5)
    parser.add_argument("--cg-iters", type=int, default=5)
    parser.add_argument("--tta-lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 1:
        raise ValueError("GCC-PaDIS requires exactly one GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal GCC-PaDIS evaluation requires CUDA")
    if (
        args.image_size != 384
        or args.pad != 64
        or args.psize != 64
        or args.mask_select != 7
    ):
        raise ValueError("Formal method is fixed to image384/pad64/patch64/mask7")
    if (
        args.steps != 78
        or args.inner_loops != 10
        or args.refinement_iters != 5
        or args.cg_iters != 5
        or args.tta_lr != 1e-5
    ):
        raise ValueError(
            "Formal method is fixed to 78x10, TTA 5/event, CG 5, lr 1e-5"
        )

    torch.cuda.set_device(args.gpus[0])
    device = torch.device(f"cuda:{args.gpus[0]}")

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as handle:
        model = pickle.load(handle)["ema"].to(device).eval()

    budget = formal_budget(
        num_steps=args.steps,
        inner_loops=args.inner_loops,
        events=GCC_TTA_EVENTS,
        refinement_iters=args.refinement_iters,
        cg_iters=args.cg_iters,
    )
    print(
        "[GCC-PaDIS] holdout TTA -> patch prior -> hard full CG "
        "-> conditional score"
    )
    print(
        f"[GCC-PaDIS] events={list(GCC_TTA_EVENTS)} "
        f"| refinement/event={args.refinement_iters} "
        f"| lr={args.tta_lr:g} | cg_iters={args.cg_iters}"
    )
    print(
        "[GCC-PaDIS] conditioning=A^H A z=A^H y, "
        "initialized from patch diffusion estimate"
    )
    print(
        f"[Budget] denoisers={budget['denoiser_calls']} "
        f"| TTA backprops={budget['tta_backprops']} "
        f"| full CG calls={budget['full_cg_calls']} "
        f"| total CG iterations={budget['total_cg_iterations']}"
    )

    evaluator = GCCEvaluator(
        model=model,
        val_dir=args.val_dir,
        image_size=args.image_size,
        pad=args.pad,
        psize=args.psize,
        mask_select=args.mask_select,
        val_count=args.val_count,
        seed=args.seed,
        sample_indices=_parse_indices(args.sample_indices) or None,
        tta_lr=args.tta_lr,
        device=device,
    )
    result = evaluator.evaluate(
        save_dir=args.save_dir,
        num_steps=args.steps,
        inner_loops=args.inner_loops,
        refinement_iters=args.refinement_iters,
        cg_iters=args.cg_iters,
        fixed_seed_per_sample=args.fixed_seed_per_sample,
    )

    summary = result["metrics"].get("summary", {})
    if summary:
        print(
            f"PSNR: {summary['psnr_mean']:.2f} +/- {summary['psnr_std']:.2f}"
        )
        print(
            f"SSIM: {summary['ssim_mean']:.4f} +/- {summary['ssim_std']:.4f}"
        )
        print(
            f"NRMSE: {summary['nrmse_mean']:.4f} +/- {summary['nrmse_std']:.4f}"
        )


if __name__ == "__main__":
    main()
