"""Command-line entry point for PaDIS-MRI projected-target Scan-TTA."""

import argparse
import os
import pickle
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("PADIS_REPO_ROOT", str(THIS_DIR.parent))).resolve()
for import_path in (REPO_ROOT, REPO_ROOT / "eval", REPO_ROOT / "train" / "padis-mri"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import torch
import dnnlib

try:
    from .evaluator_tta import ScanTTAEvaluator
    from .scan_tta import refinement_diffusion_indices
except ImportError:
    from evaluator_tta import ScanTTAEvaluator
    from scan_tta import refinement_diffusion_indices


def _parse_indices(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="MRI projected-target Scan-TTA")
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
    parser.add_argument("--tta-interval", dest="tta_interval", type=int, default=10)
    parser.add_argument(
        "--tta-max-diffusion", dest="tta_max_diffusion", type=int, default=None,
        help="Only refine at selected K-interval diffusion indices <= this value.",
    )
    parser.add_argument("--refinement-iters", dest="refinement_iters", type=int, default=5)
    parser.add_argument("--cg-iters", dest="cg_iters", type=int, default=5)
    parser.add_argument("--tta-lr", dest="tta_lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 1:
        raise ValueError("Scan-TTA requires exactly one GPU for subject-specific state")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal PaDIS Scan-TTA evaluation requires CUDA")

    torch.cuda.set_device(args.gpus[0])
    device = torch.device(f"cuda:{args.gpus[0]}")
    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as handle:
        model = pickle.load(handle)["ema"].to(device).eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    event_indices = refinement_diffusion_indices(args.steps, args.tta_interval)
    if args.tta_max_diffusion is not None:
        event_indices = [
            index for index in event_indices
            if index <= args.tta_max_diffusion
        ]
    print(
        f"[Scan-TTA] scope=full_network | optimizer=Adam | lr={args.tta_lr:g} | "
        "weight_decay=0"
    )
    print(
        f"[Scan-TTA] full_network_parameters={parameter_count:,} | "
        f"event_diffusion_indices={event_indices} | "
        f"tta_max_diffusion={args.tta_max_diffusion}"
    )
    print(
        f"[Budget] baseline_denoiser_calls={args.steps * args.inner_loops} | "
        f"tta_denoiser_calls={len(event_indices) * args.refinement_iters} | "
        f"tta_backprops={len(event_indices) * args.refinement_iters} | "
        f"cg_iterations={len(event_indices) * args.refinement_iters * args.cg_iters}"
    )

    evaluator = ScanTTAEvaluator(
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
        zeta=args.zeta,
        num_steps=args.steps,
        inner_loops=args.inner_loops,
        tta_interval=args.tta_interval,
        tta_max_diffusion=args.tta_max_diffusion,
        refinement_iters=args.refinement_iters,
        cg_iters=args.cg_iters,
        fixed_seed_per_sample=args.fixed_seed_per_sample,
    )
    summary = result["metrics"].get("summary", {})
    if summary:
        print(f"PSNR:  {summary['psnr_mean']:.2f} +/- {summary['psnr_std']:.2f}")
        print(f"SSIM:  {summary['ssim_mean']:.4f} +/- {summary['ssim_std']:.4f}")
        print(f"NRMSE: {summary['nrmse_mean']:.4f} +/- {summary['nrmse_std']:.4f}")


if __name__ == "__main__":
    main()
