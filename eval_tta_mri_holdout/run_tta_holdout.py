"""Formal S200-10k sample runner for k-space holdout Scan-TTA."""

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

import dnnlib
import torch

try:
    from .evaluator_tta_holdout import HoldoutScanTTAEvaluator
    from .scan_tta_holdout import formal_event_indices
except ImportError:
    from evaluator_tta_holdout import HoldoutScanTTAEvaluator
    from scan_tta_holdout import formal_event_indices


def _parse_indices(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Formal PaDIS-MRI mask7 k-space holdout Scan-TTA"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--sample_indices", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 1:
        raise ValueError("Formal Holdout Scan-TTA requires exactly one GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Holdout Scan-TTA requires CUDA")

    torch.cuda.set_device(args.gpus[0])
    device = torch.device(f"cuda:{args.gpus[0]}")
    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as handle:
        model = pickle.load(handle)["ema"].to(device).eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    events = formal_event_indices(78)
    print(
        "[Holdout Scan-TTA] scope=full_network | Adam lr=1e-5 | "
        "gamma=1 | CG=5 | refinements/event=5"
    )
    print(
        f"[Holdout Scan-TTA] parameters={parameter_count:,} | "
        f"events={events} | holdout=11/54 acquired columns"
    )
    print(
        "[Budget] baseline_denoiser_calls=780 | tta_denoiser_calls=20 | "
        "tta_backprops=20 | cg_iterations=100"
    )

    evaluator = HoldoutScanTTAEvaluator(
        model=model,
        val_dir=args.val_dir,
        sample_indices=_parse_indices(args.sample_indices),
        device=device,
        seed=123,
    )
    result = evaluator.evaluate(args.save_dir)
    summary = result["metrics"].get("summary", {})
    if summary:
        print(f"PSNR:  {summary['psnr_mean']:.2f} +/- {summary['psnr_std']:.2f}")
        print(f"SSIM:  {summary['ssim_mean']:.4f} +/- {summary['ssim_std']:.4f}")
        print(f"NRMSE: {summary['nrmse_mean']:.4f} +/- {summary['nrmse_std']:.4f}")


if __name__ == "__main__":
    main()
