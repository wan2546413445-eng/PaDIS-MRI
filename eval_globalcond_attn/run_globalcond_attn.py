import os
import sys
import argparse
import pickle
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(REPO_DIR, "eval")
TRAIN_DIR = os.path.join(REPO_DIR, "train", "padis-mri")

sys.path.insert(0, REPO_DIR)
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, TRAIN_DIR)

import dnnlib
from evaluator_globalcond_attn import DPSHyperEvaluator

from utils import post_eval_normalize


def parse_args():
    p = argparse.ArgumentParser(description="GC-Attn PaDIS evaluation runner")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--psize", type=int, default=64)
    p.add_argument("--mask_select", type=int, default=7)
    p.add_argument("--val_count", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sample_indices", type=str, default="")

    p.add_argument("--algo", type=str, default="padis", choices=["padis"])
    p.add_argument("--zeta", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=78)
    p.add_argument("--inner_loops", type=int, default=10)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=None)
    p.add_argument("--report_every", type=int, default=1)

    p.add_argument("--global_context_size", type=int, default=96)
    p.add_argument("--patch_batch_size", type=int, default=8)

    p.add_argument("--run_evaluate", action="store_true")
    p.add_argument("--save_intermediate", action="store_true")
    p.add_argument("--intermediate_every", type=int, default=10)

    return p.parse_args()


def parse_list(csv_str, cast=int):
    if csv_str is None or csv_str == "":
        return []
    return [cast(x.strip()) for x in csv_str.split(",") if x.strip() != ""]


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'Loading GC-Attn network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as f:
        model = pickle.load(f)["ema"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    sample_indices = parse_list(args.sample_indices, cast=int)

    opt = DPSHyperEvaluator(
        model=model,
        mask_select=args.mask_select,
        val_dir=args.val_dir,
        image_size=args.image_size,
        pad=args.pad,
        psize=args.psize,
        val_count=args.val_count,
        seed=args.seed,
        sample_indices=sample_indices if sample_indices else None,
        global_context_size=args.global_context_size,
        patch_batch_size=args.patch_batch_size,
    )

    print(
        f"[GC-PaDIS Budget] steps={args.steps}, "
        f"inner_loops={args.inner_loops}, "
        f"total_updates={args.steps * args.inner_loops}, "
        f"global_context_size={args.global_context_size}, "
        f"patch_batch_size={args.patch_batch_size}"
    )

    if args.run_evaluate:
        metrics = opt.evaluate(
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
            lam=1e-4,
            save_intermediate=args.save_intermediate,
            intermediate_every=args.intermediate_every,
        )

        s = metrics["summary"]
        print(f"PSNR:  {s['psnr_mean']:.2f} ± {s['psnr_std']:.2f}")
        print(f"SSIM:  {s['ssim_mean']:.4f} ± {s['ssim_std']:.4f}")
        print(f"NRMSE: {s['nrmse_mean']:.4f} ± {s['nrmse_std']:.4f}")

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
    except Exception as e:
        print(f"[post_eval_normalize] Skipped due to error: {e}")


if __name__ == "__main__":
    main()
