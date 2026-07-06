import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dnnlib
from evaluator import DPSHyperEvaluator
from utils import post_eval_normalize
from recon_context_center import dps2_context_center


def parse_args():
    p = argparse.ArgumentParser(description="Context-center PaDIS-MRI evaluation runner")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--val_dir", type=str, required=True)

    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--psize", type=int, default=64)
    p.add_argument("--context_margin", type=int, default=16)
    p.add_argument("--chunk", type=int, default=8)

    p.add_argument("--mask_select", type=int, default=7)
    p.add_argument("--val_count", type=int, default=1)
    p.add_argument("--sample_indices", type=str, default="")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--algo", type=str, default="padis", choices=["padis"])
    p.add_argument("--zeta", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=78)
    p.add_argument("--inner_loops", type=int, default=10)

    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=None)
    p.add_argument("--report_every", type=int, default=1)
    p.add_argument("--run_evaluate", action="store_true")

    return p.parse_args()


def parse_list(csv_str, cast=int):
    if csv_str is None or csv_str == "":
        return []
    return [cast(x.strip()) for x in csv_str.split(",") if x.strip() != ""]


class DPSContextCenterEvaluator(DPSHyperEvaluator):
    def __init__(self, *args, context_margin=16, chunk=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_margin = int(context_margin)
        self.chunk = int(chunk)

    def dps2_wrapper(
        self,
        inverse_op,
        measurement,
        clean,
        zeta,
        pad,
        psize,
        num_steps,
        save_dir=None,
        tag=None,
        save_intermediate=False,
        intermediate_every=10,
        inner_loops=10,
    ):
        if measurement is None:
            raise NotImplementedError("Context-center unconditional sampling is not implemented.")

        recon, a, b, c, d, e, f = dps2_context_center(
            net=self.model,
            latents=self.latents,
            latents_pos=self.latents_pos,
            inverseop=inverse_op,
            measurement=measurement,
            clean=clean,
            pad=pad,
            psize=psize,
            context_margin=self.context_margin,
            chunk=self.chunk,
            zeta=zeta,
            num_steps=num_steps,
            inner_loops=inner_loops,
            device=self.device,
            save_dir=save_dir,
            tag=tag,
        )
        return recon, a, b, c, d, e, f


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.algo != "padis":
        raise ValueError("run_context_center.py currently supports only --algo padis")

    print("=" * 60)
    print("Context-center PaDIS-MRI Eval")
    print(f"model_path      = {args.model_path}")
    print(f"val_dir         = {args.val_dir}")
    print(f"save_dir        = {args.save_dir}")
    print(f"psize           = {args.psize}")
    print(f"context_margin  = {args.context_margin}")
    print(f"context input   = {args.psize + 2 * args.context_margin}")
    print(f"chunk           = {args.chunk}")
    print(f"steps           = {args.steps}")
    print(f"inner_loops     = {args.inner_loops}")
    print(f"total_updates   = {args.steps * args.inner_loops}")
    print("=" * 60)

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as f:
        model = pickle.load(f)["ema"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("[context-center] Frozen network parameters for eval-time input-gradient DPS.")

    sample_indices = parse_list(args.sample_indices, cast=int)

    opt = DPSContextCenterEvaluator(
        model=model,
        mask_select=args.mask_select,
        val_dir=args.val_dir,
        image_size=args.image_size,
        pad=args.pad,
        psize=args.psize,
        val_count=args.val_count,
        seed=args.seed,
        sample_indices=sample_indices if sample_indices else None,
        context_margin=args.context_margin,
        chunk=args.chunk,
    )

    tag = f"patch_context_center_m{args.context_margin}"

    if args.run_evaluate:
        metrics = opt.evaluate(
            zeta=args.zeta,
            num_steps=args.steps,
            inner_loops=args.inner_loops,
            pad=args.pad,
            psize=args.psize,
            algo=args.algo,
            save_dir=os.path.join(args.save_dir, "evaluate"),
            tag=tag,
            gpus=args.gpus,
            report_every=args.report_every,
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
