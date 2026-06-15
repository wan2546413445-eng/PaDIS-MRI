import argparse
import os
import pickle
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dnnlib

from evaluator import DPSHyperEvaluator
from recon import dps_uncond
from recon_shifted_fusion import dps2_shifted_fusion
from utils import post_eval_normalize


SHIFTED_FUSION_TAG = "patch"


def parse_args():
    p = argparse.ArgumentParser(
        description="Shifted uniform fusion PaDIS evaluator"
    )
    p.add_argument("--model_path", type=str, required=True, help="Path to network-snapshot .pkl")
    p.add_argument("--val_dir", type=str, required=True, help="Validation directory with .pt samples")
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--psize", type=int, default=64)
    p.add_argument("--mask_select", type=int, default=7)
    p.add_argument("--val_count", type=int, default=32)
    p.add_argument("--sample_indices", type=str, default="")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--zeta", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=78)
    p.add_argument("--inner_loops", type=int, default=10)
    p.add_argument("--fusion_start_update", type=int, default=520)
    p.add_argument("--shift", type=int, default=32)
    p.add_argument("--save_dir", type=str, required=True, help="Where to write outputs")
    p.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU ids (e.g. --gpus 0 1)")
    p.add_argument("--report_every", type=int, default=1)
    p.add_argument("--save_intermediate", action="store_true")
    p.add_argument("--intermediate_every", type=int, default=10)
    return p.parse_args()


def parse_list(csv_str, cast=int):
    if csv_str is None or csv_str == "":
        return []
    return [cast(x.strip()) for x in csv_str.split(",") if x.strip() != ""]


class ShiftedFusionEvaluator(DPSHyperEvaluator):
    def __init__(self, *args, fusion_start_update: int = 520, shift: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_start_update = fusion_start_update
        self.shift = shift

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
        save_intermediate: bool = False,
        intermediate_every: int = 10,
        inner_loops: int = 10,
    ):
        if measurement is None:
            recon = dps_uncond(
                net=self.model,
                batch_size=1,
                resolution=self.image_size,
                psize=psize,
                pad=pad,
                num_steps=num_steps,
                sigma_min=0.003,
                sigma_max=10.0,
                rho=7,
                device=self.device,
                randn_like=torch.randn_like,
            )
            recon_cpu = recon.cpu()
            mag = torch.abs(recon_cpu.squeeze(0).squeeze(0)).numpy()
            return mag.clip(0, 1), 0, 0, 0, 0, 0, 0

        return dps2_shifted_fusion(
            net=self.model,
            latents=self.latents,
            latents_pos=self.latents_pos,
            inverseop=inverse_op,
            measurement=measurement,
            clean=clean,
            pad=pad,
            psize=psize,
            zeta=zeta,
            num_steps=num_steps,
            inner_loops=inner_loops,
            save_dir=save_dir,
            tag=tag,
            save_intermediate=save_intermediate,
            intermediate_every=intermediate_every,
            fusion_start_update=self.fusion_start_update,
            shift=self.shift,
        )


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as f:
        model = pickle.load(f)["ema"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    sample_indices = parse_list(args.sample_indices, cast=int)
    opt = ShiftedFusionEvaluator(
        model=model,
        mask_select=args.mask_select,
        val_dir=args.val_dir,
        image_size=args.image_size,
        pad=args.pad,
        psize=args.psize,
        val_count=args.val_count,
        seed=args.seed,
        sample_indices=sample_indices if sample_indices else None,
        fusion_start_update=args.fusion_start_update,
        shift=args.shift,
    )

    print(
        f"[Shifted Fusion Budget] steps={args.steps}, inner_loops={args.inner_loops}, "
        f"total_updates={args.steps * args.inner_loops}, fusion_start_update={args.fusion_start_update}, "
        f"shift={args.shift}"
    )
    opt.evaluate(
        zeta=args.zeta,
        num_steps=args.steps,
        inner_loops=args.inner_loops,
        pad=args.pad,
        psize=args.psize,
        algo="padis",
        save_dir=os.path.join(args.save_dir, "evaluate"),
        tag=SHIFTED_FUSION_TAG,
        gpus=args.gpus,
        report_every=args.report_every,
        save_intermediate=args.save_intermediate,
        intermediate_every=args.intermediate_every,
    )
    try:
        post_eval_normalize(
            recon_dir=os.path.join(args.save_dir, "evaluate", "recons"),
            val_dir=args.val_dir,
            plot_dir=os.path.join(args.save_dir, "evaluate", "comp_plots"),
            json_basename="results",
            mask_select=args.mask_select,
        )
    except Exception as e:
        print(f"[post_eval_normalize] Skipped due to error: {e}")


if __name__ == "__main__":
    main()
