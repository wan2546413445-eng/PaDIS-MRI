"""
One-sample runner for the clean PaDIS-pULA refactor.

Recommended first check:
1) run `test_score_oracle_one_step.py` to verify the PaDIS prior extraction.
2) run this file with a conservative deterministic configuration.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
EVAL_DIR = PROJECT_ROOT / "eval"
for _p in (str(EVAL_DIR), str(PROJECT_ROOT), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dnnlib
from inverse_operators import MRI_utils
from utils import fftmod, makeFigures
from pula_sampler import PULASamplerConfig, padis_pula_reconstruct


def build_pos(image_size: int, pad: int, device: torch.device) -> torch.Tensor:
    r = image_size + 2 * pad
    x = torch.linspace(-1, 1, r, device=device)
    y = torch.linspace(-1, 1, r, device=device)
    return torch.stack(
        [x.view(1, -1).repeat(r, 1), y.view(-1, 1).repeat(1, r)],
        dim=0,
    ).unsqueeze(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--sample_idx", type=int, default=3)
    p.add_argument("--mask_select", type=int, default=7)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--psize", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--corrector_steps", type=int, default=1)
    p.add_argument("--cg_iters", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--sigma_max", type=float, default=10.0)
    p.add_argument("--score_mode", choices=["direct", "legacy_dps2"], default="direct")
    p.add_argument("--init_mode", choices=["bart_source", "precond_mean", "adjoint"], default="adjoint")
    p.add_argument("--prior_scale", type=float, default=1.0)
    p.add_argument("--likelihood_scale", type=float, default=1.0)
    p.add_argument("--init_noise_scale", type=float, default=0.0)
    p.add_argument("--step_noise_scale", type=float, default=0.0)
    p.add_argument("--freeze_patch_indices", action="store_true")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as f:
        net = pickle.load(f)["ema"].to(device).eval()

    data = torch.load(Path(args.val_dir) / f"sample_{args.sample_idx}.pt", weights_only=False)
    gt = data["gt"][None, None, ...].to(device)
    s_maps = fftmod(data["s_map"])[None, ...].to(device)
    fs_ksp = fftmod(data["ksp"])[None, ...].to(device)
    mask = data[f"mask_{args.mask_select}"][None, ...].to(device)
    meas = mask * fs_ksp
    invop = MRI_utils(maps=s_maps, mask=mask)

    cfg = PULASamplerConfig(
        num_steps=args.num_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        corrector_steps=args.corrector_steps,
        cg_iters=args.cg_iters,
        gamma=args.gamma,
        score_mode=args.score_mode,
        init_mode=args.init_mode,
        prior_scale=args.prior_scale,
        likelihood_scale=args.likelihood_scale,
        init_noise_scale=args.init_noise_scale,
        step_noise_scale=args.step_noise_scale,
        image_size=args.image_size,
        pad=args.pad,
        psize=args.psize,
        freeze_patch_indices=args.freeze_patch_indices,
    )

    print("[PaDIS-pULA clean refactor]")
    print(json.dumps(cfg.__dict__, indent=2))

    recon, diag = padis_pula_reconstruct(
        net=net,
        inverseop=invop,
        measurement=meas,
        latents_pos=build_pos(args.image_size, args.pad, device),
        config=cfg,
    )

    np.save(save_dir / f"recon_pula_sample_{args.sample_idx}.npy", recon.detach().cpu().numpy())
    with open(save_dir / f"diagnostics_pula_sample_{args.sample_idx}.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    try:
        vals = makeFigures(
            noisy2=invop.adjoint(meas).detach().squeeze(1),
            denoised2=recon.detach(),
            orig2=gt.detach().squeeze(1),
            i=args.sample_idx,
            out_dir=str(save_dir),
            tag=f"pula_clean_sample{args.sample_idx}",
            plot=True,
        )
        metrics = {
            "noisy_psnr": float(vals[0]),
            "recon_psnr": float(vals[1]),
            "noisy_ssim": float(vals[2]),
            "recon_ssim": float(vals[3]),
            "noisy_nrmse": float(vals[4]),
            "recon_nrmse": float(vals[5]),
        }
        with open(save_dir / f"metrics_pula_sample_{args.sample_idx}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))
    except Exception as exc:
        print(f"[PaDIS-pULA clean] Metric/plot helper skipped: {exc}")


if __name__ == "__main__":
    main()
