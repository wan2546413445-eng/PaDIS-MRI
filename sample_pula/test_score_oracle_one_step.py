"""
One-step equivalence test:
verify that PaDISPatchScoreOracle(mode="legacy_dps2") reproduces the score
construction in `eval/recon.py::dps2()` for the same x, sigma, noise, and patch
indices.

This test isolates the prior interface extraction before using the oracle in
pULA.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
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
from utils import fftmod
from denoise_padding import denoisedFromPatches, getIndices
from padis_score_oracle import PaDISPatchScoreOracle, PaDISScoreOracleConfig


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
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(args.model_path, verbose=False) as f:
        net = pickle.load(f)["ema"].to(device).eval()

    data = torch.load(Path(args.val_dir) / f"sample_{args.sample_idx}.pt", weights_only=False)
    gt = data["gt"][None, None, ...].to(device)
    s_maps = fftmod(data["s_map"])[None, ...].to(device)
    fs_ksp = fftmod(data["ksp"])[None, ...].to(device)
    mask = data[f"mask_{args.mask_select}"][None, ...].to(device)
    meas = mask * fs_ksp
    invop = MRI_utils(maps=s_maps, mask=mask)

    x_init = invop.adjoint(meas).detach()
    x_pad = torch.nn.functional.pad(x_init, (args.pad, args.pad, args.pad, args.pad), "constant", 0)
    sigma = torch.tensor(args.sigma, device=device, dtype=x_pad.real.dtype)
    eps = torch.randn_like(x_pad)
    x_noisy = x_pad + sigma * eps

    patches = args.image_size // args.psize + 1
    spaced = np.linspace(0, (patches - 1) * args.psize, patches, dtype=int)
    indices = getIndices(spaced, patches, args.pad, args.psize, freezeindex=True)

    # Manual legacy DPS2 score.
    x_real_noisy = torch.view_as_real(x_noisy.squeeze(1)).permute(0, 3, 1, 2)
    D_real_manual = denoisedFromPatches(
        net, x_real_noisy, sigma, build_pos(args.image_size, args.pad, device),
        None, indices, t_goal=0, wrong=False
    )
    D_cplx_manual = torch.complex(D_real_manual[:, 0], D_real_manual[:, 1]).unsqueeze(1)
    score_manual = (D_cplx_manual - x_noisy) / (sigma ** 2)

    oracle = PaDISPatchScoreOracle(
        net,
        PaDISScoreOracleConfig(
            image_size=args.image_size,
            pad=args.pad,
            psize=args.psize,
            mode="legacy_dps2",
            freeze_patch_indices=True,
        ),
    )
    out = oracle(
        x_pad,
        sigma,
        build_pos(args.image_size, args.pad, device),
        mode="legacy_dps2",
        indices=indices,
        eps=eps,
    )

    diff = (out.score - score_manual).abs()
    result = {
        "max_abs_diff": float(diff.max().detach().cpu()),
        "mean_abs_diff": float(diff.mean().detach().cpu()),
        "allclose_atol_1e-6": bool(torch.allclose(out.score, score_manual, atol=1e-6, rtol=1e-5)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
