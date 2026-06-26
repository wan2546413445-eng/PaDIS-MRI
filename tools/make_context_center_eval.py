#!/usr/bin/env python3
"""
Generate non-overwriting context-center eval files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_recon(text: str) -> str:
    text = text.replace(
        "from denoise_padding import denoisedFromPatches, getIndices",
        "from denoise_padding_context_center import denoisedFromPatches, getIndices",
        1,
    )

    if "context_margin" not in text[text.find("def dps2("): text.find("def dps2(") + 1200]:
        text = text.replace(
            "tag: Optional[str] = None,\n) -> Tuple",
            "tag: Optional[str] = None,\n    context_margin: int = 0,\n) -> Tuple",
            1,
        )

    text = text.replace(
        "indices = getIndices(spaced, patches, pad, psize)",
        "indices = getIndices(spaced, patches, pad, psize, context_margin=context_margin)",
        1,
    )

    text = text.replace(
        "D_real = denoisedFromPatches(net, x_real_noisy, t_cur, latents_pos, None, indices, t_goal=0, wrong=False)",
        "D_real = denoisedFromPatches(net, x_real_noisy, t_cur, latents_pos, None, indices, t_goal=0, wrong=False, context_margin=context_margin)",
        1,
    )

    return text


def patch_evaluator(text: str) -> str:
    text = text.replace(
        "from recon import dps2, dps_uncond, dps_edm, dps_uncond_edm",
        "from recon_context_center import dps2, dps_uncond, dps_edm, dps_uncond_edm",
        1,
    )

    text = text.replace(
        "tag: str=None)-> Tuple[torch.Tensor, float, float, float, float, float, float]:",
        "tag: str=None,\n                     context_margin: int=0)-> Tuple[torch.Tensor, float, float, float, float, float, float]:",
        1,
    )

    text = text.replace(
        "tag=tag\n                                    )",
        "tag=tag,\n                                        context_margin=context_margin\n                                    )",
        1,
    )

    text = text.replace(
        "lam: float = 1e-4, \n    ):",
        "lam: float = 1e-4,\n        context_margin: int = 0,\n    ):",
        1,
    )
    text = text.replace(
        "lam: float = 1e-4,\n    ):",
        "lam: float = 1e-4,\n        context_margin: int = 0,\n    ):",
        1,
    )

    text = text.replace(
        "invop, meas, gt, zeta, pad, psize, num_steps, save_dir, tag_label\n                    )",
        "invop, meas, gt, zeta, pad, psize, num_steps, save_dir, tag_label,\n                        context_margin=context_margin\n                    )",
        1,
    )

    return text


def patch_run(text: str) -> str:
    text = text.replace(
        "from evaluator import DPSHyperEvaluator",
        "from evaluator_context_center import DPSHyperEvaluator",
        1,
    )

    if "--context_margin" not in text:
        marker = 'p.add_argument("--psize", type=int, default=64)'
        if marker not in text:
            raise RuntimeError("Could not find --psize argument in eval/run.py.")
        text = text.replace(
            marker,
            marker + '\n    p.add_argument("--context_margin", type=int, default=0, help="Context margin used by context-center PaDIS models")',
            1,
        )

    text = text.replace(
        "lam=args.lam,\n        )",
        "lam=args.lam,\n            context_margin=args.context_margin,\n        )",
        1,
    )

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="eval")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    required = [
        eval_dir / "recon.py",
        eval_dir / "evaluator.py",
        eval_dir / "run.py",
        eval_dir / "denoise_padding_context_center.py",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    (eval_dir / "recon_context_center.py").write_text(
        "# Auto-generated context-center eval file. Source: recon.py\n\n" +
        patch_recon((eval_dir / "recon.py").read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    (eval_dir / "evaluator_context_center.py").write_text(
        "# Auto-generated context-center eval file. Source: evaluator.py\n\n" +
        patch_evaluator((eval_dir / "evaluator.py").read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    (eval_dir / "run_context_center.py").write_text(
        "# Auto-generated context-center eval file. Source: run.py\n\n" +
        patch_run((eval_dir / "run.py").read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    print("[OK] Wrote eval/recon_context_center.py")
    print("[OK] Wrote eval/evaluator_context_center.py")
    print("[OK] Wrote eval/run_context_center.py")
    print("[Check] grep -R \"context_margin\" -n eval/*context_center.py")


if __name__ == "__main__":
    main()
