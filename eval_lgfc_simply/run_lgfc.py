"""Command-line entrypoint for one simplified LGFC reconstruction."""

import argparse
from dataclasses import asdict
import json
import os
import pickle
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(REPO_ROOT, "eval")
TRAIN_DIR = os.path.join(REPO_ROOT, "train", "padis-mri")
for path in (TRAIN_DIR, REPO_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_lgfc_simply.config import (
    IMAGE_SIZE,
    INNER_LOOPS,
    PAD,
    PATCH_SIZE,
    SEED,
    STEPS,
    ZETA,
    LGFCConfig,
)


def parse_args():
    defaults = LGFCConfig()
    parser = argparse.ArgumentParser(
        description="PaDIS-MRI reconstruction with simplified LGFC"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--mask_select", type=int, required=True)
    parser.add_argument(
        "--reference_mode",
        choices=["last", "ema"],
        default=defaults.reference_mode,
    )
    parser.add_argument("--ema_beta", type=float, default=defaults.ema_beta)
    parser.add_argument("--start_outer", type=int, default=defaults.start_outer)
    parser.add_argument(
        "--correction_every", type=int, default=defaults.correction_every
    )
    parser.add_argument(
        "--unsampled_weight", type=float, default=defaults.unsampled_weight
    )
    parser.add_argument(
        "--max_relative_update", type=float, default=defaults.max_relative_update
    )
    parser.add_argument("--save_diagnostics", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    import dnnlib
    import torch
    from utils import post_eval_normalize
    from eval_lgfc_simply.evaluator_lgfc import LgfcDPSHyperEvaluator

    os.makedirs(args.save_dir, exist_ok=True)
    lgfc_config = LGFCConfig(
        reference_mode=args.reference_mode,
        ema_beta=args.ema_beta,
        start_outer=args.start_outer,
        correction_every=args.correction_every,
        unsampled_weight=args.unsampled_weight,
        max_relative_update=args.max_relative_update,
    )
    run_config = {
        "model_path": args.model_path,
        "val_dir": args.val_dir,
        "save_dir": args.save_dir,
        "sample_index": args.sample_index,
        "mask_select": args.mask_select,
        "save_diagnostics": args.save_diagnostics,
        "padis": {
            "image_size": IMAGE_SIZE,
            "pad": PAD,
            "patch_size": PATCH_SIZE,
            "steps": STEPS,
            "inner_loops": INNER_LOOPS,
            "zeta": ZETA,
            "seed": SEED,
        },
        "lgfc": asdict(lgfc_config),
    }
    with open(
        os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(run_config, handle, indent=2)
    print(json.dumps(run_config, indent=2))

    print(f'Loading network from "{args.model_path}"...')
    with dnnlib.util.open_url(args.model_path, verbose=False) as handle:
        model = pickle.load(handle)["ema"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    evaluator = LgfcDPSHyperEvaluator(
        model=model,
        mask_select=args.mask_select,
        val_dir=args.val_dir,
        image_size=IMAGE_SIZE,
        pad=PAD,
        psize=PATCH_SIZE,
        val_count=1,
        seed=SEED,
        sample_indices=[args.sample_index],
        lgfc_config=lgfc_config,
        save_diagnostics=args.save_diagnostics,
    )
    evaluate_dir = os.path.join(args.save_dir, "evaluate")
    evaluator.evaluate(
        zeta=ZETA,
        num_steps=STEPS,
        inner_loops=INNER_LOOPS,
        pad=PAD,
        psize=PATCH_SIZE,
        algo="padis",
        save_dir=evaluate_dir,
        tag="patch",
        report_every=1,
    )

    # DPSHyperEvaluator writes placeholder zero metrics because reconstruction
    # metrics are computed by post_eval_normalize below.
    os.remove(os.path.join(evaluate_dir, "metrics.json"))

    post_eval_normalize(
        recon_dir=os.path.join(evaluate_dir, "recons"),
        val_dir=args.val_dir,
        plot_dir=os.path.join(evaluate_dir, "comp_plots"),
        json_basename="results",
        mask_select=args.mask_select,
    )


if __name__ == "__main__":
    main()
