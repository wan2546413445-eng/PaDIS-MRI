"""Run the fixed47 pULA + clean-DC + stochastic posterior bridge sampler."""

import argparse
import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "train/padis-mri")]

from eval_pula.run import (  # noqa: E402
    claim_config,
    karras_schedule,
    load_measurement,
    load_noise,
    read_json,
    sample_list,
    save_image,
    valid_image,
    valid_metrics,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", choices=("bridge",), default="bridge")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--sample_indices", type=sample_list, default="0,8,22,26")
    parser.add_argument("--mask_select", type=int, default=7)
    parser.add_argument("--acs", type=int, default=24)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--psize", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid_seed", type=int, default=123)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--cg_iters", type=int, default=10)
    parser.add_argument("--cg_rtol", type=float, default=1e-6)
    parser.add_argument("--bridge_start_level", type=int, default=23,
                        help="Number of full five-step pULA levels; 23 starts bridging at level 24.")
    parser.add_argument("--bridge_eta", type=float, default=0.5)
    parser.add_argument("--bridge_tau_scale", type=float, default=1.0)
    parser.add_argument("--bridge_mode", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument("--plan_only", action="store_true")
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--noise_dir")
    noise.add_argument("--noise_complex_var", type=float)
    noise.add_argument("--noise_covariance")
    args = parser.parse_args()

    if args.mask_select != 7 or args.acs != 24:
        parser.error("The first Bridge experiment is fixed to R7 mask_select=7 and ACS=24.")
    if args.image_size != 384 or args.pad != 64 or args.psize != 64:
        parser.error("Bridge fixed47 requires image_size=384 and pad=psize=64.")
    if args.seed != 123 or args.grid_seed != 123:
        parser.error("The controlled four-sample experiment requires seed=grid_seed=123.")
    if not (0 <= args.bridge_eta <= 1 and args.bridge_tau_scale > 0):
        parser.error("bridge_eta must be in [0,1] and bridge_tau_scale must be positive.")
    if not (0 <= args.bridge_start_level < 47):
        parser.error("bridge_start_level must be in [0,46].")
    if not (args.cg_iters > 0 and 0 <= args.cg_rtol < 1 and args.gamma > 0):
        parser.error("Invalid CG or pULA parameters.")
    for name in ("model_path", "val_dir", "out_root", "noise_dir", "noise_covariance"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    return args


def fixed47_grid():
    grid = karras_schedule(47, 0.003, 1.0, 7.0)
    grid[0], grid[-1] = 1.0, 0.003
    return grid


def compute_plan(args):
    bridge_updates = 47 - args.bridge_start_level
    pula_updates = args.bridge_start_level * 5 + bridge_updates * 4
    grid = fixed47_grid()
    return {
        **vars(args),
        "pula_convention": "padis_real",
        "partition_policy": "shared_cycle_k",
        "sigma_schedule": "fixed47_karras_rho7",
        "sigma_levels": grid,
        "sigma_first": grid[0],
        "sigma_last": grid[-1],
        "network_forwards": 47 * 5,
        "network_backprops": 0,
        "pula_updates": pula_updates,
        "bridge_updates": bridge_updates,
        "update_stage_cg_solves": pula_updates + bridge_updates,
        "initialization_cg_solves": 1,
        "total_cg_solves": 1 + pula_updates + bridge_updates,
        "gpu_executed": False,
    }


def summarize_run(out_root, indices):
    rows = []
    for index in indices:
        base = Path(out_root) / f"sample_{index}" / "evaluate"
        metrics = read_json(base / "comp_plots" / "results.json")
        timing = read_json(base / "timing.json")
        row = metrics["per_image"][str(index)]
        rows.append({
            "sample_index": index,
            "psnr": row["psnr"],
            "ssim": row["ssim"],
            "nrmse": row["nrmse"],
            "reconstruction_seconds": timing["reconstruction_seconds"],
        })
    means = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("psnr", "ssim", "nrmse", "reconstruction_seconds")
    }
    summary = {"per_image": rows, "mean": means, "sample_count": len(rows)}
    write_json(Path(out_root) / "summary.json", summary)
    return summary


def main():
    args = parse_args()
    plan = compute_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    import torch
    import dnnlib
    from eval_pula.recon_bridge import reconstruct_bridge
    from eval_pula.recon_pula import PULASystem, whiten_measurement
    from eval_pula.score_opa import PatchScore, patch_shifts

    if not torch.cuda.is_available():
        raise RuntimeError("Real PaDIS Bridge inference requires CUDA; run tools/test_bridge_numerics.py on CPU.")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    grid = fixed47_grid()
    shifts = patch_shifts("cycle_k", 5, args.pad, args.grid_seed)
    request = {
        **plan,
        "sampler": "pula_clean_dc_stochastic_posterior_bridge_v1",
        "bridge_start_index_zero_based": args.bridge_start_level,
        "bridge_first_level_one_based": args.bridge_start_level + 1,
        "clean_dc_objective": "0.5*||Aw*z-yw||^2 + 0.5/tau^2*||z-xhat0||^2",
        "tau_policy": "tau=bridge_tau_scale*sigma_current (heuristic initial policy)",
        "final_sigma_next": 0.0,
        "shifts": [list(value) for value in shifts],
        "rng_policy": "one private torch.Generator per sample, each seeded with 123",
        "gpu": torch.cuda.get_device_name(device),
    }
    for key in ("out_root", "sample_indices", "plan_only", "gpu_executed"):
        request.pop(key, None)

    oracle = None
    for index in args.sample_indices:
        sample_dir = Path(args.out_root) / f"sample_{index}"
        out = sample_dir / "evaluate"
        config_path = out / "config.json"
        timing_path = out / "timing.json"
        recon_path = out / "recons" / f"recon_patch_{index}.npy"
        metric_path = out / "comp_plots" / "results.json"
        expected = {**request, "sample_index": index}
        config = read_json(config_path)
        if config is not None and (
            not isinstance(config, dict)
            or any(config.get(key) != value for key, value in expected.items())
        ):
            raise ValueError(f"Existing Bridge configuration differs for {sample_dir}; use a new OUT_ROOT.")
        if config is None and out.exists() and any(out.iterdir()):
            raise ValueError(f"Cannot verify nonempty {out} without config.json; use a new OUT_ROOT.")
        complete = (
            valid_image(recon_path, args.image_size)
            and isinstance(read_json(timing_path), dict)
            and valid_metrics(metric_path, index)
        )
        if complete:
            print(f"Skip sample {index}: Bridge reconstruction and metrics are complete.", flush=True)
            continue

        covariance, calibration = load_noise(args, index)
        if config is not None and config.get("noise_calibration") != calibration:
            raise ValueError(f"Noise calibration differs for {sample_dir}; use a new OUT_ROOT.")
        out.mkdir(parents=True, exist_ok=True)
        if config is None:
            claim_config(config_path, {**expected, "noise_calibration": calibration})

        with (sample_dir / "run.log").open("a", buffering=1) as log:
            def report(message):
                print(message, flush=True)
                print(message, file=log)

            metric_path.unlink(missing_ok=True)
            timing_path.unlink(missing_ok=True)
            report(
                f"Sample {index} | fixed47 shared_cycle_k | padis_real | "
                f"pULA levels 1-{args.bridge_start_level}, Bridge levels "
                f"{args.bridge_start_level + 1}-47 | mode={args.bridge_mode}"
            )
            if oracle is None:
                report(f"Loading {args.model_path}")
                with dnnlib.util.open_url(args.model_path, verbose=False) as file:
                    net = pickle.load(file)["ema"].to(device).eval()
                oracle = PatchScore(net, args.image_size, args.pad, args.psize, device)

            raw_operator, raw_measurement = load_measurement(
                args.val_dir, index, args.mask_select, device,
            )
            if raw_measurement.shape[-2:] != (args.image_size, args.image_size):
                raise ValueError("Measurement FOV differs from --image_size.")
            operator, measurement = whiten_measurement(
                raw_operator, raw_measurement, covariance, convention="padis_real",
            )
            generator = torch.Generator(device=device).manual_seed(args.seed)
            system = PULASystem(
                operator, measurement, generator, args.cg_iters, args.cg_rtol,
                convention="padis_real",
            )
            state, timing = reconstruct_bridge(
                system, oracle, grid, shifts, args.gamma,
                bridge_start_level=args.bridge_start_level,
                bridge_eta=args.bridge_eta,
                bridge_tau_scale=args.bridge_tau_scale,
                bridge_mode=args.bridge_mode,
                report=report,
            )
            timing.update(
                gpu=torch.cuda.get_device_name(device),
                partition_policy="shared_cycle_k",
                pula_convention="padis_real",
                sample_index=index,
            )
            if timing["network_forwards"] != 235 or timing["cg_solves"] != 236:
                raise RuntimeError("Bridge compute budget mismatch; refusing to save reconstruction.")
            save_image(recon_path, state)
            write_json(timing_path, timing)
            report(
                f"Budget verified: forwards={timing['network_forwards']} "
                f"backprops={timing['network_backprops']} pULA={timing['pula_updates']} "
                f"bridge={timing['bridge_updates']} CG={timing['cg_solves']}"
            )

            from utils import post_eval_normalize
            metrics = post_eval_normalize(
                recon_dir=str(recon_path.parent), val_dir=args.val_dir,
                plot_dir=str(metric_path.parent), mask_select=args.mask_select,
            )
            if not valid_image(recon_path, args.image_size) or not valid_metrics(metric_path, index):
                raise RuntimeError(f"Sample {index}: reconstruction or metrics are invalid.")
            report(f"Sample {index} complete | {metrics['per_image'][str(index)]}")

    summary = summarize_run(args.out_root, args.sample_indices)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

