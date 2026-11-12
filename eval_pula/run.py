"""Run one OPA-pULA trajectory per selected PaDIS-MRI validation sample."""

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "train/padis-mri")]


def sample_list(value):
    indices = []
    for item in value.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item.strip())
        if not match:
            raise argparse.ArgumentTypeError(f"Invalid sample or inclusive range: {item!r}")
        first, last = int(match[1]), int(match[2] or match[1])
        if last < first:
            raise argparse.ArgumentTypeError(f"Reversed sample range: {item!r}")
        indices.extend(range(first, last + 1))
    return list(dict.fromkeys(indices))


def karras_schedule(steps, sigma_min, sigma_max, rho):
    return [
        (sigma_max ** (1 / rho) + i / (steps - 1) *
         (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        for i in range(steps)
    ]


def random_shifts(count, pad, seed):
    rng = random.Random(seed)
    return [(rng.randrange(pad), rng.randrange(pad)) for _ in range(count)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--sample_indices", type=sample_list, default="26")
    parser.add_argument("--grid_mode", choices=("fixed", "cycle_k"), default="fixed")
    parser.add_argument("--grid_seed", type=int, default=123)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mask_select", type=int, default=7)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--psize", type=int, default=64)
    parser.add_argument("--sigma_max", type=float, default=1.0, help="PaDIS per-real-channel standard deviation")
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_schedule", choices=("exponential",), default="exponential")
    parser.add_argument("--num_noise_levels", type=int, default=40)
    parser.add_argument("--langevin_steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--cg_iters", type=int, default=10)
    parser.add_argument("--cg_rtol", type=float, default=1e-6)
    parser.add_argument("--dps_tail_steps", type=int, default=0,
                        help="Use the last N levels of the original 78-level PaDIS VE schedule after pULA")
    parser.add_argument("--dps_base_steps", type=int, default=78)
    parser.add_argument("--dps_inner_loops", type=int, default=10)
    parser.add_argument("--dps_zeta", type=float, default=3.0)
    parser.add_argument("--dps_sigma_min", type=float, default=0.003)
    parser.add_argument("--dps_sigma_max", type=float, default=10.0)
    parser.add_argument("--dps_rho", type=float, default=7.0)
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--noise_dir", help="Explicit directory of legacy PaDIS noise_var_*.pt sidecars")
    noise.add_argument("--noise_complex_var", type=float, help="Calibrated E|n|^2 for already whitened coils")
    noise.add_argument("--noise_covariance", help="Calibrated complex coil covariance in a [C,C] NPY file")
    parser.add_argument("--diagnose_patch_score", action="store_true")
    parser.add_argument("--diagnose_levels", type=sample_list, help="Zero-based noise level indices")
    parser.add_argument("--terminal_denoise", action="store_true", help="Also save a separate denoised image")
    args = parser.parse_args()
    if min(args.num_noise_levels, args.langevin_steps, args.cg_iters, args.psize, args.image_size,
           args.dps_base_steps, args.dps_inner_loops) < 1:
        parser.error("Positive level, step, CG and image sizes are required.")
    if args.pad != args.psize or args.image_size % args.psize or args.langevin_steps > args.pad ** 2:
        parser.error("PaDIS shifted grids need pad=psize, image_size divisible by psize and K <= pad^2.")
    if not (0 < args.sigma_min <= args.sigma_max < math.inf and 0 < args.gamma < math.inf
            and 0 <= args.cg_rtol < 1):
        parser.error("Invalid sigma range, gamma or relative CG tolerance.")
    if not (0 <= args.dps_tail_steps <= args.dps_base_steps and args.dps_base_steps > 1
            and 0 < args.dps_sigma_min < args.dps_sigma_max and args.dps_rho > 0
            and args.dps_zeta > 0):
        parser.error("Invalid DPS tail schedule or zeta.")
    if args.diagnose_patch_score and args.diagnose_levels is None:
        args.diagnose_levels = sorted({0, args.num_noise_levels // 2, args.num_noise_levels - 1})
    if args.diagnose_patch_score and args.pad < 2:
        parser.error("Patch diagnostics require room for at least two shifts.")
    if args.diagnose_levels and (not args.diagnose_patch_score or max(args.diagnose_levels) >= args.num_noise_levels):
        parser.error("Diagnostic levels must be within the schedule and require --diagnose_patch_score.")
    for name in ("model_path", "val_dir", "out_root", "noise_dir", "noise_covariance"):
        if getattr(args, name) is not None:
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    return args


def load_measurement(val_dir, index, mask_select, device):
    import torch
    from inverse_operators import MRI_utils, fftmod

    # Same .pt fields and orientation as DPSHyperEvaluator._load_sample.
    data = torch.load(Path(val_dir) / f"sample_{index}.pt", map_location="cpu", weights_only=False)
    maps = fftmod(data["s_map"])[None].to(device=device, dtype=torch.complex64)
    kspace = fftmod(data["ksp"])[None].to(device=device, dtype=torch.complex64)
    mask = data[f"mask_{mask_select}"][None].to(device)
    if mask.shape != (1, 1, *kspace.shape[-2:]) or not ((mask == 0) | (mask == 1)).all().item():
        raise ValueError("pULA requires the PaDIS binary mask shared by all coils.")
    mask = mask.bool()
    return MRI_utils(mask=mask, maps=maps), mask * kspace


def sidecar_path(val, index, noise_dir):
    sample_name = f"sample_{index}.pt"
    indexed = f"noise_var_{index}.pt"
    mapping = val / "mapping.txt"
    source = None
    if mapping.is_file():
        for line in mapping.read_text().splitlines():
            name, _, original = line.partition(" <- ")
            if name == sample_name:
                source = Path(original).name.replace("sample_", "noise_var_", 1)
                break
    if noise_dir:
        directory = Path(noise_dir)
        candidates = ([directory / source] if source else []) + [directory / indexed]
    else:
        candidates = [val / indexed]
        if source:
            candidates.append(val / source)
            if val.parent.name.endswith("_subsamp"):
                origin = val.parent.with_name(val.parent.name[:-len("_subsamp")]) / val.name
                candidates.append(origin / source)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Sample {index}: no matched PaDIS noise calibration. Checked {candidates}. "
        "Set NOISE_DIR to the original sidecars, NOISE_COMPLEX_VAR to a calibrated whitened value, "
        "or NOISE_COVARIANCE to an independently calibrated matrix."
    )


def load_noise(args, index):
    import numpy as np
    import torch

    if args.noise_covariance:
        covariance = torch.from_numpy(np.load(args.noise_covariance, allow_pickle=False))
        return covariance, {"kind": "external_complex_coil_covariance", "source": args.noise_covariance}
    if args.noise_complex_var is not None:
        variance = args.noise_complex_var
        kind, source = "external_whitened_scalar", "--noise_complex_var"
    elif args.noise_dir:
        path = sidecar_path(Path(args.val_dir), index, args.noise_dir)
        data = torch.load(path, map_location="cpu", weights_only=False)
        variance = float(data["noise_var_noisy"])
        kind, source = "legacy_whitened_corner_estimate", str(path)
    else:
        # brain_val_data.py whitens with the estimated complex coil covariance,
        # then divides k-space by norm_consts_99. For 32dB no extra synthetic
        # noise is added, so E|n|^2 in stored coordinates is 1 / norm^2.
        if Path(args.val_dir).name != "32dB":
            raise ValueError(
                "Automatic norm-constant calibration is defined only for the PaDIS 32dB pipeline. "
                "Provide --noise_complex_var, --noise_covariance or --noise_dir."
            )
        sample_path = Path(args.val_dir) / f"sample_{index}.pt"
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        norm_const = float(sample["norm_consts_99"])
        if not math.isfinite(norm_const) or norm_const <= 0:
            raise ValueError(f"Sample {index}: invalid norm_consts_99={norm_const} in {sample_path}.")
        variance = 1.0 / norm_const ** 2
        kind, source = "bart_whiten_identity_then_acs99", f"{sample_path}:norm_consts_99"
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError(f"Sample {index}: invalid complex measurement variance {variance} from {source}.")
    calibration = {"kind": kind, "source": source, "complex_variance": variance,
                   "variance_per_real_channel": variance / 2}
    if kind == "bart_whiten_identity_then_acs99":
        calibration.update(pre_normalization_complex_variance=1.0, normalization_const=norm_const)
    return variance, calibration


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def valid_image(path, size):
    import numpy as np

    try:
        image = np.load(path, allow_pickle=False)
        return image.shape == (1, size, size) and np.iscomplexobj(image) and np.isfinite(image).all()
    except (OSError, ValueError, EOFError):
        return False


def valid_metrics(path, index):
    data = read_json(path)
    rows = data.get("per_image") if isinstance(data, dict) else None
    row = rows.get(str(index)) if isinstance(rows, dict) else None
    return isinstance(row, dict) and all(
        isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
        for key in ("psnr", "ssim", "nrmse")
    )


def save_image(path, tensor):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, tensor.cpu().numpy())
    tmp.replace(path)


def main():
    args = parse_args()
    import torch

    import dnnlib
    from eval_pula.recon_pula import PULASystem, reconstruct, whiten_measurement
    from eval_pula.score_opa import PatchScore, patch_shifts

    shifts = patch_shifts(args.grid_mode, args.langevin_steps, args.pad, args.grid_seed)
    diagnostic_shifts = patch_shifts("cycle_k", max(2, args.langevin_steps), args.pad, args.grid_seed) if args.diagnose_patch_score else []
    # BART uses sigma_max at initialization and N lower, log-spaced corrector levels.
    sigmas = [args.sigma_max * (args.sigma_min / args.sigma_max) ** (i / args.num_noise_levels)
              for i in range(args.num_noise_levels + 1)]
    sigmas[-1] = args.sigma_min
    if args.dps_tail_steps:
        dps_base = karras_schedule(args.dps_base_steps, args.dps_sigma_min,
                                   args.dps_sigma_max, args.dps_rho)
        dps_sigmas = dps_base[-args.dps_tail_steps:]
        pula_levels = sum(sigma >= dps_sigmas[0] for sigma in sigmas[1:])
        if pula_levels < 1:
            raise ValueError("DPS tail begins above the first pULA update level.")
        dps_shifts = random_shifts(args.dps_tail_steps * args.dps_inner_loops,
                                   args.pad, args.grid_seed)
    else:
        dps_sigmas, dps_shifts = [], []
        pula_levels = args.num_noise_levels
    request = {key: value for key, value in vars(args).items() if key not in ("sample_indices", "out_root")}
    request.update(
        sampler="opa_pula_dps_tail_v1" if dps_sigmas else "opa_pula_v2", trajectories=1,
        main_output="pula_state_after_original_padis_dps_tail" if dps_sigmas else "single_trajectory_final_state",
        gradient="paired_real_euclidean", sigma_units="per_real_channel_std",
        initialization="perturb_and_solve_gaussian_approximation", cg_initial="bart_image_term",
        rng_policy="fixed_per_sample_private_generator", sigma_init=sigmas[0],
        sigma_levels=sigmas[1:pula_levels + 1], pula_levels=pula_levels,
        pula_handoff_sigma=sigmas[pula_levels], dps_tail_sigmas=dps_sigmas,
        shifts=[list(shift) for shift in shifts], diagnostic_shifts=[list(shift) for shift in diagnostic_shifts],
        dps_shifts=[list(shift) for shift in dps_shifts],
        patch_batch=int(os.environ.get("PADIS_PATCH_BATCH", 49)),
        patch_checkpoint=os.environ.get("PADIS_PATCH_CHECKPOINT", "0"),
    )
    if request["patch_batch"] < 1:
        raise ValueError("PADIS_PATCH_BATCH must be positive.")
    oracle = None
    for index in args.sample_indices:
        sample_dir = Path(args.out_root) / args.grid_mode / f"sample_{index}"
        out = sample_dir / "evaluate"
        config_path = out / "config.json"
        expected = {**request, "sample_index": index}
        config = read_json(config_path)
        if config is not None and (not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items())):
            raise ValueError(f"Existing configuration differs for {sample_dir}. Use a different OUT_ROOT.")
        if config is None and out.exists() and any(out.iterdir()):
            raise ValueError(f"Cannot verify ownership of {out} without its configuration. Use a different OUT_ROOT.")

        name = f"recon_patch_{index}.npy"
        recon_path = out / "recons" / name
        terminal_path = out / "terminal_denoised" / name
        metric_path = out / "comp_plots" / "results.json"
        timing_path = out / "timing.json"
        diagnostic_path = out / "patch_score_diagnostics.json"
        reconstructed = (valid_image(recon_path, args.image_size) and isinstance(read_json(timing_path), dict)
                         and (not args.terminal_denoise or valid_image(terminal_path, args.image_size))
                         and (not args.diagnose_patch_score or isinstance(read_json(diagnostic_path), list)))
        if reconstructed and valid_metrics(metric_path, index):
            print(f"Skip sample {index}: reconstruction and metrics verified.", flush=True)
            continue

        if not reconstructed:
            covariance, calibration = load_noise(args, index)
            if config is not None and config.get("noise_calibration") != calibration:
                raise ValueError(f"Noise calibration differs for {sample_dir}. Use a different OUT_ROOT.")
        if not torch.cuda.is_available():
            raise RuntimeError("PaDIS patch inference requires CUDA. Use tools/test_pula_numerics.py for CPU verification.")
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        out.mkdir(parents=True, exist_ok=True)
        with (sample_dir / "run.log").open("a", buffering=1) as log:
            def report(message):
                print(message, flush=True)
                print(message, file=log)

            if not reconstructed:
                if config is None:
                    write_json(config_path, {**expected, "noise_calibration": calibration,
                                             "gpu": torch.cuda.get_device_name(device)})
                metric_path.unlink(missing_ok=True)
                timing_path.unlink(missing_ok=True)
                report(f"Sample {index} | {args.grid_mode} | noise calibration {calibration}")
                if oracle is None:
                    report(f"Loading {args.model_path}")
                    with dnnlib.util.open_url(args.model_path, verbose=False) as file:
                        net = pickle.load(file)["ema"].to(device).eval()
                    oracle = PatchScore(net, args.image_size, args.pad, args.psize, device)
                raw_operator, raw_measurement = load_measurement(args.val_dir, index, args.mask_select, device)
                if raw_measurement.shape[-2:] != (args.image_size, args.image_size):
                    raise ValueError("Measurement FOV differs from --image_size.")
                operator, measurement = whiten_measurement(raw_operator, raw_measurement, covariance)
                generator = torch.Generator(device=device).manual_seed(args.seed)
                system = PULASystem(operator, measurement, generator, args.cg_iters, args.cg_rtol)
                if dps_sigmas:
                    report(f"Hybrid handoff: pULA {pula_levels} levels through sigma={sigmas[pula_levels]:.6g}; "
                           f"DPS last {len(dps_sigmas)} levels from sigma={dps_sigmas[0]:.6g} "
                           f"to {dps_sigmas[-1]:.6g}, inner={args.dps_inner_loops}, zeta={args.dps_zeta}")
                state, terminal, timing, diagnostics = reconstruct(
                    system, oracle, sigmas, shifts, args.gamma, diagnose_levels=args.diagnose_levels or (),
                    diagnostic_shifts=diagnostic_shifts, terminal=args.terminal_denoise, report=report,
                    pula_levels=pula_levels, dps_operator=raw_operator, dps_measurement=raw_measurement,
                    dps_sigmas=dps_sigmas, dps_shifts=dps_shifts,
                    dps_inner_loops=args.dps_inner_loops, dps_zeta=args.dps_zeta,
                )
                save_image(recon_path, state)
                if terminal is not None:
                    save_image(terminal_path, terminal)
                if args.diagnose_patch_score:
                    write_json(diagnostic_path, diagnostics)
                write_json(timing_path, timing)
                report(f"Reconstruction took {timing['reconstruction_seconds']:.3f} s")
            else:
                report(f"Sample {index}: evaluating saved reconstruction.")

            from utils import post_eval_normalize

            metrics = post_eval_normalize(recon_dir=str(recon_path.parent), val_dir=args.val_dir,
                                          plot_dir=str(metric_path.parent), mask_select=args.mask_select)
            if not valid_image(recon_path, args.image_size) or not valid_metrics(metric_path, index):
                raise RuntimeError(f"Sample {index}: saved reconstruction or per-sample metrics are invalid.")
            report(f"Sample {index} complete | {metrics['per_image'][str(index)]}")


if __name__ == "__main__":
    main()
