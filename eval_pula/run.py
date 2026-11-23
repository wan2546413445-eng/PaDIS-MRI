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
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
_EDM_ENVIRONMENT = None


def prepare_fullimage_environment():
    """Import the actual EDM entry point without adding either training tree."""
    global _EDM_ENVIRONMENT
    if _EDM_ENVIRONMENT is None:
        import runpy

        # Match `python eval/run.py`, then execute its real import chain.
        sys.path[0] = str(ROOT / "eval")
        _EDM_ENVIRONMENT = runpy.run_path(str(ROOT / "eval/run.py"), run_name="_pula_edm_environment")
    return _EDM_ENVIRONMENT


def load_fullimage_checkpoint(model_path, device):
    """Use eval/run.py's imported dnnlib/pickle and its EMA loading convention."""
    edm = prepare_fullimage_environment()
    with edm["dnnlib"].util.open_url(model_path, verbose=False) as file:
        model = edm["pickle"].load(file)["ema"]
    return model.to(device).eval()


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
    parser.add_argument("--score_mode", choices=("patch", "full"), default="patch")
    parser.add_argument("--grid_mode", choices=("fixed", "cycle_k", "random_step"), default="fixed")
    parser.add_argument("--pula_convention", choices=("padis_real", "paper_complex"), default="padis_real")
    parser.add_argument("--pula_variant", choices=("baseline", "noise08", "balanced_k0", "balanced_full"), default="baseline")
    parser.add_argument("--pa_block_size", type=int, default=8)
    parser.add_argument("--pa_epsilon", type=float, default=1e-12)
    parser.add_argument("--pula_dc_guide", choices=("none", "vjp", "fd", "edm_skip"), default="none")
    parser.add_argument("--pula_dc_last_levels", type=int, default=8)
    parser.add_argument("--pula_dc_guide_weight", type=float, default=1.0)
    parser.add_argument("--pula_dc_mix", choices=("replace", "add"), default="replace")
    parser.add_argument("--pula_dc_fd_rel_eps", type=float, default=0.01)
    parser.add_argument("--dry_run", action="store_true", help="Validate and print the plan without torch/model/data/GPU execution")
    parser.add_argument("--save_pula_handoff", action="store_true",
                        help="Save the FOV state immediately after pULA and before DPS")
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
    parser.add_argument("--fixed47_switch", type=int, default=None,
                        help="Controlled ablation: use one shared 47-level Karras sigma grid; "
                             "switch first N levels to pULA, remaining levels to DPS. "
                             "Requires N=47, K=5 for both algorithms.")
    parser.add_argument("--fixed47_patch_policy",
                        choices=("shared_cycle_k", "native_global_random"),
                        default="shared_cycle_k",
                        help="Grid control for fixed47: same cycle_k for both methods, or "
                             "native DPS random shifts indexed globally.")
    parser.add_argument("--consensus_mode", choices=("off", "mean", "dual"), default="off",
                        help="Opt-in alternative: 47-level multi-partition consensus; no pULA update or DPS")
    parser.add_argument("--fixed47_skip_pula", action="store_true",
                        help="Diagnostic control: keep common CG initialization, "
                             "skip all pULA updates, and execute only the chosen DPS tail.")
    parser.add_argument("--dps_tail_steps", type=int, default=0,
                        help="Use the last N levels of the original 78-level PaDIS VE schedule after pULA")
    parser.add_argument("--dps_base_steps", type=int, default=78)
    parser.add_argument("--dps_inner_loops", type=int, default=10)
    parser.add_argument("--dps_zeta", type=float, default=3.0)
    parser.add_argument("--dps_sigma_min", type=float, default=0.003)
    parser.add_argument("--dps_sigma_max", type=float, default=10.0)
    parser.add_argument("--dps_rho", type=float, default=7.0)
    parser.add_argument(
        "--dps_tail_sigma_max",
        type=float,
        help=(
            "Generate an independent DPS tail from this sigma to "
            "--dps_sigma_min. If omitted, keep the original behavior."
        ),
    )
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--noise_dir", help="Explicit directory of legacy PaDIS noise_var_*.pt sidecars")
    noise.add_argument("--noise_complex_var", type=float, help="Calibrated E|n|^2 for already whitened coils")
    noise.add_argument("--noise_covariance", help="Calibrated complex coil covariance in a [C,C] NPY file")
    parser.add_argument("--diagnose_patch_score", action="store_true")
    parser.add_argument("--diagnose_dps_jacobian", action="store_true", help="Read-only DPS Jacobian mechanism diagnostics")
    parser.add_argument("--jacobian_levels", type=sample_list, help="Zero-based DPS tail outer levels")
    parser.add_argument("--jacobian_inner_steps", type=sample_list, help="Zero-based DPS tail inner steps")
    parser.add_argument("--diagnose_levels", type=sample_list, help="Zero-based noise level indices")
    parser.add_argument("--terminal_denoise", action="store_true", help="Also save a separate denoised image")
    parser.add_argument("--num_trajectories", type=int, default=None,
                        help="Opt into single-process trajectory batching; omitted keeps the legacy runner")
    parser.add_argument("--trajectory_batch_size", type=int, default=1)
    parser.add_argument("--trajectory_seed_start", type=int, default=None)
    parser.add_argument("--aggregate_counts", type=sample_list, default="1,2,3,5,8,10")
    parser.add_argument("--save_each_trajectory", type=lambda x: bool_value(x), default=True)
    parser.add_argument("--measure_cuda_memory", type=lambda x: bool_value(x), default=True)
    args = parser.parse_args()
    if args.consensus_mode != "off":
        if (args.fixed47_switch != 47 or args.score_mode != "patch"
                or args.grid_mode != "cycle_k" or args.fixed47_patch_policy != "shared_cycle_k"
                or args.langevin_steps != 5 or args.num_noise_levels != 47
                or args.pula_variant != "baseline" or args.num_trajectories is not None
                or args.fixed47_skip_pula or args.diagnose_patch_score
                or args.diagnose_dps_jacobian):
            parser.error("consensus pilot requires --fixed47_switch 47, patch/cycle_k, "
                         "shared_cycle_k, K=5, N=47, baseline variant, no diagnostics")
    if args.fixed47_skip_pula and args.fixed47_switch is None:
        parser.error("--fixed47_skip_pula requires --fixed47_switch.")
    if args.fixed47_switch is not None:
        n = args.fixed47_switch
        if args.fixed47_skip_pula and not 0 < n < 47:
            parser.error("skip-pULA control requires a nonempty DPS tail (1<=switch<=46).")
        if args.dps_inner_loops != 5 and not (n == 0 and args.dps_inner_loops == 3
                                             and args.fixed47_patch_policy == "shared_cycle_k"):
            parser.error("fixed47 accepts DPS inner=5, or switch=0 + shared_cycle_k + inner=3.")
        if not 0 <= n <= 47:
            parser.error("--fixed47_switch must be in [0,47].")
        if (args.score_mode != "patch" or args.grid_mode != "cycle_k"
                or args.num_noise_levels != 47 or args.langevin_steps != 5
                or args.dps_base_steps != 47
                or args.dps_tail_sigma_max is not None):
            parser.error("fixed47 requires patch/cycle_k, N=47, pULA K=5, "
                         "DPS base=47 and no custom DPS tail sigma max.")
        if (not math.isclose(args.sigma_max, args.dps_sigma_max, rel_tol=0, abs_tol=1e-12)
                or not math.isclose(args.sigma_min, args.dps_sigma_min, rel_tol=0, abs_tol=1e-12)):
            parser.error("fixed47 requires identical pULA and DPS sigma_max/min.")
        args.dps_tail_steps = 47 - n
    if args.pa_block_size < 1 or not (math.isfinite(args.pa_epsilon) and args.pa_epsilon > 0):
        parser.error("PA block size and epsilon must be positive and finite.")
    if args.pula_dc_guide != "none":
        if (args.score_mode != "patch" or args.num_trajectories is not None
                or args.pula_variant != "baseline"):
            parser.error("pULA DC guidance supports single-trajectory patch/baseline pULA only.")
        if args.fixed47_switch == 0:
            parser.error("pULA DC guidance needs at least one pULA level.")
        if (args.pula_dc_last_levels < 1
                or not 0 <= args.pula_dc_guide_weight <= 1
                or not math.isfinite(args.pula_dc_guide_weight)
                or not math.isfinite(args.pula_dc_fd_rel_eps) or args.pula_dc_fd_rel_eps <= 0):
            parser.error("Invalid pULA DC guidance last-levels, weight, or FD relative epsilon.")
    if args.pula_variant != "baseline" and (args.grid_mode != "random_step" or args.score_mode != "patch"):
        parser.error("New pULA variants require patch score and grid_mode=random_step for paired comparisons.")
    if args.num_trajectories is not None:
        if min(args.num_trajectories, args.trajectory_batch_size) < 1:
            parser.error("Trajectory count and batch size must be positive.")
        if args.score_mode != "patch" or args.dps_tail_steps or args.terminal_denoise or args.diagnose_patch_score or args.save_pula_handoff or args.diagnose_dps_jacobian:
            parser.error("Trajectory batching supports pure block pULA raw states only.")
        if any(count < 1 for count in args.aggregate_counts):
            parser.error("aggregate_counts must be positive.")
    if min(args.num_noise_levels, args.langevin_steps, args.cg_iters, args.image_size,
           args.dps_base_steps, args.dps_inner_loops) < 1:
        parser.error("Positive level, step, CG and image sizes are required.")
    if args.score_mode == "patch" and args.psize < 1:
        parser.error("Positive level, step, CG and image sizes are required.")
    if args.score_mode == "patch" and (args.pad != args.psize or args.image_size % args.psize
                                      or args.langevin_steps > args.pad ** 2):
        parser.error("PaDIS shifted grids need pad=psize, image_size divisible by psize and K <= pad^2.")
    if args.score_mode == "full" and (args.diagnose_patch_score or args.diagnose_levels or args.dps_tail_steps or args.diagnose_dps_jacobian):
        parser.error("Full-image pULA does not use patch grid diagnostics or the patch DPS tail.")
    if not (0 < args.sigma_min <= args.sigma_max < math.inf and 0 < args.gamma < math.inf
            and 0 <= args.cg_rtol < 1):
        parser.error("Invalid sigma range, gamma or relative CG tolerance.")
    if not (0 <= args.dps_tail_steps <= args.dps_base_steps and args.dps_base_steps > 1
            and 0 < args.dps_sigma_min < args.dps_sigma_max and args.dps_rho > 0
            and args.dps_zeta > 0):
        parser.error("Invalid DPS tail schedule or zeta.")
    if args.dps_tail_sigma_max is not None:
        if args.dps_tail_steps < 2:
            parser.error(
                "--dps_tail_sigma_max requires --dps_tail_steps >= 2."
            )
        if not args.dps_sigma_min < args.dps_tail_sigma_max < math.inf:
            parser.error(
                "--dps_tail_sigma_max must be greater than --dps_sigma_min."
            )
    if args.diagnose_patch_score and args.diagnose_levels is None:
        args.diagnose_levels = sorted({0, args.num_noise_levels // 2, args.num_noise_levels - 1})
    if args.diagnose_patch_score and args.pad < 2:
        parser.error("Patch diagnostics require room for at least two shifts.")
    if args.diagnose_levels and (not args.diagnose_patch_score or max(args.diagnose_levels) >= args.num_noise_levels):
        parser.error("Diagnostic levels must be within the schedule and require --diagnose_patch_score.")
    if args.diagnose_dps_jacobian:
        if not args.dps_tail_steps or args.score_mode != "patch":
            parser.error("Jacobian diagnostics require a patch DPS tail.")
        args.jacobian_levels = args.jacobian_levels if args.jacobian_levels is not None else [0, 11, 23]
        args.jacobian_inner_steps = args.jacobian_inner_steps if args.jacobian_inner_steps is not None else [0]
        if max(args.jacobian_levels) >= args.dps_tail_steps or max(args.jacobian_inner_steps) >= args.dps_inner_loops:
            parser.error("Jacobian diagnostic index exceeds the actual DPS tail outer/inner range.")
    elif args.jacobian_levels is not None or args.jacobian_inner_steps is not None:
        parser.error("Jacobian index flags require --diagnose_dps_jacobian.")
    for name in ("model_path", "val_dir", "out_root", "noise_dir", "noise_covariance"):
        if getattr(args, name) is not None:
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    return args


def bool_value(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false.")
    return value.lower() == "true"


def sigma_schedule(args):
    sigmas = [args.sigma_max * (args.sigma_min / args.sigma_max) ** (i / args.num_noise_levels)
              for i in range(args.num_noise_levels + 1)]
    sigmas[-1] = args.sigma_min
    return sigmas


def sampling_schedule(args):
    """Original schedule and handoff calculation, shared with CPU dry-run."""
    if args.fixed47_switch is not None:
        # All cases share EXACTLY the same 47 descending Karras sigma values.
        # The leading sigma is used only to initialize the pULA CG state;
        # sigma[1:] contains all 47 actual update-layer sigmas.
        grid = karras_schedule(47, args.sigma_min, args.sigma_max, args.dps_rho)
        switch = args.fixed47_switch
        return [args.sigma_max, *grid], grid[switch:], (0 if args.fixed47_skip_pula else switch)
    sigmas = sigma_schedule(args)
    if args.dps_tail_steps:
        if args.dps_tail_sigma_max is None:
            dps_sigmas = karras_schedule(args.dps_base_steps, args.dps_sigma_min,
                                         args.dps_sigma_max, args.dps_rho)[-args.dps_tail_steps:]
        else:
            dps_sigmas = karras_schedule(args.dps_tail_steps, args.dps_sigma_min,
                                         args.dps_tail_sigma_max, args.dps_rho)
        pula_levels = sum(sigma >= dps_sigmas[0] for sigma in sigmas[1:])
        if pula_levels < 1:
            raise ValueError("DPS tail begins above the first pULA update level.")
    else:
        dps_sigmas, pula_levels = [], args.num_noise_levels
    return sigmas, dps_sigmas, pula_levels


def dry_run_plan(args):
    sigmas, dps_sigmas, levels = sampling_schedule(args)
    updates = levels * args.langevin_steps
    dps = len(dps_sigmas) * args.dps_inner_loops
    return {**vars(args), "partition_mode": args.grid_mode,
            "fixed47_global_sigmas": sigmas[1:] if args.fixed47_switch is not None else None,
            "pula_levels": levels,
            "sigma_init": sigmas[0], "pula_handoff_sigma": sigmas[levels],
            "dps_first_sigma": dps_sigmas[0] if dps_sigmas else None,
            "pula_updates": updates, "dps_updates": dps, "logical_updates": updates + dps,
            "logical_nfe_per_trajectory": updates + dps, "cg_solves_per_trajectory": 1 + updates,
            "checkpoint_and_data_checked": False, "gpu_executed": False}


def load_measurement(val_dir, index, mask_select, device, data=None):
    import torch
    from inverse_operators import MRI_utils, fftmod

    # Same .pt fields and orientation as DPSHyperEvaluator._load_sample.
    if data is None:
        data = torch.load(Path(val_dir) / f"sample_{index}.pt", map_location="cpu", weights_only=False)
    # 线圈灵敏度图。沿用 PaDIS 的 FFT 调制约定，
    # [None] 加一个 batch 维，得到 [1, 线圈数, H, W]。
    maps = fftmod(data["s_map"].clone())[None].to(device=device, dtype=torch.complex64)
    # 多线圈 k-space。
    kspace = fftmod(data["ksp"].clone())[None].to(device=device, dtype=torch.complex64)
    # mask_select=4，所以取 mask_4。形状是 [1, 1, H, W]。
    mask = data[f"mask_{mask_select}"][None].to(device)
    if mask.shape != (1, 1, *kspace.shape[-2:]) or not ((mask == 0) | (mask == 1)).all().item():
        raise ValueError("pULA requires the PaDIS binary mask shared by all coils.")
    mask = mask.bool()
    # 左边返回测量算子 A，右边返回实际观测 y。
    # 同一张 mask 广播到所有线圈，未采样位置置零。
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


def load_noise(args, index, sample=None):
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
        if sample is None:
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


def claim_config(path, config):
    # Publish the initial config atomically so concurrent runs cannot replace it.
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        tmp = Path(file.name)
        try:
            file.write(json.dumps(config, indent=2, allow_nan=False) + "\n")
            file.close()
            os.link(tmp, path)
        except FileExistsError as error:
            raise ValueError(f"Another run created {path}. Check its pula_convention or use a different OUT_ROOT.") from error
        finally:
            tmp.unlink()


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
    if args.dry_run:
        plan = dry_run_plan(args)
        if args.consensus_mode != "off":
            plan.update(algorithm="annealed_multi_partition_consensus_v1",
                        pula_updates=0, dps_updates=0, logical_updates=47,
                        logical_nfe_per_trajectory=235, denoiser_evaluations=235,
                        cg_solves_per_trajectory=48, consensus_levels=47,
                        consensus_updates=47)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if args.score_mode == "patch":
        # Keep the original patch import order, before any torch/project import.
        sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "train/padis-mri")]
    else:
        prepare_fullimage_environment()
    import torch

    if args.num_trajectories is not None:
        from eval_pula.trajectories import run_trajectories
        return run_trajectories(args)

    import dnnlib
    from eval_pula.recon_pula import PULASystem, reconstruct, whiten_measurement
    if args.consensus_mode != "off":
        from functools import partial
        from eval_pula.recon_consensus import reconstruct_consensus
        reconstruct = partial(reconstruct_consensus, mode=args.consensus_mode)
    if args.score_mode == "patch":
        from eval_pula.score_opa import PatchScore, patch_shifts

        shifts = [] if args.grid_mode == "random_step" else patch_shifts(
            args.grid_mode, args.langevin_steps, args.pad, args.grid_seed,
        )
        diagnostic_shifts = patch_shifts("cycle_k", max(2, args.langevin_steps), args.pad, args.grid_seed) if args.diagnose_patch_score else []
    else:
        from eval_pula.score_fullimage import FullImageScoreOracle, checkpoint_environment

        # K iteration slots for the unchanged sampler, without any patch grid.
        shifts, diagnostic_shifts = [None] * args.langevin_steps, []
    # BART uses sigma_max at initialization and N lower, log-spaced corrector levels.
    sigmas, dps_sigmas, pula_levels = sampling_schedule(args)
    if args.fixed47_switch is not None:
        if args.fixed47_patch_policy == "shared_cycle_k":
            # Same K partition offsets at every noise level for BOTH operators.
            # For 47x3 select the first 3 of the shared 5 offsets at EACH level.
            # Existing 47x5 paths are byte-for-byte equivalent to the old rule.
            dps_shifts = shifts[:args.dps_inner_loops] * len(dps_sigmas)
        else:
            # Native DPS random partitions, but tied to global sigma level.
            all_dps_shifts = random_shifts(47 * args.dps_inner_loops, args.pad, args.grid_seed)
            dps_shifts = all_dps_shifts[args.fixed47_switch * args.dps_inner_loops:]
    else:
        dps_shifts = (random_shifts(len(dps_sigmas) * args.dps_inner_loops, args.pad, args.grid_seed)
                      if dps_sigmas else [])
    # Handoff saving requests an additional artifact for the same trajectory.
    pula_shifts_by_level = None
    if args.score_mode == "patch" and args.grid_mode == "random_step":
        steps = args.langevin_steps
        offsets = random_shifts(pula_levels * steps, args.pad, f"pula:{args.grid_seed}")
        pula_shifts_by_level = [
            offsets[start:start + steps] for start in range(0, len(offsets), steps)
        ]
        shifts = pula_shifts_by_level[0]
    request = {key: value for key, value in vars(args).items()
               if key not in ("sample_indices", "out_root", "save_pula_handoff", "dry_run")}
    if args.consensus_mode == "off":
        request.pop("consensus_mode", None)  # Preserve legacy resume identity.
    if not args.fixed47_skip_pula:
        request.pop("fixed47_skip_pula", None)  # All prior configs stay identical.
    if args.fixed47_switch is None:
        request.pop("fixed47_switch", None)  # Keep old configs resume-compatible.
        request.pop("fixed47_patch_policy", None)
    if not args.diagnose_dps_jacobian:
        for key in ("diagnose_dps_jacobian", "jacobian_levels", "jacobian_inner_steps"):
            request.pop(key)
    for key in ("num_trajectories", "trajectory_batch_size", "trajectory_seed_start",
                "aggregate_counts", "save_each_trajectory", "measure_cuda_memory"):
        request.pop(key)
    if args.pula_dc_guide == "none":
        for key in ("pula_dc_guide", "pula_dc_last_levels",
                    "pula_dc_guide_weight", "pula_dc_fd_rel_eps", "pula_dc_mix"):
            request.pop(key, None)  # Preserve legacy resume signatures.
    if args.pula_variant != "balanced_full":
        request.pop("pa_block_size")
        request.pop("pa_epsilon")
    if args.score_mode == "patch":
        request.pop("score_mode")  # Preserve existing config/resume comparisons.
    request.update(
        sampler="opa_pula_dps_tail_v1" if dps_sigmas else "opa_pula_v2", trajectories=1,
        main_output="pula_state_after_original_padis_dps_tail" if dps_sigmas else "single_trajectory_final_state",
        gradient="paired_real_euclidean" if args.pula_convention == "padis_real" else "complex_wirtinger",
        sigma_units="per_real_channel_std",
        initialization="perturb_and_solve_gaussian_approximation", cg_initial="bart_image_term",
        rng_policy="fixed_per_sample_private_generator", partition_mode=args.grid_mode, sigma_init=sigmas[0],
        sigma_levels=sigmas[1:pula_levels + 1], pula_levels=pula_levels,
        pula_handoff_sigma=sigmas[pula_levels], dps_tail_sigmas=dps_sigmas,
        shifts=[None if shift is None else list(shift) for shift in shifts], diagnostic_shifts=[list(shift) for shift in diagnostic_shifts],
        dps_shifts=[list(shift) for shift in dps_shifts],
    )
    if args.consensus_mode != "off":
        request.update(sampler="opa_annealed_multi_partition_consensus_v1",
                       main_output="consensus_quadratic_dc_fov",
                       consensus_denoiser_evaluations=235, consensus_cg_solves=48)
    if args.fixed47_switch is not None:
        request.update(fixed47_global_sigmas=sigmas[1:],
                       fixed47_initialization="same_pula_cg_initialization_all_cases",
                       fixed47_partition_policy=args.fixed47_patch_policy)
    if args.score_mode == "patch":
        request.update(patch_batch=int(os.environ.get("PADIS_PATCH_BATCH", 49)),
                       patch_checkpoint=os.environ.get("PADIS_PATCH_CHECKPOINT", "0"))
    else:
        for key in ("grid_mode", "grid_seed", "pad", "psize"):
            request.pop(key)
        request.update(sampler="fullimage_pula_v2", score_oracle="FullImageScoreOracle",
                       patch_stitching=False, position_encoding=False)
    if pula_shifts_by_level is not None:
        request["pula_shifts_by_level"] = [
            [list(shift) for shift in level] for level in pula_shifts_by_level
        ]
        print(
            f"pULA partitions: {len(offsets)} random draws, {len(set(offsets))} unique",
            flush=True,
        )
    if args.score_mode == "patch" and request["patch_batch"] < 1:
        raise ValueError("PADIS_PATCH_BATCH must be positive.")
    oracle = None
    for index in args.sample_indices:
        sample_dir = Path(args.out_root) / (args.grid_mode if args.score_mode == "patch" else "FullImage") / f"sample_{index}"
        out = sample_dir / "evaluate"
        config_path = out / "config.json"
        expected = {**request, "sample_index": index}
        config = read_json(config_path)
        if isinstance(config, dict):
            config.setdefault("pula_convention", "padis_real")
            config.setdefault("pula_variant", "baseline")
            config.setdefault("partition_mode", config.get("grid_mode", args.grid_mode))
            if config["pula_convention"] != args.pula_convention:
                raise ValueError(f"Existing pula_convention={config['pula_convention']} for {sample_dir} "
                                 f"differs from {args.pula_convention}. Use a different OUT_ROOT.")
        if config is not None and (not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items())):
            raise ValueError(f"Existing configuration differs for {sample_dir}. Use a different OUT_ROOT.")
        if config is None and out.exists() and any(out.iterdir()):
            raise ValueError(f"Cannot verify ownership of {out} without its configuration. Use a different OUT_ROOT.")

        name = f"recon_patch_{index}.npy"
        recon_path = out / "recons" / name
        handoff_path = out / "pula_handoff.npy"
        terminal_path = out / "terminal_denoised" / name
        metric_path = out / "comp_plots" / "results.json"
        timing_path = out / "timing.json"
        diagnostic_path = out / "patch_score_diagnostics.json"
        jacobian_path = out / "jacobian_diagnostics.json"
        reconstructed = (valid_image(recon_path, args.image_size) and isinstance(read_json(timing_path), dict)
                         and (not args.save_pula_handoff or valid_image(handoff_path, args.image_size))
                         and (not args.terminal_denoise or valid_image(terminal_path, args.image_size))
                         and (not args.diagnose_patch_score or isinstance(read_json(diagnostic_path), list))
                         and (not args.diagnose_dps_jacobian or isinstance(read_json(jacobian_path), dict)))
        if reconstructed and valid_metrics(metric_path, index):
            print(f"Skip sample {index}: reconstruction and metrics verified "
                  f"(pula_convention={args.pula_convention}).", flush=True)
            if args.save_pula_handoff:
                print(f"pULA handoff: {handoff_path}", flush=True)
            continue

        if not reconstructed:
            covariance, calibration = load_noise(args, index)
            if config is not None and config.get("noise_calibration") != calibration:
                raise ValueError(f"Noise calibration differs for {sample_dir}. Use a different OUT_ROOT.")
        if not torch.cuda.is_available():
            if args.score_mode == "full":
                raise RuntimeError("Full-image pULA inference requires CUDA.")
            raise RuntimeError("PaDIS patch inference requires CUDA. Use tools/test_pula_numerics.py for CPU verification.")
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        out.mkdir(parents=True, exist_ok=True)
        if config is None:
            claim_config(config_path, {**expected, "noise_calibration": calibration,
                                       "gpu": torch.cuda.get_device_name(device)})
        with (sample_dir / "run.log").open("a", buffering=1) as log:
            def report(message):
                print(message, flush=True)
                print(message, file=log)

            def save_handoff(state):
                save_image(handoff_path, state)
                report(f"Saved pULA handoff: {handoff_path}")

            report(f"Sample {index} | pula_convention={args.pula_convention} pula_variant={args.pula_variant} partition_mode={args.grid_mode}")
            if not reconstructed:
                metric_path.unlink(missing_ok=True)
                timing_path.unlink(missing_ok=True)
                report(f"Sample {index} | {args.grid_mode if args.score_mode == 'patch' else 'FullImage'} | noise calibration {calibration}")
                if args.score_mode == "full":
                    report("score_mode=full\nscore_oracle=FullImageScoreOracle\n"
                           "patch_stitching=False\nposition_encoding=False")
                    report(f"checkpoint={args.model_path} pula_convention={args.pula_convention} "
                           f"sigma_max={args.sigma_max} sigma_min={args.sigma_min} "
                           f"N={args.num_noise_levels} K={args.langevin_steps} gamma={args.gamma} "
                           f"cg_iters={args.cg_iters} cg_rtol={args.cg_rtol} seed={args.seed}")
                if oracle is None:
                    report(f"Loading {args.model_path}")
                    if args.score_mode == "patch":
                        with dnnlib.util.open_url(args.model_path, verbose=False) as file:
                            net = pickle.load(file)["ema"].to(device).eval()
                        oracle = PatchScore(net, args.image_size, args.pad, args.psize, device)
                    else:
                        net = load_fullimage_checkpoint(args.model_path, device)
                        oracle = FullImageScoreOracle(net)
                        oracle.checkpoint_environment = checkpoint_environment(net)
                        report(f"Checkpoint environment: {oracle.checkpoint_environment}")
                raw_operator, raw_measurement = load_measurement(args.val_dir, index, args.mask_select, device)
                if raw_measurement.shape[-2:] != (args.image_size, args.image_size):
                    raise ValueError("Measurement FOV differs from --image_size.")
                operator, measurement = whiten_measurement(raw_operator, raw_measurement, covariance,
                                                           convention=args.pula_convention)

                generator = torch.Generator(device=device).manual_seed(args.seed)
                system = PULASystem(operator, measurement, generator, args.cg_iters, args.cg_rtol,
                                    convention=args.pula_convention)
                if args.score_mode == "full":
                    oracle.first_forward = None
                if dps_sigmas:
                    report(f"Hybrid handoff: pULA {pula_levels} levels through sigma={sigmas[pula_levels]:.6g}; "
                           f"DPS last {len(dps_sigmas)} levels from sigma={dps_sigmas[0]:.6g} "
                           f"to {dps_sigmas[-1]:.6g}, inner={args.dps_inner_loops}, zeta={args.dps_zeta}")
                jacobian_records = [] if args.diagnose_dps_jacobian else None
                state, terminal, timing, diagnostics = reconstruct(
                    system, oracle, sigmas, shifts, args.gamma, diagnose_levels=args.diagnose_levels or (),
                    diagnostic_shifts=diagnostic_shifts, terminal=args.terminal_denoise, report=report,
                    pula_levels=pula_levels, pula_shifts_by_level=pula_shifts_by_level,
                    dps_operator=raw_operator, dps_measurement=raw_measurement,
                    dps_sigmas=dps_sigmas, dps_shifts=dps_shifts,
                    dps_inner_loops=args.dps_inner_loops, dps_zeta=args.dps_zeta,
                    jacobian_levels=args.jacobian_levels if args.diagnose_dps_jacobian else (),
                    jacobian_inner_steps=args.jacobian_inner_steps if args.diagnose_dps_jacobian else (),
                    jacobian_records=jacobian_records,
                    save_handoff=save_handoff if args.save_pula_handoff else None,
                    pula_variant=args.pula_variant, pa_block_size=args.pa_block_size, pa_epsilon=args.pa_epsilon,
                    pula_dc_guide=args.pula_dc_guide,
                    pula_dc_last_levels=args.pula_dc_last_levels,
                    pula_dc_guide_weight=args.pula_dc_guide_weight,
                    pula_dc_mix=args.pula_dc_mix,
                    pula_dc_fd_rel_eps=args.pula_dc_fd_rel_eps,
                )
                if args.score_mode == "full":
                    iterations = [timing["initialization_cg"]["iterations"]] + [
                        row["iterations"] for row in timing["update_cg"]]
                    timing.update(score_mode="full", score_oracle="FullImageScoreOracle",
                                  patch_stitching=False, position_encoding=False,
                                  cg_iteration_range=[min(iterations), max(iterations)],
                                  checkpoint_environment=oracle.checkpoint_environment,
                                  **oracle.first_forward)
                    report(f"FullImage first forward: {oracle.first_forward}")
                    report(f"network_forwards={timing['network_forwards']} cg_solves={timing['cg_solves']} "
                           f"cg_iteration_range={timing['cg_iteration_range']}")
                timing["partition_mode"] = args.grid_mode
                if timing.get("sigma_data") is not None:
                    stored_config = read_json(config_path)
                    stored_config["sigma_data"] = float(timing["sigma_data"])
                    write_json(config_path, stored_config)
                save_image(recon_path, state)
                if terminal is not None:
                    save_image(terminal_path, terminal)
                if args.diagnose_patch_score:
                    write_json(diagnostic_path, diagnostics)
                if args.diagnose_dps_jacobian:
                    from eval_pula.jacobian_diagnostics import write_outputs
                    from tools.analyze_jacobian_diagnostics import summarize
                    metadata = {
                        "model_path": args.model_path, "val_dir": args.val_dir,
                        "sample_index": index, "mask_select": args.mask_select,
                        "pula_convention": args.pula_convention, "pula_variant": args.pula_variant,
                        "grid_mode": args.grid_mode, "seed": args.seed, "grid_seed": args.grid_seed,
                        "pad": args.pad, "image_size": args.image_size, "psize": args.psize,
                        "pula_levels": pula_levels, "pula_updates": timing["langevin_updates"],
                        "dps_tail_sigmas": dps_sigmas,
                        "dps_inner_loops": args.dps_inner_loops,
                        "dps_shifts": [list(s) for s in dps_shifts],
                        "dps_updates": timing["dps_updates"],
                        "network_forwards": timing["network_forwards"],
                        "network_backprops": timing["network_backprops"],
                        "extra_network_forwards": 0, "extra_network_backwards": 0,
                        "extra_raw_mri_adjoint_calls": len(jacobian_records),
                        "mri_operator": "raw_unwhitened_dps_operator",
                    }
                    write_outputs(out, jacobian_records, metadata)
                    summarize(jacobian_path, out / "jacobian_cosines.png")
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

