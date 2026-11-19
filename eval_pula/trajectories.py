"""Independent trajectory systems sharing one unchanged pULA loop and patch net."""

from collections import defaultdict
import hashlib
import json
import os
import pickle
from pathlib import Path
from time import perf_counter

import torch

from eval_pula.recon_pula import PULASystem, reconstruct, whiten_measurement
from eval_pula.score_batch import TrajectoryPatchScore, synchronized_stage
from eval_pula.score_opa import patch_shifts


class TrajectorySystems:
    """Serial independent RHS/CG calls, with a batched state for the shared loop."""
    def __init__(self, systems):
        self.systems = systems
        self.y = systems[0].y
        self.convention = systems[0].convention
        self.stages = defaultdict(float)

    @property
    def forward_calls(self):
        return sum(system.forward_calls for system in self.systems)

    @property
    def adjoint_calls(self):
        return sum(system.adjoint_calls for system in self.systems)

    def system_sigma(self, sigma):
        return self.systems[0].system_sigma(sigma)

    def initialization_rhs(self, sigma):
        with synchronized_stage(self.y.device, self.stages, "rhs_and_cg_seconds"):
            pairs = [system.initialization_rhs(sigma) for system in self.systems]
            return tuple(torch.cat(column) for column in zip(*pairs))

    def update_rhs(self, state, score, sigma, gamma):
        with synchronized_stage(self.y.device, self.stages, "rhs_and_cg_seconds"):
            pairs = [system.update_rhs(state[j:j+1], score[j:j+1], sigma, gamma)
                     for j, system in enumerate(self.systems)]
            return tuple(torch.cat(column) for column in zip(*pairs))

    def solve(self, rhs, initial, sigma):
        with synchronized_stage(self.y.device, self.stages, "rhs_and_cg_seconds"):
            pairs = [system.solve(rhs[j:j+1], initial[j:j+1], sigma)
                     for j, system in enumerate(self.systems)]
            states, records = zip(*pairs)
            relative = [r["relative_residual"] for r in records if r["relative_residual"] is not None]
            info = {"iterations": sum(r["iterations"] for r in records),
                    "residual_norm": max(r["residual_norm"] for r in records),
                    "relative_residual": max(relative) if relative else None,
                    "per_trajectory": list(records)}
            return torch.cat(states), info


def partitions(args, partition_seed):
    from eval_pula.run import random_shifts

    if args.grid_mode == "random_step":
        flat = random_shifts(args.num_noise_levels * args.langevin_steps, args.pad,
                             f"pula:{partition_seed}")
        return [flat[i:i + args.langevin_steps]
                for i in range(0, len(flat), args.langevin_steps)]
    shifts = patch_shifts(args.grid_mode, args.langevin_steps, args.pad, partition_seed)
    return [list(shifts) for _ in range(args.num_noise_levels)]


def partition_digest(sequence):
    return hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest()


@torch.no_grad()
def sample_batch(operator, measurement, oracle, args, seeds, sequences, sigmas, report=print):
    systems = [PULASystem(operator, measurement, torch.Generator(device=measurement.device).manual_seed(seed),
                          args.cg_iters, args.cg_rtol, convention=args.pula_convention) for seed in seeds]
    facade = TrajectorySystems(systems)
    # [level][step][trajectory][row,col]; no averaging of grids or states.
    shifts_by_level = [[list(shifts) for shifts in zip(*(seq[level] for seq in sequences))]
                       for level in range(args.num_noise_levels)]
    state, _, timing, _ = reconstruct(
        facade, oracle, sigmas, shifts_by_level[0], args.gamma,
        pula_shifts_by_level=shifts_by_level, report=report,
    )
    records = []
    for j, seed in enumerate(seeds):
        initial = timing["initialization_cg"]["per_trajectory"][j]
        updates = [row["per_trajectory"][j] for row in timing["update_cg"]]
        iterations = [initial["iterations"]] + [row["iterations"] for row in updates]
        records.append({"seed": seed, "completed_updates": len(updates),
                        "cg_solve_count": 1 + len(updates), "cg_iteration_range": [min(iterations), max(iterations)],
                        "cg_total_iterations": sum(iterations), "initialization_cg": initial, "update_cg": updates})
    return state, records, timing, dict(facade.stages)


def run_trajectories(args):
    from eval_pula.run import load_measurement, load_noise, save_image, sigma_schedule, write_json
    from utils import post_eval_normalize
    import dnnlib

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run_manifest.json").exists() or any((root / f"sample_{i}").exists() for i in args.sample_indices):
        raise FileExistsError(f"Existing trajectory results at {root}; choose a new OUT_ROOT.")
    if not torch.cuda.is_available():
        raise RuntimeError("Block trajectory inference requires CUDA; use the CPU fixture tests separately.")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    end_to_end_start = perf_counter()
    seed_start = args.seed if args.trajectory_seed_start is None else args.trajectory_seed_start
    seeds = list(range(seed_start, seed_start + args.num_trajectories))
    partition_seeds = list(range(args.grid_seed, args.grid_seed + args.num_trajectories))
    sequences = [partitions(args, seed) for seed in partition_seeds]
    sigmas = sigma_schedule(args)
    counts = sorted({s for s in args.aggregate_counts if s <= len(seeds)} | {len(seeds)})
    request = {**vars(args), "seeds": seeds, "partition_seeds": partition_seeds,
               "sigma_init": sigmas[0], "sigma_levels": sigmas[1:], "aggregate_counts": counts,
               "logical_updates_per_trajectory": args.num_noise_levels * args.langevin_steps,
               "logical_updates_total": args.num_trajectories * args.num_noise_levels * args.langevin_steps,
               "cg_batched": False, "network_trajectory_batched": True,
               "source_base_commit": "0e5b8cc4dc216895c451dbffc079b6a0fe56d698",
               "network_patch_microbatch": os.environ.get("PADIS_PATCH_BATCH", "all packed patches"),
               "foreground_metrics": "not available in this commit's post_eval_normalize interface",
               "status": "running"}
    write_json(root / "run_manifest.json", request)
    print(f"Block pULA: checkpoint={args.model_path}\nvalidation={args.val_dir}\n"
          f"mask=R{args.mask_select} samples={args.sample_indices} N={args.num_noise_levels} K={args.langevin_steps} "
          f"sigma={args.sigma_max}->{args.sigma_min} gamma={args.gamma} CG={args.cg_iters}/{args.cg_rtol}\n"
          f"trajectories={len(seeds)} batch_size={args.trajectory_batch_size} seeds={seeds} "
          f"grid_mode={args.grid_mode} partition_seeds={partition_seeds}", flush=True)
    try:
        torch.cuda.synchronize(device)
        start = perf_counter()
        with dnnlib.util.open_url(args.model_path, verbose=False) as file:
            net = pickle.load(file)["ema"].to(device).eval()
        torch.cuda.synchronize(device)
        checkpoint_seconds = perf_counter() - start
        oracle = TrajectoryPatchScore(net, args.image_size, args.pad, args.psize, device)
        warm_batch = min(args.trajectory_batch_size, len(seeds))
        start = perf_counter()
        with torch.random.fork_rng(devices=[device.index]):
            warm_state = torch.zeros((warm_batch, 1, args.image_size, args.image_size),
                                     dtype=torch.complex64, device=device)
            oracle.denoise(warm_state, sigmas[1], [(0, 0)] * warm_batch)
            torch.cuda.synchronize(device)
        del warm_state
        warmup_seconds = perf_counter() - start
        run_totals = {"checkpoint_load_seconds": checkpoint_seconds, "warmup_seconds": warmup_seconds,
                      "sampling_wall_seconds": 0., "aggregation_and_evaluation_seconds": 0.}
        sample_stats = []
        for index in args.sample_indices:
            sample_root = root / f"sample_{index}"
            sample_root.mkdir()
            data = torch.load(Path(args.val_dir) / f"sample_{index}.pt", map_location="cpu", weights_only=False)
            covariance, calibration = load_noise(args, index, sample=data)
            raw_operator, raw_measurement = load_measurement(args.val_dir, index, args.mask_select, device, data=data)
            operator, measurement = whiten_measurement(raw_operator, raw_measurement, covariance,
                                                       convention=args.pula_convention)
            if measurement.shape[-2:] != (args.image_size, args.image_size):
                raise ValueError("Measurement FOV differs from --image_size.")
            oracle.reset_stats()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            sample_start = perf_counter()
            images, trajectory_records, batches = [], [], []
            rhs_seconds = 0.
            trajectory_save_seconds = 0.
            for offset in range(0, len(seeds), args.trajectory_batch_size):
                stop = min(offset + args.trajectory_batch_size, len(seeds))
                state, records, timing, stages = sample_batch(
                    operator, measurement, oracle, args, seeds[offset:stop], sequences[offset:stop], sigmas,
                    report=lambda message: print(f"sample {index} batch {offset}:{stop} | {message}", flush=True))
                images.extend(state.cpu().unbind(0))
                del state
                torch.cuda.synchronize(device)
                completion = perf_counter() - sample_start
                for j, record in enumerate(records, offset):
                    record.update(trajectory_index=j, partition_seed=partition_seeds[j], partition_strategy=args.grid_mode,
                                  partition_shifts=sequences[j], partition_sha256=partition_digest(sequences[j]),
                                  prefix_batch_completion_seconds=completion)
                trajectory_records.extend(records)
                batches.append({"start": offset, "stop": stop, "completion_seconds": completion,
                                "network_batch_calls": timing["network_forwards"],
                                "peak_memory_allocated_gib": timing.get("peak_allocated_gib"),
                                "peak_memory_reserved_gib": timing.get("peak_reserved_gib")})
                rhs_seconds += stages["rhs_and_cg_seconds"]
                # Preserve a finished batch on disk before starting the next one.
                # This I/O is measured separately and excluded from sampling.
                io_start = perf_counter()
                if args.save_each_trajectory:
                    for j in range(offset, stop):
                        out = sample_root / f"trajectory_{j:03d}_seed{seeds[j]}" / "evaluate"
                        save_image(out / "recons" / f"recon_patch_{index}.npy", images[j].unsqueeze(0))
                        write_json(out / "timing.json", trajectory_records[j])
                io_seconds = perf_counter() - io_start
                trajectory_save_seconds += io_seconds
                sample_start += io_seconds
            torch.cuda.synchronize(device)
            sampling_seconds = perf_counter() - sample_start
            run_totals["sampling_wall_seconds"] += sampling_seconds
            iterations = [n for r in trajectory_records for n in r["cg_iteration_range"]]
            stats = {"N": args.num_noise_levels, "K": args.langevin_steps,
                     "logical_updates_per_trajectory": args.num_noise_levels * args.langevin_steps,
                     "logical_updates_total": sum(r["completed_updates"] for r in trajectory_records),
                     "num_trajectories": len(seeds), "trajectory_batch_size": args.trajectory_batch_size,
                     "cg_iters": args.cg_iters, "cg_rtol": args.cg_rtol,
                     "sigma_max": args.sigma_max, "sigma_min": args.sigma_min, "gamma": args.gamma,
                     "sigma_levels": sigmas[1:], "pula_convention": args.pula_convention,
                     "checkpoint_load_seconds": checkpoint_seconds, "warmup_seconds": warmup_seconds,
                     "trajectory_save_seconds": trajectory_save_seconds,
                     "sampling_wall_seconds": sampling_seconds,
                     "effective_sampling_seconds_per_trajectory": sampling_seconds / len(seeds),
                     "cg_solve_count": sum(r["cg_solve_count"] for r in trajectory_records),
                     "cg_iteration_range": [min(iterations), max(iterations)],
                     "network_batch_calls": oracle.network_batch_calls, "patches_processed": oracle.patches_processed,
                     "peak_memory_allocated_gib": max(b["peak_memory_allocated_gib"] for b in batches) if args.measure_cuda_memory else None,
                     "peak_memory_reserved_gib": max(b["peak_memory_reserved_gib"] for b in batches) if args.measure_cuda_memory else None,
                     "stages": {**dict(oracle.stages), "rhs_and_cg_seconds": rhs_seconds}, "batches": batches}
            manifest = {**request, "sample_index": index, "noise_calibration": calibration,
                        "base_patch_indices": [[int(v) for v in entry] for entry in oracle.indices],
                        "patches_per_partition": len(oracle.indices), "trajectory_records": trajectory_records}
            write_json(sample_root / "manifest.json", manifest)
            start = perf_counter()
            curve, cumulative = [], torch.zeros_like(images[0])
            for j, image in enumerate(images, 1):
                cumulative = cumulative + image
                if j not in counts:
                    continue
                aggregate_start = perf_counter()
                out = sample_root / f"S{j:02d}_mean" / "evaluate"
                recon_dir = out / "recons"
                save_image(recon_dir / f"recon_patch_{index}.npy", (cumulative / j).unsqueeze(0))
                post_eval_normalize(str(recon_dir), args.val_dir, str(out / "comp_plots"),
                                    mask_select=args.mask_select, sample_cache={index: data})
                # Reuse the existing evaluator's JSON conversion of NumPy scalars.
                result = json.loads((out / "comp_plots" / "results.json").read_text())
                write_json(out / "results.json", result)
                if str(index) not in result.get("per_image", {}):
                    raise RuntimeError(f"Metrics missing for sample {index}, S{j}.")
                curve.append({"count": j, **result["per_image"][str(index)],
                              "prefix_batch_completion_seconds": trajectory_records[j-1]["prefix_batch_completion_seconds"],
                              "aggregation_and_evaluation_seconds": perf_counter() - aggregate_start,
                              "actual_result_ready_end_to_end_seconds": perf_counter() - end_to_end_start})
            torch.cuda.synchronize(device)
            stats["aggregation_and_evaluation_seconds"] = perf_counter() - start
            run_totals["aggregation_and_evaluation_seconds"] += stats["aggregation_and_evaluation_seconds"]
            stats["end_to_end_seconds"] = perf_counter() - end_to_end_start
            sample_stats.append(stats)
            write_json(sample_root / "timing.json", stats)
            write_json(sample_root / "quality_time_curve.json", curve)
            write_json(sample_root / "manifest.json", {**manifest, "status": "complete"})
            print(f"sample {index} summary: {json.dumps(stats)}", flush=True)
            # No trajectory/system holds this sample's GPU data between samples.
            del operator, measurement, raw_operator, raw_measurement
        run_totals["end_to_end_seconds"] = perf_counter() - end_to_end_start
        run_totals.update(num_trajectories=len(seeds), num_samples=len(args.sample_indices),
                          N=args.num_noise_levels, K=args.langevin_steps,
                          sigma_max=args.sigma_max, sigma_min=args.sigma_min, gamma=args.gamma,
                          cg_iters=args.cg_iters, cg_rtol=args.cg_rtol, seeds=seeds,
                          partition_strategy=args.grid_mode,
                          cg_solve_count=sum(s["cg_solve_count"] for s in sample_stats),
                          cg_iteration_range=[min(s["cg_iteration_range"][0] for s in sample_stats),
                                              max(s["cg_iteration_range"][1] for s in sample_stats)],
                          trajectory_batch_size=args.trajectory_batch_size,
                          effective_sampling_seconds_per_trajectory=run_totals["sampling_wall_seconds"] / (len(seeds) * len(args.sample_indices)),
                          peak_memory_allocated_gib=max(s["peak_memory_allocated_gib"] for s in sample_stats) if args.measure_cuda_memory else None,
                          peak_memory_reserved_gib=max(s["peak_memory_reserved_gib"] for s in sample_stats) if args.measure_cuda_memory else None,
                          logical_updates_total=sum(s["logical_updates_total"] for s in sample_stats),
                          network_batch_calls=sum(s["network_batch_calls"] for s in sample_stats),
                          patches_processed=sum(s["patches_processed"] for s in sample_stats),
                          trajectory_save_seconds=sum(s["trajectory_save_seconds"] for s in sample_stats))
        write_json(root / "timing.json", run_totals)
        write_json(root / "run_manifest.json", {**request, "status": "complete"})
    except torch.cuda.OutOfMemoryError as error:
        write_json(root / "failure.json", {"status": "CUDA_OOM", "message": str(error),
                                           "trajectory_batch_size": args.trajectory_batch_size})
        raise SystemExit(42) from error
