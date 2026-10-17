#!/usr/bin/env python3
"""Coordinate-only E0/AGGA diagnosis. Never loads a network or checkpoint."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDITED_COMMIT = "4f58b8569a75e7e9c114b9944533e2dddebcc2cc"
BATCH_MUL = {512: 1, 256: 1, 128: 4, 96: 8, 64: 16, 56: 16,
             48: 16, 32: 32, 24: 32, 16: 64}
SOURCE_PATHS = [
    "train/padis-mri/train.py", "train/padis-mri/train_agg_simplified.py",
    "train/padis-mri/training/training_loop.py",
    "train/padis-mri/training/patch_loss.py",
    "train/padis-mri/training/patch_loss_simplified_agg.py",
    "train/padis-mri/training/agg_sampler.py",
    "train/padis-mri/training/dataset.py", "torch_utils/misc.py",
    "tools/visualize_agga_core_panels.py",
]
METRICS = ("R64", "Rall", "R64_total")
DEFAULT_DATA = ("/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/"
                "brain_train_d384_s200/32dB/noisy.pt")


def load_agg_module():
    path = ROOT / "train/padis-mri/training/agg_sampler.py"
    spec = importlib.util.spec_from_file_location("oc_diagnostic_original_agg", path)
    module = importlib.util.module_from_spec(spec)
    old_flag = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_flag
    return module


def infinite_indices(dataset_size, seed, window_size=0.5):
    """Single-rank misc.InfiniteSampler algorithm, without importing Torch."""
    order = np.arange(dataset_size)
    rnd = np.random.RandomState(seed)
    rnd.shuffle(order)
    window = int(np.rint(dataset_size * window_size))
    idx = 0
    while True:
        i = idx % dataset_size
        yield int(order[i])
        if window >= 2:
            j = (i - rnd.randint(window)) % dataset_size
            order[i], order[j] = order[j], order[i]
        idx += 1


def normalized_probs(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("patch_probs must be finite, nonnegative, with positive sum")
    return p / p.sum()


def schedule(dataset_size, sample_index, patch_sizes, patch_probs,
             batch_size, batch_gpu, seed, final_nimg, padded_size):
    """Yield complete optimization iterations; only retain sample0 events.

    The index stream is consumed once for both modes. Spatial RNG never enters
    this generator. cur_nimg is the logged counter, not the event count.
    """
    probs = normalized_probs(patch_probs)
    multipliers = dict(BATCH_MUL)
    rounds = batch_size // batch_gpu
    increment = int(batch_size * float(np.sum([
        p * multipliers[s] for s, p in zip(patch_sizes, probs)])))
    if increment < 1:
        raise ValueError("cur_nimg increment must be positive")
    scale_rng = np.random.RandomState(seed)
    indices = infinite_indices(dataset_size, seed)
    total = cur_nimg = iteration = 0
    digest = hashlib.sha256()
    while cur_nimg < final_nimg:
        iteration += 1
        events = []
        for round_index in range(rounds):
            size = int(scale_rng.choice(patch_sizes, p=probs))
            n = multipliers[size] * batch_gpu
            batch = np.fromiter((next(indices) for _ in range(n)), dtype=np.int64, count=n)
            digest.update(np.asarray([iteration, round_index, size], dtype="<i8").tobytes())
            digest.update(batch.astype("<i8", copy=False).tobytes())
            for offset in np.flatnonzero(batch == sample_index):
                events.append({"iteration": iteration, "accumulation_round": round_index,
                               "round_offset": int(offset), "global_patch_event": total + int(offset) + 1,
                               "patch_size": size})
            total += n
        cur_nimg += increment
        yield {"cur_nimg": cur_nimg, "actual_kimg": cur_nimg / 1000,
               "total_patch_events": total, "iteration": iteration,
               "schedule_sha256": digest.hexdigest(), "events": events,
               "progress_increment": increment}


def intersection_area(a, b, size):
    return max(0, size - abs(int(a[0]) - int(b[0]))) * max(
        0, size - abs(int(a[1]) - int(b[1])))


class OverlapIndex:
    """Exact existential overlap index, with one historical witness per site.

    Every historical rectangle paints all legal top-left sites whose individual
    overlap with that rectangle meets the threshold. Union of these sites is
    an existential query, never a sum/union of pixel intersection areas.
    """
    def __init__(self, height, width, size, threshold=0.5):
        self.size = int(size)
        self.height, self.width = int(height), int(width)
        if size <= 0 or size > min(height, width):
            raise ValueError("invalid patch geometry")
        fraction = Fraction(str(threshold))
        if not 0 < fraction <= 1:
            raise ValueError("overlap_threshold must be in (0,1]")
        self.required = (size * size * fraction.numerator + fraction.denominator - 1) // fraction.denominator
        shape = (height - size + 1, width - size + 1)
        self.witness = np.full(shape, -1, dtype=np.int64)
        self.seen = np.zeros(shape, dtype=bool)
        self.n = self.overlap = 0
        self.spans = []
        for dy in range(-size + 1, size):
            h = size - abs(dy)
            dxmax = size - (self.required + h - 1) // h
            if dxmax >= 0:
                self.spans.append((dy, dxmax))

    def observe(self, top, left, event_id):
        top, left = int(top), int(left)
        if not (0 <= top < self.witness.shape[0] and 0 <= left < self.witness.shape[1]):
            raise ValueError("patch top-left outside legal canvas")
        previous = int(self.witness[top, left])
        self.n += 1
        self.overlap += int(previous >= 0)
        # Repeated coordinates add no new eligible query sites.
        if not self.seen[top, left]:
            for dy, dxmax in self.spans:
                row = top + dy
                if 0 <= row < self.witness.shape[0]:
                    lo = max(0, left - dxmax)
                    hi = min(self.witness.shape[1], left + dxmax + 1)
                    self.witness[row, lo:hi] = event_id
            self.seen[top, left] = True
        return previous


def ratio(count, denominator):
    return 100.0 * count / denominator if denominator else None


def summarize(indices):
    n = sum(index.n for index in indices.values())
    overlaps = sum(index.overlap for index in indices.values())
    i64 = indices.get(64)
    n64 = i64.n if i64 else 0
    o64 = i64.overlap if i64 else 0
    row = {"sample0_total_patches": n, "R64": ratio(o64, n64),
           "Rall": ratio(overlaps, n), "R64_total": ratio(o64, n)}
    for size in set(indices) | {16, 32, 64}:
        index = indices.get(size)
        row[f"sample0_n{size}"] = index.n if index else 0
        row[f"sample0_overlap{size}"] = index.overlap if index else 0
    if n and n64:
        assert math.isclose(row["R64_total"], row["R64"] * n64 / n, abs_tol=1e-12)
    assert n == sum(index.n for index in indices.values())
    assert all(0 <= index.overlap <= index.n for index in indices.values())
    return row


def cdf_from_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("invalid sampling probabilities")
    p = p / p.sum()
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    return p, cdf


def draw_locations(cdf, width, uniforms):
    flat = np.searchsorted(cdf, uniforms, side="right")
    return np.column_stack((flat // width, flat % width))


def empirical_check(probabilities, cdf, seed, draws=60000):
    """Check a dense distribution with equal-index bins of adequate counts.

    Individual candidate frequencies are too sparse for a useful per-pixel
    test. The 32-bin check is deterministic and does not alter simulation RNG.
    """
    p = np.asarray(probabilities)
    bins = min(32, p.size)
    groups = np.minimum(np.arange(p.size) * bins // p.size, bins - 1)
    expected = np.bincount(groups, weights=p, minlength=bins)
    flat = np.searchsorted(cdf, np.random.default_rng(seed).random(draws), side="right")
    observed = np.bincount(groups[flat], minlength=bins) / draws
    tolerance = 7 * np.sqrt(expected * (1 - expected) / draws) + 7 / draws
    passed = bool(np.all(np.abs(observed - expected) <= tolerance))
    return {"passed": passed, "draws": draws, "bins": bins,
            "maximum_bin_error": float(np.max(np.abs(observed - expected))),
            "expected_bins": expected.tolist(), "observed_bins": observed.tolist()}


def build_cache(image, sizes, rho, gamma, seed):
    """Call the original score functions and intercept original sampler p.

    Interception is temporary/in-memory and restored in finally. No source
    files are modified. A deterministic dummy multinomial avoids repeated
    expensive sampling, while recording the exact probabilities it receives.
    """
    import torch
    module = load_agg_module()
    maps = module._structure_maps(image)
    height, width = image.shape[-2:]
    cache, report = {}, {}
    for size in sizes:
        shape = (height - size + 1, width - size + 1)
        if maps is None:
            native = torch.full((shape[0] * shape[1],), 1 / (shape[0] * shape[1]), dtype=torch.float32)
        else:
            score = module._region_score(*maps, (size, size), gamma)
            if tuple(score.shape) != shape:
                raise AssertionError("candidate geometry differs from source")
            native = module._sampling_probabilities(score, rho)
        captured = []
        def capture(probabilities, num_samples, *args, **kwargs):
            captured.append(probabilities.detach().clone())
            return torch.tensor([0], dtype=torch.long, device=probabilities.device)
        original = torch.multinomial
        try:
            torch.multinomial = capture
            with torch.random.fork_rng(devices=[]):
                top, left = module.sample_single_locations(image[None], size, rho, gamma)
        finally:
            torch.multinomial = original
        if maps is not None:
            if len(captured) != 1 or not torch.equal(captured[0], native):
                raise AssertionError("cache differs from original sampler probabilities")
            if (int(top[0]), int(left[0])) != (0, 0):
                raise AssertionError("original flattening is not row-major")
        p, cdf = cdf_from_probabilities(native.cpu().numpy())
        check = empirical_check(p, cdf, seed + size)
        if not check["passed"]:
            raise AssertionError("inverse-CDF empirical check failed")
        cache[size] = (p, cdf, shape[1])
        report[str(size)] = {"candidate_shape": list(shape), "probability_sum": float(p.sum()),
            "source_sum_float32": float(native.sum()), "cache_source_max_abs_error": 0.0,
            "float64_renormalization_max_abs_error": float(np.max(np.abs(p - native.cpu().numpy()))),
            "source_probability_sha256": hashlib.sha256(native.cpu().numpy().tobytes()).hexdigest(),
            "uniform_fallback": maps is None, "original_call_intercepted": maps is not None,
            "mapping": "row-major: top=flat//candidate_width, left=flat%candidate_width",
            "empirical_check": check}
    return cache, report


def load_image(path, sample_index, pad_width):
    import torch
    # noisy.pt is an explicitly supplied trusted local training dataset.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload["x_est_gt"]
    if not torch.is_complex(tensor) or tensor.ndim != 3:
        raise ValueError("current ImageFolderDatasetX requires complex x_est_gt [N,H,W]")
    if not 0 <= sample_index < tensor.shape[0]:
        raise ValueError("sample_index outside actual dataset")
    raw_shape = list(tensor.shape)
    # Same operations and float32 conversion as ImageFolderDatasetX.
    channels = torch.view_as_real(tensor[sample_index]).numpy().astype(np.float32).transpose(2, 0, 1)
    padded = np.pad(channels, ((0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode="constant")
    image = torch.from_numpy(padded)
    metadata = {"dataset_size": int(tensor.shape[0]), "x_est_gt_shape": raw_shape,
                "x_est_gt_dtype": str(tensor.dtype), "image_size": raw_shape[-2:],
                "padded_size": list(image.shape[-2:]), "data": str(path.resolve()),
                "preprocessing": "complex -> view_as_real -> numpy float32 CHW -> zero pad",
                "sample_image_sha256": hashlib.sha256(padded.tobytes()).hexdigest()}
    return image, metadata


def git_info():
    def run(*args):
        p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "status": run("status", "--short"), "remote": run("remote", "-v"),
            "audited_commit": AUDITED_COMMIT,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS}}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def simulate(config, cache, magnitude=None, output=None):
    """Both modes consume one immutable schedule and paired spatial uniforms."""
    height, width = config["padded_size"]
    sizes, modes = config["patch_sizes"], config["modes"]
    indices = {m: {s: OverlapIndex(height, width, s, config["overlap_threshold"])
                   for s in sizes} for m in modes}
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 0x4F43]))
    final_kimg = max(config["kimg_checkpoints"])
    targets = set(float(x) for x in config["kimg_checkpoints"])
    targets.update(x for x in (1, 5, 10, 20, 50, 100, 200) if x <= final_kimg)
    targets.update(np.arange(100, final_kimg + 1e-10, config["snapshot_interval"]).tolist())
    targets = sorted(targets)
    target_index = 0
    rows = {m: [] for m in modes}
    event_records = {m: [] for m in modes}
    pairs = {}
    schedule_digest = None
    for step in schedule(config["dataset_size"], config["sample_index"], sizes,
                         config["patch_probs"], config["batch_size"], config["batch_gpu"],
                         config["seed"], math.ceil(final_kimg * 1000), config["padded_size"]):
        events = step["events"]
        # Generate common random numbers in event order, independent of mode.
        uniforms = rng.random(len(events))
        for mode in modes:
            locations = np.empty((len(events), 2), dtype=np.int64)
            for size in sizes:
                offsets = [i for i, event in enumerate(events) if event["patch_size"] == size]
                if offsets:
                    _, cdf, candidate_width = cache[mode][size]
                    locations[offsets] = draw_locations(cdf, candidate_width, uniforms[offsets])
            for event, (top, left) in zip(events, locations):
                event_id = len(event_records[mode])
                size = event["patch_size"]
                witness = indices[mode][size].observe(top, left, event_id)
                record = {**event, "sample0_event_id": event_id, "cur_nimg_after_iteration": step["cur_nimg"],
                          "top": int(top), "left": int(left), "overlap": witness >= 0,
                          "historical_witness_event_id": witness}
                if size == 64 and witness >= 0 and mode not in pairs:
                    history = event_records[mode][witness]
                    pairs[mode] = {"current": dict(record), "historical": dict(history),
                        "intersection_area": intersection_area((top, left), (history["top"], history["left"]), size),
                        "selection": "first qualifying 64x64 event in actual sample0 stream"}
                event_records[mode].append(record)
        while target_index < len(targets) and step["cur_nimg"] >= targets[target_index] * 1000:
            target = targets[target_index]
            for mode in modes:
                rho, gamma = (0., 0.) if mode == "uniform_e0" else (0.65, 1.)
                row = {"mode": mode, "target_kimg": target, "actual_kimg": step["actual_kimg"],
                       "is_formal_checkpoint": target in config["kimg_checkpoints"],
                       "cur_nimg": step["cur_nimg"], "total_patch_events": step["total_patch_events"],
                       **summarize(indices[mode]), "rho": rho, "gamma": gamma,
                       **{k: config[k] for k in ["seed", "dataset_size", "image_size", "padded_size", "pad_width",
                          "patch_probs", "batch_size", "batch_gpu", "overlap_threshold"]},
                       "batch_mul_dict": {s: BATCH_MUL[s] for s in sizes},
                       "schedule_sha256": step["schedule_sha256"],
                       "sampling_implementation": "float64 normalized original probabilities + inverse CDF",
                       "simulation_configuration_source": config["configuration_source"]}
                rows[mode].append(row)
            if len(modes) == 2:
                a, b = (rows[m][-1] for m in modes)
                assert all(a[f"sample0_n{s}"] == b[f"sample0_n{s}"] for s in sizes)
                assert a["schedule_sha256"] == b["schedule_sha256"]
            if target in config["kimg_checkpoints"]:
                print(f"Checkpoint {target:g} kimg -> actual {step['actual_kimg']:.3f}, "
                      f"sample0={sum(i.n for i in indices[modes[0]].values())}", flush=True)
            target_index += 1
        schedule_digest = step["schedule_sha256"]
    comparisons = []
    if {"uniform_e0", "agga"} <= set(modes):
        for a, b in zip(rows["uniform_e0"], rows["agga"]):
            row = {"target_kimg": a["target_kimg"], "actual_kimg": a["actual_kimg"],
                   "is_formal_checkpoint": a["is_formal_checkpoint"]}
            for metric in METRICS:
                row[f"E0_{metric}"] = a[metric]
                row[f"AGGA_{metric}"] = b[metric]
                row[f"AGGA_minus_E0_{metric}_pp"] = b[metric] - a[metric] if a[metric] is not None and b[metric] is not None else None
            comparisons.append(row)
    result = {"rows": rows, "pairs": pairs, "comparison": comparisons,
              "schedule_sha256": schedule_digest, "events": event_records}
    if output is not None:
        for mode in modes:
            folder = output / mode
            folder.mkdir()
            write_csv(folder / "checkpoints.csv", [r for r in rows[mode] if r["is_formal_checkpoint"]])
            write_csv(folder / "curve.csv", rows[mode])
            write_json(folder / "statistics.json", rows[mode])
            write_csv(folder / "sample0_events.csv", event_records[mode])
            write_json(folder / "example_pair.json", pairs.get(mode))
            np.savez_compressed(folder / "probabilities.npz", **{f"p{s}": cache[mode][s][0] for s in sizes})
        (output / "comparison").mkdir()
        write_csv(output / "comparison/comparison.csv", comparisons)
        write_json(output / "comparison/comparison.json", comparisons)
        plot_results(output, result, magnitude, config)
    return result


def plot_results(output, result, magnitude, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["svg.fonttype"] = "none"
    def save(fig, base):
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(str(base) + ".svg", bbox_inches="tight")
        plt.close(fig)
    for mode, rows in result["rows"].items():
        fig, axis = plt.subplots(figsize=(7, 4))
        for metric in METRICS:
            axis.plot([r["actual_kimg"] for r in rows], [np.nan if r[metric] is None else r[metric] for r in rows], label=metric)
            formal = [r for r in rows if r["is_formal_checkpoint"] and r[metric] is not None]
            axis.scatter([r["actual_kimg"] for r in formal], [r[metric] for r in formal], s=22)
        axis.set(xlabel="Training Progress (kimg)", ylabel="Cumulative Overlap Ratio (%)", ylim=(0, 101), title=mode)
        axis.legend()
        axis.grid(alpha=.2)
        save(fig, output / mode / "cumulative_overlap")
    fig, axis = plt.subplots(figsize=(7, 4))
    for mode, rows in result["rows"].items():
        axis.plot([r["actual_kimg"] for r in rows], [np.nan if r["R64"] is None else r["R64"] for r in rows], label={"uniform_e0": "E0 Uniform", "agga": "AGGA"}[mode])
        formal = [r for r in rows if r["is_formal_checkpoint"] and r["R64"] is not None]
        axis.scatter([r["actual_kimg"] for r in formal], [r["R64"] for r in formal], s=22)
    axis.set(xlabel="Training Progress (kimg)", ylabel="Cumulative 64-Patch Overlap Ratio (%)", ylim=(0, 101))
    axis.grid(alpha=.2)
    axis.legend()
    save(fig, output / "comparison/R64_comparison")
    coverages = {}
    for mode in config["modes"]:
        pair = result["pairs"].get(mode)
        if pair and magnitude is not None:
            fig, axis = plt.subplots(figsize=(6, 6))
            axis.imshow(magnitude, cmap="gray", origin="upper")
            a, b = pair["current"], pair["historical"]
            for item, color in [(a, "#ee7744"), (b, "#7766cc")]:
                axis.add_patch(Rectangle((item["left"] - .5, item["top"] - .5), 64, 64, fill=False, ec=color, lw=1.8))
            top, left = max(a["top"], b["top"]), max(a["left"], b["left"])
            bottom, right = min(a["top"], b["top"]) + 64, min(a["left"], b["left"]) + 64
            axis.add_patch(Rectangle((left - .5, top - .5), right-left, bottom-top, color="#aa55bb", alpha=.45))
            axis.set_title(f"{mode}: overlap {100 * pair['intersection_area'] / 4096:.4f}%\n"
                           f"sample0 events {b['sample0_event_id']} and {a['sample0_event_id']}")
            axis.set(xlabel="Padded-image column", ylabel="Padded-image row")
            save(fig, output / mode / "actual_overlap_pair")
        diff = np.zeros((config["padded_size"][0] + 1, config["padded_size"][1] + 1), dtype=np.int64)
        for event in result["events"][mode]:
            if event["patch_size"] == 64:
                y, x = event["top"], event["left"]
                diff[y, x] += 1
                diff[y + 64, x] -= 1
                diff[y, x + 64] -= 1
                diff[y + 64, x + 64] += 1
        coverage = diff.cumsum(0).cumsum(1)[:-1, :-1]
        coverages[mode] = coverage
        np.save(output / mode / "coverage64.npy", coverage)
    vmax = max(1, *(int(c.max()) for c in coverages.values()))
    for mode, coverage in coverages.items():
        fig, axis = plt.subplots(figsize=(6, 5))
        heat = axis.imshow(coverage, origin="upper", cmap="magma", vmin=0, vmax=vmax)
        fig.colorbar(heat, ax=axis, label="64x64 patch coverage count per pixel")
        axis.set_title(mode)
        save(fig, output / mode / "coverage64")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--modes", nargs="+", choices=["uniform_e0", "agga"], default=["uniform_e0", "agga"])
    p.add_argument("--patch_sizes", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--patch_probs", nargs="+", type=float, default=[.2, .3, .5])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--batch_gpu", type=int, default=1)
    p.add_argument("--pad_width", type=int, default=64)
    p.add_argument("--kimg_checkpoints", nargs="+", type=float, default=[500, 1000, 5000, 10000])
    p.add_argument("--overlap_threshold", type=float, default=.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--snapshot_interval", type=float, default=100)
    p.add_argument("--output_dir", type=Path, default=Path("results/oc_overlap_diagnostic"))
    p.add_argument("--configuration_source", default="Controlled simulation using user-supplied E0 resumed-training configuration; start counter=0, no resume state")
    p.add_argument("--torch_threads", type=int, default=2)
    args = p.parse_args(argv)
    if len(args.patch_sizes) != len(args.patch_probs) or len(set(args.patch_sizes)) != len(args.patch_sizes):
        p.error("patch_sizes and patch_probs must have equal length and unique scales")
    if not args.patch_sizes or any(s not in BATCH_MUL for s in args.patch_sizes):
        p.error("unsupported patch scale")
    if len(set(args.modes)) != len(args.modes):
        p.error("duplicate modes")
    if args.batch_size <= 0 or args.batch_gpu <= 0 or args.batch_size % args.batch_gpu:
        p.error("positive batch_size must be divisible by positive batch_gpu (single GPU)")
    if args.pad_width < 0 or args.torch_threads < 1 or not 0 < args.overlap_threshold <= 1:
        p.error("invalid padding, threads or overlap threshold")
    if not np.isfinite(args.snapshot_interval) or args.snapshot_interval <= 0:
        p.error("snapshot_interval must be finite and positive")
    if not all(math.isfinite(x) and x > 0 for x in args.kimg_checkpoints):
        p.error("checkpoints must be finite positive kimg")
    args.kimg_checkpoints = sorted(set(args.kimg_checkpoints))
    args.patch_probs = normalized_probs(args.patch_probs).tolist()
    return args


def main(argv=None):
    args = parse_args(argv)
    # Fail before creating a run or importing heavy dependencies if data absent.
    if not args.data.is_file():
        raise FileNotFoundError(f"Training data unavailable: {args.data}. No experiment results generated.")
    import torch
    torch.set_num_threads(args.torch_threads)
    started = time.perf_counter()
    source_before = git_info()
    if source_before["branch"] != "main-WSY":
        raise RuntimeError("Run from the intended main-WSY checkout")
    image, metadata = load_image(args.data, args.sample_index, args.pad_width)
    if max(args.patch_sizes) > min(metadata["padded_size"]):
        raise ValueError("patch size exceeds actual padded canvas")
    config = {**vars(args), **metadata, "data": str(args.data.resolve()), "output_dir": str(args.output_dir),
              "start_cur_nimg": 0, "world_size": 1, "progressive": False, "train_on_latents": False,
              "batch_mul_dict": {s: BATCH_MUL[s] for s in args.patch_sizes},
              "seed_streams": {"scale": "numpy RandomState(seed)", "dataset": "independent numpy RandomState(seed)",
                               "spatial": "default_rng(SeedSequence([seed, 0x4F43])); common uniforms"}}
    cache, cache_report = {}, {}
    for mode in args.modes:
        rho, gamma = (0., 0.) if mode == "uniform_e0" else (.65, 1.)
        cache[mode], cache_report[mode] = build_cache(image, args.patch_sizes, rho, gamma, args.seed)
    # Only requested modes are evaluated; no neural network is constructed.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir / run_id
    output.mkdir(exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "probability_checks.json", cache_report)
    magnitude = image[:2].square().sum(0).sqrt().numpy()
    result = simulate(config, cache, magnitude, output)
    source_after = git_info()
    if source_before["source_sha256"] != source_after["source_sha256"]:
        raise AssertionError("A protected source changed during diagnosis")
    lines = ["# Spatial Overlap Diagnosis", "", "Actual coordinate simulation on supplied MRI data.",
             "Counts refer to the selected sample only unless explicitly labelled total_patch_events.", "",
             "| Mode | Target kimg | Actual kimg | Total sample patches | 64 patches | R64 (%) | Rall (%) | R64,total (%) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in args.modes:
        for r in result["rows"][mode]:
            if r["is_formal_checkpoint"]:
                metrics = ["NaN" if r[k] is None else f"{r[k]:.4f}" for k in METRICS]
                lines.append(f"| {mode} | {r['target_kimg']:g} | {r['actual_kimg']:.3f} | {r['sample0_total_patches']} | {r['sample0_n64']} | " + " | ".join(metrics) + " |")
    lines.extend(["", "Cumulative repeated spatial coverage does not establish denoising disagreement or OC effectiveness.",
                  "This controlled schedule is not the original training RNG trajectory.",
                  "Source: user-provided resumed E0 configuration, replayed continuously from 0 kimg."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "execution.json", {"status": "completed", "elapsed_seconds": time.perf_counter()-started,
        "command": sys.argv, "source_before": source_before, "source_after": source_after,
        "schedule_sha256": result["schedule_sha256"], "python": sys.version, "numpy": np.__version__, "torch": torch.__version__})
    print(f"Completed: {output.resolve()}")
    print("\n".join(lines))
    return output


if __name__ == "__main__":
    main()
