#!/usr/bin/env python3
"""Legacy-vs-batched S1 GPU check, or self-contained CPU batch/RNG fixture."""

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "0e5b8cc4dc216895c451dbffc079b6a0fe56d698"


def cpu_fixture():
    import random
    import types
    import numpy as np
    import torch
    from unittest.mock import patch

    sys.path.insert(0, str(ROOT))
    # Only redirect hard-coded CUDA allocations in the unchanged old helper.
    # External MRI/BART/plotting imports are unnecessary for these functions.
    source = ast.parse((ROOT / "eval/denoise_padding.py").read_text())
    nodes = ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    shim = types.ModuleType("denoise_padding")
    shim.__dict__.update(np=np, torch=torch, random=random, os=os)
    exec(ast.unparse(nodes).replace("torch.device('cuda')", "torch.device('cpu')"), shim.__dict__)
    sys.modules["denoise_padding"] = shim
    from eval_pula.recon_pula import PULASystem, reconstruct
    from eval_pula.score_opa import PatchScore
    from eval_pula.score_batch import TrajectoryPatchScore
    from eval_pula.trajectories import sample_batch, partitions
    from eval_pula.run import sigma_schedule

    class Net(torch.nn.Module):
        def forward(self, x, sigma, position, labels):
            return (x.tanh() + .03 * x.mean((-2, -1), keepdim=True)
                    + .02 * position) / (1 + sigma.square())

    class Operator:
        def __init__(self):
            g = torch.Generator().manual_seed(11)
            self.maps = torch.randn((1, 2, 8, 8), dtype=torch.complex64, generator=g)
            self.mask = torch.ones((1, 1, 8, 8))
            self.mask[..., 1::2] = 0

        def forward(self, x):
            z = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)
            return self.mask * torch.fft.fft2(self.maps * z, norm="ortho")

        def adjoint(self, y):
            return (self.maps.conj() * torch.fft.ifft2(self.mask * y, norm="ortho")).sum(1, keepdim=True)

    args = argparse.Namespace(num_noise_levels=3, langevin_steps=2, sigma_max=1., sigma_min=.01,
                              gamma=.5, cg_iters=10, cg_rtol=1e-6, pad=2)
    net, operator = Net().eval(), Operator()
    y = operator.mask * torch.randn((1, 2, 8, 8), dtype=torch.complex64, generator=torch.Generator().manual_seed(12))
    oracle = TrajectoryPatchScore(net, 8, 2, 2, torch.device("cpu"))
    old_oracle = PatchScore(net, 8, 2, 2, torch.device("cpu"))
    for mode in ("fixed", "cycle_k", "random_step"):
        args.grid_mode = mode
        sequences = [partitions(args, 123 + j) for j in range(10)]
        for convention in ("padis_real", "paper_complex"):
            args.pula_convention = convention
            reference_images, reference_records = [], []
            for j in range(10):
                system = PULASystem(operator, y, torch.Generator().manual_seed(123+j), 10, 1e-6, convention)
                image, _, timing, _ = reconstruct(system, old_oracle, sigma_schedule(args), sequences[j][0], .5,
                                                  pula_shifts_by_level=sequences[j], report=lambda _: None)
                reference_images.append(image[0])
                reference_records.append(timing)
            for size in (1, 2, 4, 8, 10):
                images, records = [], []
                for start in range(0, 10, size):
                    stop = min(start+size, 10)
                    image, row, _, _ = sample_batch(operator, y, oracle, args, list(range(123+start, 123+stop)),
                                                   sequences[start:stop], sigma_schedule(args), report=lambda _: None)
                    images.extend(image.unbind(0)); records.extend(row)
                for j in range(10):
                    assert records[j]["completed_updates"] == args.num_noise_levels * args.langevin_steps
                    assert records[j]["cg_solve_count"] == 1 + args.num_noise_levels * args.langevin_steps
                    torch.testing.assert_close(images[j], reference_images[j], atol=1e-6, rtol=1e-5)
                    assert records[j]["initialization_cg"] == reference_records[j]["initialization_cg"], (mode, convention, size, j, "CG init")
                    assert records[j]["update_cg"] == reference_records[j]["update_cg"], (mode, convention, size, j, "CG update")
                expected = torch.stack(reference_images).mean(0)
                actual = torch.stack(images).mean(0)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                print(f"PASS CPU fixture: mode={mode} convention={convention} batch={size}, seeds 123..132", flush=True)
    # Make the micro-batch boundary cross trajectory boundaries.
    with patch.dict(os.environ, {"PADIS_PATCH_BATCH": "7"}):
        image, _, _, _ = sample_batch(operator, y, oracle, args, [123,124], sequences[:2], sigma_schedule(args), report=lambda _: None)
        torch.testing.assert_close(image[0], reference_images[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(image[1], reference_images[1], atol=1e-6, rtol=1e-5)
    print("PASS CPU fixture: cross-trajectory micro-batch offsets/stitching; no GPU/model/data/metrics validation claimed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu_fixture", action="store_true")
    p.add_argument("--model_path")
    p.add_argument("--val_dir")
    p.add_argument("--out_root")
    p.add_argument("--baseline_run", help="Optional saved old run.py if BASE is unavailable in local git")
    p.add_argument("--num_noise_levels", type=int, default=60)
    p.add_argument("--langevin_steps", type=int, default=4)
    args = p.parse_args()
    if args.cpu_fixture:
        cpu_fixture()
        return
    if not all((args.model_path, args.val_dir, args.out_root)):
        p.error("GPU equivalence requires model_path, val_dir and a new out_root.")
    import numpy as np

    out = Path(args.out_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    legacy_source = (Path(args.baseline_run).read_text() if args.baseline_run else
                     subprocess.check_output(["git", "show", f"{BASE}:eval_pula/run.py"], cwd=ROOT, text=True))
    snapshot = out / "legacy_run_source.py"
    snapshot.write_text(legacy_source)
    common = ["--model_path", args.model_path, "--val_dir", args.val_dir, "--sample_indices", "0",
              "--mask_select", "4", "--grid_mode", "cycle_k", "--seed", "123", "--grid_seed", "123",
              "--sigma_max", "10", "--sigma_min", "0.01", "--num_noise_levels", str(args.num_noise_levels),
              "--langevin_steps", str(args.langevin_steps), "--gamma", "0.5", "--cg_iters", "10", "--cg_rtol", "1e-6",
              "--pula_convention", "padis_real"]
    # Compile the saved source with its original __file__, preserving ROOT and
    # all actual project imports; no old checkout is overwritten.
    code = "import sys; s=sys.argv.pop(1); f=sys.argv.pop(1); exec(compile(open(s).read(),f,'exec'),{'__name__':'__main__','__file__':f})"
    commands = [[sys.executable, "-c", code, str(snapshot), str(ROOT / "eval_pula/run.py"), *common, "--out_root", str(out / "legacy")],
                [sys.executable, str(ROOT / "eval_pula/run.py"), *common, "--out_root", str(out / "batched"),
                 "--num_trajectories", "1", "--trajectory_batch_size", "1", "--trajectory_seed_start", "123", "--aggregate_counts", "1"]]
    for name, command in zip(("legacy", "batched"), commands):
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    old = out / "legacy/cycle_k/sample_0/evaluate"
    new = out / "batched/sample_0"
    config = json.loads((old / "config.json").read_text())
    manifest = json.loads((new / "manifest.json").read_text())
    record = manifest["trajectory_records"][0]
    assert config["sigma_init"] == manifest["sigma_init"], "initialization sigma differs"
    assert config["sigma_levels"] == manifest["sigma_levels"], "sigma schedule differs"
    old_partitions = [config["shifts"] for _ in config["sigma_levels"]]
    assert old_partitions == record["partition_shifts"], "partition RNG/sequence differs"
    old_timing = json.loads((old / "timing.json").read_text())
    assert old_timing["cg_solves"] == record["cg_solve_count"], "CG solve count differs"
    assert old_timing["initialization_cg"] == record["initialization_cg"], "CG initialization differs; inspect network/input RNG"
    assert [r["iterations"] for r in old_timing["update_cg"]] == [r["iterations"] for r in record["update_cg"]], "CG iteration sequence differs"
    left = np.load(old / "recons/recon_patch_0.npy")
    right = np.load(new / "trajectory_000_seed123/evaluate/recons/recon_patch_0.npy")
    np.testing.assert_allclose(right, left, rtol=1e-5, atol=1e-6, err_msg="state differs: check patch packing/stitching or batch kernels")
    old_metrics = json.loads((old / "comp_plots/results.json").read_text())["per_image"]["0"]
    new_metrics = json.loads((new / "S01_mean/evaluate/results.json").read_text())["per_image"]["0"]
    for key in ("psnr", "ssim", "nrmse"):
        np.testing.assert_allclose(new_metrics[key], old_metrics[key], atol=1e-5, rtol=1e-6, err_msg=f"normalization/metric differs: {key}")
    result = {"baseline_commit": BASE, "sigma_sequence_equal": True, "partitions_equal": True,
              "cg_solve_count": record["cg_solve_count"], "cg_iterations_equal": True,
              "max_abs_complex_error": float(np.abs(right-left).max()), "metrics": new_metrics,
              "metric_tolerance": {"atol": 1e-5, "rtol": 1e-6}, "status": "PASS"}
    (out / "equivalence.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
