"""Run sample26 with parameters read from an existing formal LGFC config.json."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing formal LGFC config.json")
    parser.add_argument("--save_dir", required=True, help="Separate Figure reconstruction output")
    parser.add_argument("--figure_outer_step", type=int, default=None)
    parser.add_argument("--print_command", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fixed = {
        "val_dir": "/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB",
        "steps": 78, "inner_loops": 10, "image_size": 384,
        "pad": 64, "psize": 64, "seed": 123,
    }
    for key, value in fixed.items():
        if config[key] != value:
            raise ValueError(f"Formal config {key}={config[key]!r}, expected {value!r}")
    if not config["enable_lgfc"]:
        raise ValueError("The supplied config is not an enabled formal LGFC run")
    if "zeta" in config and config["zeta"] != 3.0:
        raise ValueError("The supplied formal config differs from the requested zeta=3.0")
    output_dir = Path(args.save_dir).resolve()
    if output_dir == Path(config["save_dir"]).resolve():
        raise ValueError("Use a separate --save_dir to preserve formal experiment outputs")

    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-u", str(root / "eval_lgfc/run_lgfc.py")]
    parameters = (
        "model_path", "val_dir", "mask_select", "steps", "inner_loops",
        "image_size", "pad", "psize", "seed", "reference_mode", "ema_beta",
        "global_start_outer", "global_every", "unsampled_weight",
        "max_relative_update", "dc_growth_limit", "backtrack_factor", "max_backtracks",
    )
    # Every experimental value is explicit. Missing formal parameters raise KeyError.
    for key in parameters:
        command.extend([f"--{key}", str(config[key])])
    command.extend([
        "--sample_indices", "26", "--val_count", "1", "--zeta", "3.0",
        "--save_dir", str(output_dir), "--enable_lgfc", "--run_evaluate",
        "--save_lgfc_figure_assets",
    ])
    if config["gpus"] is not None:
        command.append("--gpus")
        command.extend(str(gpu) for gpu in config["gpus"])
    for flag in ("save_lgfc_diagnostics", "profile_memory"):
        if config[flag]:
            command.append(f"--{flag}")
    if args.figure_outer_step is not None:
        command.extend(["--figure_outer_step", str(args.figure_outer_step)])
    print(f"Formal config: {config_path}", flush=True)
    print(shlex.join(command), flush=True)
    if not args.print_command:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
