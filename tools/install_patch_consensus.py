#!/usr/bin/env python3
"""Install the isolated multi-partition consensus pilot into PaDIS-MRI.

Usage (run in the local project root):
  python tools/install_patch_consensus.py --check
  python tools/install_patch_consensus.py --apply

Does not change eval_pula/recon_pula.py or the existing VJP/DPS code.
Installs eval_pula/recon_consensus.py and bash/eval_fixed47_consensus_gpu6.sh,
adds one opt-in --consensus_mode flag to eval_pula/run.py.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil


def replace_one(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: anchor occurs {count} times instead of once; unchanged")
    return source.replace(old, new, 1)


def patched_run(src: str) -> str:
    src = replace_one(src,
        '    parser.add_argument("--fixed47_skip_pula", action="store_true",\n',
        '    parser.add_argument("--consensus_mode", choices=("off", "mean", "dual"), default="off",\n'
        '                        help="Opt-in alternative: 47-level multi-partition consensus; no pULA update or DPS")\n'
        '    parser.add_argument("--fixed47_skip_pula", action="store_true",\n',
        "CLI option")
    src = replace_one(src,
        '    args = parser.parse_args()\n',
        '    args = parser.parse_args()\n'
        '    if args.consensus_mode != "off":\n'
        '        if (args.fixed47_switch != 47 or args.score_mode != "patch"\n'
        '                or args.grid_mode != "cycle_k" or args.fixed47_patch_policy != "shared_cycle_k"\n'
        '                or args.langevin_steps != 5 or args.num_noise_levels != 47\n'
        '                or args.pula_variant != "baseline" or args.num_trajectories is not None\n'
        '                or args.fixed47_skip_pula or args.diagnose_patch_score\n'
        '                or args.diagnose_dps_jacobian):\n'
        '            parser.error("consensus pilot requires --fixed47_switch 47, patch/cycle_k, "\n'
        '                         "shared_cycle_k, K=5, N=47, baseline variant, no diagnostics")\n',
        "mode validation")
    src = replace_one(src,
        '    if args.dry_run:\n        print(json.dumps(dry_run_plan(args), indent=2, ensure_ascii=False))\n        return\n',
        '    if args.dry_run:\n'
        '        plan = dry_run_plan(args)\n'
        '        if args.consensus_mode != "off":\n'
        '            plan.update(algorithm="annealed_multi_partition_consensus_v1",\n'
        '                        pula_updates=0, dps_updates=0, logical_updates=47,\n'
        '                        logical_nfe_per_trajectory=235, denoiser_evaluations=235,\n'
        '                        cg_solves_per_trajectory=48, consensus_levels=47,\n'
        '                        consensus_updates=47)\n'
        '        print(json.dumps(plan, indent=2, ensure_ascii=False))\n'
        '        return\n',
        "honest dry-run counts")
    src = replace_one(src,
        '    from eval_pula.recon_pula import PULASystem, reconstruct, whiten_measurement\n',
        '    from eval_pula.recon_pula import PULASystem, reconstruct, whiten_measurement\n'
        '    if args.consensus_mode != "off":\n'
        '        from functools import partial\n'
        '        from eval_pula.recon_consensus import reconstruct_consensus\n'
        '        reconstruct = partial(reconstruct_consensus, mode=args.consensus_mode)\n',
        "reconstruct dispatch")
    src = replace_one(src,
        '    if not args.fixed47_skip_pula:\n',
        '    if args.consensus_mode == "off":\n'
        '        request.pop("consensus_mode", None)  # Preserve legacy resume identity.\n'
        '    if not args.fixed47_skip_pula:\n',
        "remove opt-in-only key for legacy runs")
    src = replace_one(src,
        '    if args.fixed47_switch is not None:\n        request.update(fixed47_global_sigmas=sigmas[1:],\n',
        '    if args.consensus_mode != "off":\n'
        '        request.update(sampler="opa_annealed_multi_partition_consensus_v1",\n'
        '                       main_output="consensus_quadratic_dc_fov",\n'
        '                       consensus_denoiser_evaluations=235, consensus_cg_solves=48)\n'
        '    if args.fixed47_switch is not None:\n        request.update(fixed47_global_sigmas=sigmas[1:],\n',
        "request sampler identity")
    ast.parse(src, filename="eval_pula/run.py")
    return src


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    run_path = root / "eval_pula/run.py"
    if not run_path.is_file() or not (root / "eval_pula/cg.py").is_file():
        parser.error("Run from the PaDIS-MRI repository root containing eval_pula/run.py")
    src = run_path.read_text()
    if 'parser.add_argument("--consensus_mode"' in src:
        raise RuntimeError("Consensus mode appears already installed; do not install twice")
    new_run = patched_run(src)

    installer_path = Path(__file__).resolve()
    # The download is a three-file bundle: put the 3 files together in tools/.
    module_path = installer_path.with_name("recon_consensus.py")
    runner_path = installer_path.with_name("eval_fixed47_consensus_gpu6.sh")
    for file in (module_path, runner_path):
        if not file.is_file():
            raise FileNotFoundError(f"Missing companion file: {file}; upload all 3 files to tools/")
    ast.parse(module_path.read_text(), filename=str(module_path))

    target_solver = root / "eval_pula/recon_consensus.py"
    target_runner = root / "bash/eval_fixed47_consensus_gpu6.sh"
    if target_solver.exists() or target_runner.exists():
        raise RuntimeError("Target solver or runner already exists; refusing to overwrite")
    print("[CHECK] VJP/FD/DPS files left untouched")
    print("[CHECK] eval_pula/run.py patch parses successfully")
    print(f"[CHECK] new solver: {target_solver}")
    print(f"[CHECK] new runner: {target_runner}")
    if args.check:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = run_path.with_name(f"run.py.before_consensus_{stamp}.bak")
    shutil.copy2(run_path, backup)
    # Validate first, then add new files and finally update run.py.
    target_solver.parent.mkdir(parents=True, exist_ok=True)
    target_runner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(module_path, target_solver)
    shutil.copy2(runner_path, target_runner)
    target_runner.chmod(0o755)
    run_path.write_text(new_run)
    print(f"[OK] run.py backup: {backup}")
    print("[OK] consensus solver installed; previous experiment outputs unchanged")


if __name__ == "__main__":
    main()
