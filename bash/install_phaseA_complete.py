#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install COMPLETE Phase A Python source files by safely merging into local files.

This installer uses the EXISTING full eval_pula/run.py and recon_pula.py
as its base, preserving server-only DPS trace and unrelated changes. After
installation those files are complete runnable Python source, not patches.
No git apply, network access, GPU run or training is performed.
"""
from __future__ import annotations
import argparse
import datetime as dt
import os
from pathlib import Path
import shutil
import tempfile
import sys

MERGES = {'eval_pula/run.py': [{'old': '    parser.add_argument("--diagnose_patch_score", action="store_true")\n'
                              '    parser.add_argument("--diagnose_levels", type=sample_list, help="Zero-based noise level '
                              'indices")\n',
                       'new': '    parser.add_argument("--diagnose_patch_score", action="store_true")\n'
                              '    parser.add_argument("--diagnose_dps_jacobian", action="store_true", help="Read-only DPS '
                              'Jacobian mechanism diagnostics")\n'
                              '    parser.add_argument("--jacobian_levels", type=sample_list, help="Zero-based DPS tail '
                              'outer levels")\n'
                              '    parser.add_argument("--jacobian_inner_steps", type=sample_list, help="Zero-based DPS '
                              'tail inner steps")\n'
                              '    parser.add_argument("--diagnose_levels", type=sample_list, help="Zero-based noise level '
                              'indices")\n'},
                      {'old': '        if args.score_mode != "patch" or args.dps_tail_steps or args.terminal_denoise or '
                              'args.diagnose_patch_score or args.save_pula_handoff:\n'
                              '            parser.error("Trajectory batching supports pure block pULA raw states only.")\n',
                       'new': '        if args.score_mode != "patch" or args.dps_tail_steps or args.terminal_denoise or '
                              'args.diagnose_patch_score or args.save_pula_handoff or args.diagnose_dps_jacobian:\n'
                              '            parser.error("Trajectory batching supports pure block pULA raw states only.")\n'},
                      {'old': '    if args.score_mode == "full" and (args.diagnose_patch_score or args.diagnose_levels or '
                              'args.dps_tail_steps):\n'
                              '        parser.error("Full-image pULA does not use patch grid diagnostics or the patch DPS '
                              'tail.")\n',
                       'new': '    if args.score_mode == "full" and (args.diagnose_patch_score or args.diagnose_levels or '
                              'args.dps_tail_steps or args.diagnose_dps_jacobian):\n'
                              '        parser.error("Full-image pULA does not use patch grid diagnostics or the patch DPS '
                              'tail.")\n'},
                      {'old': '    if args.diagnose_levels and (not args.diagnose_patch_score or max(args.diagnose_levels) '
                              '>= args.num_noise_levels):\n'
                              '        parser.error("Diagnostic levels must be within the schedule and require '
                              '--diagnose_patch_score.")\n'
                              '    for name in ("model_path", "val_dir", "out_root", "noise_dir", "noise_covariance"):\n',
                       'new': '    if args.diagnose_levels and (not args.diagnose_patch_score or max(args.diagnose_levels) '
                              '>= args.num_noise_levels):\n'
                              '        parser.error("Diagnostic levels must be within the schedule and require '
                              '--diagnose_patch_score.")\n'
                              '    if args.diagnose_dps_jacobian:\n'
                              '        if not args.dps_tail_steps or args.score_mode != "patch":\n'
                              '            parser.error("Jacobian diagnostics require a patch DPS tail.")\n'
                              '        args.jacobian_levels = args.jacobian_levels if args.jacobian_levels is not None else '
                              '[0, 11, 23]\n'
                              '        args.jacobian_inner_steps = args.jacobian_inner_steps if args.jacobian_inner_steps '
                              'is not None else [0]\n'
                              '        if max(args.jacobian_levels) >= args.dps_tail_steps or '
                              'max(args.jacobian_inner_steps) >= args.dps_inner_loops:\n'
                              '            parser.error("Jacobian diagnostic index exceeds the actual DPS tail outer/inner '
                              'range.")\n'
                              '    elif args.jacobian_levels is not None or args.jacobian_inner_steps is not None:\n'
                              '        parser.error("Jacobian index flags require --diagnose_dps_jacobian.")\n'
                              '    for name in ("model_path", "val_dir", "out_root", "noise_dir", "noise_covariance"):\n'},
                      {'old': '    request = {key: value for key, value in vars(args).items()\n'
                              '               if key not in ("sample_indices", "out_root", "save_pula_handoff", '
                              '"dry_run")}\n'
                              '    for key in ("num_trajectories", "trajectory_batch_size", "trajectory_seed_start",\n',
                       'new': '    request = {key: value for key, value in vars(args).items()\n'
                              '               if key not in ("sample_indices", "out_root", "save_pula_handoff", '
                              '"dry_run")}\n'
                              '    if not args.diagnose_dps_jacobian:\n'
                              '        for key in ("diagnose_dps_jacobian", "jacobian_levels", "jacobian_inner_steps"):\n'
                              '            request.pop(key)\n'
                              '    for key in ("num_trajectories", "trajectory_batch_size", "trajectory_seed_start",\n'},
                      {'old': '        diagnostic_path = out / "patch_score_diagnostics.json"\n'
                              '        reconstructed = (valid_image(recon_path, args.image_size) and '
                              'isinstance(read_json(timing_path), dict)\n'
                              '                         and (not args.save_pula_handoff or valid_image(handoff_path, '
                              'args.image_size))\n'
                              '                         and (not args.terminal_denoise or valid_image(terminal_path, '
                              'args.image_size))\n'
                              '                         and (not args.diagnose_patch_score or '
                              'isinstance(read_json(diagnostic_path), list)))\n'
                              '        if reconstructed and valid_metrics(metric_path, index):\n',
                       'new': '        diagnostic_path = out / "patch_score_diagnostics.json"\n'
                              '        jacobian_path = out / "jacobian_diagnostics.json"\n'
                              '        reconstructed = (valid_image(recon_path, args.image_size) and '
                              'isinstance(read_json(timing_path), dict)\n'
                              '                         and (not args.save_pula_handoff or valid_image(handoff_path, '
                              'args.image_size))\n'
                              '                         and (not args.terminal_denoise or valid_image(terminal_path, '
                              'args.image_size))\n'
                              '                         and (not args.diagnose_patch_score or '
                              'isinstance(read_json(diagnostic_path), list))\n'
                              '                         and (not args.diagnose_dps_jacobian or '
                              'isinstance(read_json(jacobian_path), dict)))\n'
                              '        if reconstructed and valid_metrics(metric_path, index):\n'},
                      {'old': '                state, terminal, timing, diagnostics = reconstruct(\n'
                              '                    system, oracle, sigmas, shifts, args.gamma, '
                              'diagnose_levels=args.diagnose_levels or (),\n',
                       'new': '                jacobian_records = [] if args.diagnose_dps_jacobian else None\n'
                              '                state, terminal, timing, diagnostics = reconstruct(\n'
                              '                    system, oracle, sigmas, shifts, args.gamma, '
                              'diagnose_levels=args.diagnose_levels or (),\n'},
                      {'old': '                    dps_inner_loops=args.dps_inner_loops, dps_zeta=args.dps_zeta,\n'
                              '                    save_handoff=save_handoff if args.save_pula_handoff else None,\n',
                       'new': '                    dps_inner_loops=args.dps_inner_loops, dps_zeta=args.dps_zeta,\n'
                              '                    jacobian_levels=args.jacobian_levels if args.diagnose_dps_jacobian else '
                              '(),\n'
                              '                    jacobian_inner_steps=args.jacobian_inner_steps if '
                              'args.diagnose_dps_jacobian else (),\n'
                              '                    jacobian_records=jacobian_records,\n'
                              '                    save_handoff=save_handoff if args.save_pula_handoff else None,\n'},
                      {'old': '                if args.diagnose_patch_score:\n'
                              '                    write_json(diagnostic_path, diagnostics)\n'
                              '                write_json(timing_path, timing)\n',
                       'new': '                if args.diagnose_patch_score:\n'
                              '                    write_json(diagnostic_path, diagnostics)\n'
                              '                if args.diagnose_dps_jacobian:\n'
                              '                    from eval_pula.jacobian_diagnostics import write_outputs\n'
                              '                    from tools.analyze_jacobian_diagnostics import summarize\n'
                              '                    metadata = {\n'
                              '                        "model_path": args.model_path, "val_dir": args.val_dir,\n'
                              '                        "sample_index": index, "mask_select": args.mask_select,\n'
                              '                        "pula_convention": args.pula_convention, "pula_variant": '
                              'args.pula_variant,\n'
                              '                        "grid_mode": args.grid_mode, "seed": args.seed, "grid_seed": '
                              'args.grid_seed,\n'
                              '                        "pad": args.pad, "image_size": args.image_size, "psize": '
                              'args.psize,\n'
                              '                        "pula_levels": pula_levels, "pula_updates": '
                              'timing["langevin_updates"],\n'
                              '                        "dps_tail_sigmas": dps_sigmas,\n'
                              '                        "dps_inner_loops": args.dps_inner_loops,\n'
                              '                        "dps_shifts": [list(s) for s in dps_shifts],\n'
                              '                        "dps_updates": timing["dps_updates"],\n'
                              '                        "network_forwards": timing["network_forwards"],\n'
                              '                        "network_backprops": timing["network_backprops"],\n'
                              '                        "extra_network_forwards": 0, "extra_network_backwards": 0,\n'
                              '                        "extra_raw_mri_adjoint_calls": len(jacobian_records),\n'
                              '                        "mri_operator": "raw_unwhitened_dps_operator",\n'
                              '                    }\n'
                              '                    write_outputs(out, jacobian_records, metadata)\n'
                              '                    summarize(jacobian_path, out / "jacobian_cosines.png")\n'
                              '                write_json(timing_path, timing)\n'}],
 'eval_pula/recon_pula.py': [{'old': '                dps_zeta=3.0, save_handoff=None, pula_variant="baseline",\n'
                                     '                pa_block_size=8, pa_epsilon=1e-12):\n'
                                     '    # sigmas[0] 是高斯初始化 sigma_max；sigmas[1:] 才是各个 Langevin 更新层。\n',
                              'new': '                dps_zeta=3.0, save_handoff=None, pula_variant="baseline",\n'
                                     '                pa_block_size=8, pa_epsilon=1e-12,\n'
                                     '                jacobian_levels=(), jacobian_inner_steps=(), jacobian_records=None):\n'
                                     '    # sigmas[0] 是高斯初始化 sigma_max；sigmas[1:] 才是各个 Langevin 更新层。\n'},
                             {'old': '        if dps_sigmas:\n'
                                     '            if dps_operator is None or dps_measurement is None:\n',
                              'new': '        if dps_sigmas:\n'
                                     '            if jacobian_records is not None:\n'
                                     '                from eval_pula.jacobian_diagnostics import observe as '
                                     'observe_jacobian\n'
                                     '                if not jacobian_levels or not jacobian_inner_steps:\n'
                                     '                    raise ValueError("Jacobian diagnostics need explicit outer and '
                                     'inner indices.")\n'
                                     '                if max(jacobian_levels) >= len(dps_sigmas) or '
                                     'max(jacobian_inner_steps) >= dps_inner_loops:\n'
                                     '                    raise ValueError("Jacobian diagnostic indices outside the DPS '
                                     'tail.")\n'
                                     '            if dps_operator is None or dps_measurement is None:\n'},
                             {'old': '                            likelihood_grad = torch.autograd.grad(sse.sum(), '
                                     'current)[0]\n'
                                     '                        # score 用于 PaDIS 原始扩散更新：(D_patch(x_noisy)-x_noisy)/sigma^2。\n',
                              'new': '                            likelihood_grad = torch.autograd.grad(sse.sum(), '
                                     'current)[0]\n'
                                     '                        if (jacobian_records is not None and outer in '
                                     'jacobian_levels\n'
                                     '                                and inner in jacobian_inner_steps):\n'
                                     '                            jacobian_records.append(observe_jacobian(\n'
                                     '                                current, likelihood_grad, residual, dps_operator,\n'
                                     '                                sigma=sigma, outer=outer, inner=inner, shift=shift,\n'
                                     '                                pad=oracle.pad, image_size=oracle.image_size, '
                                     'psize=oracle.psize,\n'
                                     '                                network_forwards=network_calls,\n'
                                     '                                network_backwards=outer * dps_inner_loops + inner + '
                                     '1,\n'
                                     '                            ))\n'
                                     '                        # score 用于 PaDIS '
                                     '原始扩散更新：(D_patch(x_noisy)-x_noisy)/sigma^2。\n'},
                             {'old': '    timing["pula_seconds"] = timing["reconstruction_seconds"] - dps_tail_seconds\n',
                              'new': '    if jacobian_records is not None:\n'
                                     '        timing["jacobian_extra_ah_calls"] = len(jacobian_records)\n'
                                     '    timing["pula_seconds"] = timing["reconstruction_seconds"] - dps_tail_seconds\n'}]}
SUPPORT_FILES = (
    'eval_pula/jacobian_diagnostics.py',
    'tools/analyze_jacobian_diagnostics.py',
    'tools/compare_jacobian_phaseA_recons.py',
    'tools/test_jacobian_phaseA.py',
    'bash/run_jacobian_phaseA_sample0.sh',
)

def merge_source(path, source):
    modified = source
    edits = 0
    existing = 0
    for n, patch in enumerate(MERGES[path], 1):
        old, new = patch['old'], patch['new']
        old_count = modified.count(old)
        new_count = modified.count(new)
        if old_count == 1:
            modified = modified.replace(old, new, 1)
            edits += 1
        elif old_count == 0 and new_count == 1:
            existing += 1
        else:
            raise RuntimeError(
                f'{path}: insertion {n} conflicts with local source '
                f'(old occurrences={old_count}, new occurrences={new_count}). '
                'Do NOT overwrite the file. Please share this error with ChatGPT.'
            )
    compile(modified, path, 'exec')
    return modified, edits, existing

def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(prefix='.phaseA_', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.chmod(tmpname, path.stat().st_mode if path.exists() else 0o644)
        os.replace(tmpname, path)
    finally:
        if os.path.exists(tmpname):
            os.unlink(tmpname)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default='.', help='Your existing PaDIS-MRI repository root')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Validate all edits; change nothing')
    mode.add_argument('--install', action='store_true', help='Back up and write complete Python sources')
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    pkg = Path(__file__).resolve().parent
    planned = {}
    try:
        for rel in MERGES:
            target = repo / rel
            if not target.is_file():
                raise FileNotFoundError(f'Missing server source: {target}')
            original = target.read_text(encoding='utf-8')
            merged, edits, existing = merge_source(rel, original)
            planned[rel] = (target, original.encode('utf-8'), merged.encode('utf-8'))
            print(f'[OK] {rel}: {edits} new insertions, {existing} already present; source syntax valid')
        for rel in SUPPORT_FILES:
            source = pkg / rel
            target = repo / rel
            if not source.is_file():
                raise FileNotFoundError(f'Package missing {source}')
            content = source.read_bytes()
            if rel.endswith('.py'):
                compile(content, rel, 'exec')
            if target.exists() and target.read_bytes() != content:
                raise RuntimeError(f'Existing {target} differs from R2 package; refusing to overwrite.')
            planned[rel] = (target, target.read_bytes() if target.exists() else None, content)
            print(f'[OK] {rel}: ' + ('already matches' if target.exists() else 'new complete file'))
    except Exception as exc:
        print(f'[STOP] {exc}', file=sys.stderr)
        return 2
    if a.check:
        print('\n[CHECK PASSED] No files were changed. Run --install to create complete files.')
        return 0

    name = 'PaDIS_PhaseA_backup_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = repo.parent / name
    backup.mkdir(parents=True,exist_ok=False)
    changes = [(rel,t,prev,content) for rel,(t,prev,content) in planned.items() if prev != content]
    for rel,t,prev,_ in changes:
        if prev is not None:
            dst=backup/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(t,dst)
    print(f'[BACKUP] {backup}')
    for rel,t,_,content in changes:
        atomic_write(t,content)
        print(f'[WRITTEN] {rel}')
    print('\n[INSTALL PASSED] Complete Python files installed. GPU was not started.')
    print('Next: python eval_pula/run.py --help | grep jacobian')
    print('Then: python -m pytest -q tools/test_jacobian_phaseA.py')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
