"""Evaluate WDR candidates against a shared, configuration-checked baseline cache."""
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_SAMPLES = '0,4,8,13,18,22,26,31'
METRICS = ('psnr', 'ssim', 'nrmse', 'reconstruction_seconds', 'gpu_peak_gib')


def integer_env(key, default, minimum=1):
    value = int(os.environ.get(key, default))
    if value < minimum:
        raise ValueError(f'{key} must be >= {minimum}')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checkpoint(key):
    path = Path(os.environ[key]).expanduser().resolve(strict=True)
    match = re.fullmatch(r'network-snapshot-(\d+)\.pkl', path.name)
    if not path.is_file() or not match:
        raise ValueError(f'{key}: expected an existing EMA snapshot, got {path}')
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Checkpoint changed while reading: {path}')
    return path, int(match[1]), h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True)
        tmp = f.name
    os.replace(tmp, path)


def sample_row(group, sample, config):
    """Accept only complete results with matching recorded inputs and schedule."""
    folder = group / f'sample_{sample}'
    evaluate = folder / 'evaluate'
    try:
        requested = read_json(folder / 'requested_config.json')
        expected = dict(config, sample=sample)
        # GPU is provenance, not part of the reconstruction cache key.
        if any(requested.get(k) != v for k, v in expected.items()):
            return None
        levels = read_json(evaluate / 'dps_sigma_schedule.json')
        if (len(levels) != config['steps'] or not all(math.isfinite(t) for t in levels)
                or abs(levels[0] - 1.0) >= 1e-6 or abs(levels[-1] - .003) >= 1e-6
                or any(a <= b for a, b in zip(levels, levels[1:]))):
            return None
        m = read_json(evaluate / 'comp_plots/results.json')['per_image'][str(sample)]
        timing = read_json(evaluate / 'timing.json')
        seconds = timing.get('reconstruction_seconds')
        if seconds is None:
            seconds = timing['recon_seconds_per_sample'][str(sample)]
        row = {k: float(m[k]) for k in METRICS[:3]}
        row.update(reconstruction_seconds=float(seconds),
                   gpu_peak_gib=float((evaluate / 'peak_gib.txt').read_text()))
        if not all(math.isfinite(v) for v in row.values()):
            return None
        if row['reconstruction_seconds'] <= 0 or row['gpu_peak_gib'] <= 0:
            return None
        return row
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError):
        return None


def evaluate(root, project, name, model, common, samples, gpu):
    path, kimg, sha = model
    config = dict(common, model_path=str(path), checkpoint_sha256=sha)
    key = digest(config)
    group = root / name / f'{path.stem}_{key[:20]}'
    group.mkdir(parents=True, exist_ok=True)
    # Serialise writers so multiple variants can safely share a baseline.
    with (group / '.cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = group / 'cache_config.json'
        if manifest.exists() and read_json(manifest) != config:
            raise RuntimeError(f'Cache key/config mismatch: {group}')
        atomic_json(manifest, config)
        missing = []
        for i in samples:
            if sample_row(group, i, config) is not None:
                print(f'Cache hit: {name} R{common["mask"]} sample {i}', flush=True)
                continue
            missing.append(i)
            old = group / f'sample_{i}'
            if old.exists():
                archive = Path(tempfile.mkdtemp(prefix=f'retry_sample_{i}.', dir=group))
                shutil.move(str(old), str(archive / 'previous'))
        print(f'{name}: {len(samples)-len(missing)} cached, {len(missing)} to run | {group}', flush=True)
        # Core skips validated samples and regenerates its summaries for this exact selection.
        # A cache-only request still runs the lightweight CSV aggregation, not DPS.
        env = dict(os.environ, MODEL_PATH=str(path), METHOD=name, GPU=gpu,
                   WDR_PROFILE='1', WDR_CHECKPOINT_SHA256=sha,
                   MODEL_LABEL=f'{name}_{kimg}k', CKPT_KIMG=str(kimg),
                   MASKS=str(common['mask']), OUT_ROOT=str(group),
                   PROJECT_DIR=str(project), VAL_DIR=common['val_dir'],
                   SAMPLES=','.join(map(str, samples)), DPS_STEPS=str(common['steps']),
                   DPS_INNER=str(common['inner_loops']),
                   PADIS_PATCH_BATCH=str(common['patch_batch']), PADIS_PATCH_CHECKPOINT='0')
        subprocess.run(['bash', str(project / 'bash/eval_paired_sigma1_core.sh')],
                       env=env, cwd=project, check=True)
        rows = {i: sample_row(group, i, config) for i in samples}
        if any(row is None for row in rows.values()):
            raise RuntimeError(f'Incomplete or mismatched results: {group}')
        # Preserve selection-specific summaries even if a later request uses different samples.
        selection = group / 'selections' / digest(samples)[:16]
        selection.mkdir(parents=True, exist_ok=True)
        for filename in ('per_image.csv', 'summary.json', 'summary.csv', 'summary.md'):
            shutil.copy2(group / filename, selection / filename)
        atomic_json(selection / 'selection.json', {'samples': samples, 'config': config,
                    'sample_gpus': {str(i): read_json(group / f'sample_{i}/requested_config.json').get('gpu') for i in samples}})
        return rows, group, key


def main():
    project = Path(os.environ.get('PROJECT_DIR', Path(__file__).resolve().parents[1])).resolve()
    base = Path(os.environ.get('RESULT_ROOT', '/mnt/SSD2/wsy/PaDIS-MRI'))
    out = Path(os.environ.get('OUT_ROOT', base / 'PaDIS-MRI-recon/new_data/WDR_paired')).resolve()
    variant = os.environ.get('VARIANT', 'v0')
    mode = os.environ.get('MODE', 'e0')
    if variant not in ('baseline', 'v0', 'v1', 'v2', 'v3', 'v4') or mode not in ('e0', 'opa'):
        raise ValueError('Use VARIANT=baseline/v0/v1/v2/v3/v4 and MODE=e0/opa')
    masks = [int(v) for v in os.environ.get('MASKS', '7 4').split()]
    if not masks or len(set(masks)) != len(masks) or any(v not in (4, 7) for v in masks):
        raise ValueError('MASKS must be 7, 4, or "7 4" (no duplicates)')
    samples = [int(v.strip()) for v in os.environ.get('SAMPLES', DEFAULT_SAMPLES).split(',')]
    if not samples or any(v < 0 for v in samples) or len(set(samples)) != len(samples):
        raise ValueError('SAMPLES must contain unique non-negative comma-separated integers')
    # Check the checkpoint budgets before expensive evaluation.
    wdr = checkpoint('VARIANT_CKPT')
    baseline = checkpoint('BASELINE_CKPT')
    tolerance = integer_env('KIMG_TOLERANCE', 10, 0)
    if abs(wdr[1] - baseline[1]) > tolerance:
        raise ValueError(f'Checkpoint mismatch: candidate={wdr[1]}k, baseline={baseline[1]}k; '
                         f'allowed difference={tolerance}k (KIMG_TOLERANCE)')
    patch_batch = integer_env('PADIS_PATCH_BATCH', 49)
    if patch_batch > 49:
        raise ValueError('PADIS_PATCH_BATCH must be 1..49')
    common = dict(val_dir=str(Path(os.environ.get('VAL_DIR', '/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB')).resolve()),
                  steps=integer_env('DPS_STEPS', 40, 2), inner_loops=integer_env('DPS_INNER', 6),
                  sigma_max=1.0, sigma_min=.003, rho=7.0, seed=123, zeta=3.0,
                  image_size=384, pad=64, psize=64, patch_batch=patch_batch,
                  patch_checkpoint=0, sampler='pure_dps', profile=True)
    gpu = os.environ.get('GPU', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    print(f'{variant}: {wdr[0]} ({wdr[1]}k); baseline: {baseline[0]} ({baseline[1]}k)', flush=True)
    for mask in masks:
        config = dict(common, mask=mask)
        root = out / mode / f'R{mask}'
        reference, ref_dir, ref_key = evaluate(root, project, 'baseline', baseline, config, samples, gpu)
        candidate, cand_dir, cand_key = evaluate(root, project, variant + '_candidate', wdr, config, samples, gpu)
        # Each pairing/selection gets its own comparison file; do not overwrite another experiment.
        paired = root / 'comparisons' / variant / f'{wdr[1]}k_vs_{baseline[1]}k_{digest([cand_key, ref_key, samples])[:20]}'
        paired.mkdir(parents=True, exist_ok=True)
        rows = [[candidate[i][k] - reference[i][k] for k in METRICS] for i in samples]
        with tempfile.NamedTemporaryFile(mode='w', newline='', dir=paired, delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['sample', *METRICS])
            writer.writerows([i, *row] for i, row in zip(samples, rows))
            writer.writerow(['mean', *(statistics.mean(col) for col in zip(*rows))])
            tmp = f.name
        os.replace(tmp, paired / 'paired.csv')
        atomic_json(paired / 'comparison_config.json', {'mode': mode, 'variant': variant,
                    'candidate': str(cand_dir), 'baseline': str(ref_dir), 'samples': samples,
                    'candidate_kimg': wdr[1], 'baseline_kimg': baseline[1], 'config': config,
                    'delta': 'candidate minus baseline'})
        print(f'Paired differences: {paired / "paired.csv"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        sys.exit(str(exc))
