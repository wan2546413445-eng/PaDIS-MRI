#!/usr/bin/env bash
# Run from the PaDIS-MRI code root, using the padis_mri environment.
# GPU=0 bash bash/eval_02_OPA4_T1FLAIR_R4.sh
# SAMPLES="26" runs sample 26 only. All lambda names are CODE values.
set -euo pipefail
cd "${PROJECT_DIR:-$PWD}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD:$PWD/train/padis-mri:$PWD/eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATTERN='/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB/overnight_overlap_independent_center_symmetric_lam4p0_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform/*/network-snapshot-005000.pkl'
MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH=$(python - "$MODEL_PATTERN" <<'PY_MODEL'
import glob, sys
paths = sorted(glob.glob(sys.argv[1]))
if len(paths) != 1:
    raise SystemExit(f'Expected one checkpoint, found {len(paths)}: {paths}. Set MODEL_PATH explicitly.')
print(paths[0])
PY_MODEL
    )
fi
VAL_DIR="${VAL_DIR:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}"
MASK_SELECT=4
SAMPLES="${SAMPLES:-$(seq 0 31)}"
RESULT_BASE="${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/validation_20261105/OPA4_T1FLAIR_R4}"

RESULT_ROOT="$RESULT_BASE"
ENTRY=eval/run.py
WEIGHT=none

mkdir -p "$RESULT_ROOT"
python - "$RESULT_ROOT" "$MODEL_PATH" "$VAL_DIR" "$MASK_SELECT" "$ENTRY" "$WEIGHT" <<'PY_CONFIG'
import json, sys
from pathlib import Path
root, model, val, mask, entry, weight = sys.argv[1:]
config = dict(model_path=model, val_dir=val, mask_select=int(mask), entry=entry,
              steps=78, inner_loops=10, image_size=384, pad=64, psize=64,
              zeta=3.0, seed=123,
              rng_policy=('fixed_per_sample' if entry == 'eval/run.py' else 'one_process_per_sample_seed123'),
              lgfc=(None if weight == 'none' else dict(reference_mode='ema', ema_beta=0.8,
                    start_outer=60, every=1, weight=float(weight), cap=0.08,
                    dc_growth_limit=1.02, backtrack_factor=0.5, max_backtracks=3)))
path = Path(root) / 'run_settings.json'
if path.exists() and json.loads(path.read_text()) != config:
    raise SystemExit('Output settings differ. Set RESULT_ROOT to a new directory.')
path.write_text(json.dumps(config, indent=2))
print(json.dumps(config, indent=2))
PY_CONFIG

for idx in $SAMPLES; do
    sample_out="$RESULT_ROOT/sample_$idx"
    if [[ -s "$sample_out/evaluate/comp_plots/results.json" && -s "$sample_out/evaluate/recons/recon_patch_$idx.npy" ]]; then
        echo "[Reuse] sample=$idx"
        continue
    fi
    mkdir -p "$sample_out"
    echo "[Run] sample=$idx GPU=$CUDA_VISIBLE_DEVICES R=$MASK_SELECT"
    python -u eval/run.py \
        --run_evaluate --algo padis --model_path "$MODEL_PATH" --val_dir "$VAL_DIR" \
        --image_size 384 --pad 64 --psize 64 --mask_select "$MASK_SELECT" --val_count 1 \
        --sample_indices "$idx" --seed 123 --fixed_seed_per_sample \
        --zeta 3.0 --steps 78 --inner_loops 10 --report_every 1 \
        --save_dir "$sample_out" --gpus 0 2>&1 | tee "$sample_out/run.log"
done

python - "$RESULT_ROOT" "$SAMPLES" <<'PY_SUMMARY'
import csv, json, math, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
indices = [int(i) for i in sys.argv[2].split()]
keys = ('psnr', 'ssim', 'nrmse')
rows = []
for idx in indices:
    data = json.loads((root / f'sample_{idx}/evaluate/comp_plots/results.json').read_text())
    values = data['per_image'][str(idx)]
    row = {'sample_index': idx, **{key: float(values[key]) for key in keys}}
    if not all(math.isfinite(row[key]) for key in keys) or row['psnr'] == 0:
        raise SystemExit(f'Invalid normalized metrics: sample {idx}')
    rows.append(row)
stats = {key: {'mean': statistics.mean(r[key] for r in rows),
               'std_population': statistics.pstdev(r[key] for r in rows)} for key in keys}
tag = f'sample{indices[0]}' if len(indices) == 1 else f'all{len(indices)}'
payload = dict(settings=json.loads((root / 'run_settings.json').read_text()),
               sample_indices=indices, summary=stats, per_image=rows)
(root / f'{tag}_metrics.json').write_text(json.dumps(payload, indent=2))
with (root / f'{tag}_per_image.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for key in keys:
    print(f"{key.upper()}: {stats[key]['mean']:.6f} +/- {stats[key]['std_population']:.6f} (N={len(rows)})")
print('Saved:', root / f'{tag}_metrics.json')
PY_SUMMARY
