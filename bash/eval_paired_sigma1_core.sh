#!/usr/bin/env bash
# U-Net width ablation: exact pure-DPS implementation from original OPA 40x6 sigma1 control.
# Run with GPU=2 MODEL_PATH=/path/to/network-snapshot-001016.pkl.
# CBASE and CKPT_KIMG are labels only; weights are controlled by MODEL_PATH.
set -euo pipefail
cd "${PROJECT_DIR:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"

: "${GPU:?Set GPU=2 (or another available GPU)}"
: "${MODEL_PATH:?Set MODEL_PATH to the network-snapshot-*.pkl to evaluate}"
CBASE=${CBASE:-128}
CKPT_KIMG=${CKPT_KIMG:-1016}
model_label="${METHOD:-OPA}_C${CBASE}_L1p25_${CKPT_KIMG}k"
model="$MODEL_PATH"
val=${VAL_DIR:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}
steps=${DPS_STEPS:-40}
inner=${DPS_INNER:-6}
root=${OUT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/UNet_width_ablation/T1FLAIR_R4/OPA_C${CBASE}_L1p25_${CKPT_KIMG}k_s${steps}x${inner}_sigma1_pilot8}
samples=${SAMPLES:-0,4,8,13,18,22,26,31}
read -r -a masks <<< "${MASKS:-4}"
[[ ${#masks[@]} -eq 1 ]] || { echo "Use a single mask per output directory (MASKS=4)." >&2; exit 1; }
IFS=, read -r -a indices <<< "$samples"
export CUDA_VISIBLE_DEVICES="$GPU" MPLBACKEND=Agg
export PYTHONPATH="$PWD:$PWD/train/padis-mri:$PWD/eval${PYTHONPATH:+:$PYTHONPATH}"
export PADIS_PATCH_BATCH=49 PADIS_PATCH_CHECKPOINT=0
unset NOISE_DIR NOISE_COVARIANCE NOISE_COMPLEX_VAR
[[ -r "$model" ]] || { printf '找不到权重：%s\n' "$model" >&2; exit 1; }
printf 'GPU %s | %s | pure DPS %s×%s | %s updates | sigma 1 -> 0.003\nCheckpoint: %s\nOutput: %s\n' "$GPU" "$model_label" "$steps" "$inner" "$((steps*inner))" "$model" "$root"

for mask in "${masks[@]}"; do
    group="$root"
    mkdir -p "$group"
    for i in "${indices[@]}"; do
        out="$group/sample_$i"
        if [[ -s "$out/evaluate/comp_plots/results.json" && -s "$out/evaluate/timing.json" && -s "$out/evaluate/dps_sigma_schedule.json" ]]; then
            printf 'Skip R%s sample %s\n' "$mask" "$i"
            continue
        fi
        if [[ -d "$out/evaluate/recons" ]]; then
            retry=$(mktemp -d "$group/retry_sample_${i}.XXXXXX")
            mv -- "$out" "$retry/previous"
        fi
        mkdir -p "$out"
        printf '\nR%s | sample %s | DPS %s×%s\n' "$mask" "$i" "$steps" "$inner"
        python -u - \
            --mode dps --steps "$steps" --inner_loops "$inner" --seed 123 --zeta 3.0 \
            --image_size 384 --pad 64 --psize 64 --mask_select "$mask" \
            --model_path "$model" --val_dir "$val" --sample_indices "$i" --save_dir "$out" \
            2>&1 <<'PY_SIGMA' | tee "$out/terminal.log"
import json
import os
from pathlib import Path
import runpy
import sys
import recon

original = recon._ve_base_schedule
actual = []
def sigma1(net, num_sigmas, sigma_min, sigma_max, rho, device):
    levels = original(net, num_sigmas, 0.003, 1.0, 7.0, device)
    actual.append(levels[:-1].detach().cpu().tolist())
    return levels

recon._ve_base_schedule = sigma1
def arg(name):
    return sys.argv[sys.argv.index(name)+1]
requested = dict(model_path=arg('--model_path'), val_dir=arg('--val_dir'),
                 mask=int(arg('--mask_select')), sample=int(arg('--sample_indices')),
                 steps=int(arg('--steps')), inner_loops=int(arg('--inner_loops')),
                 sigma_max=1.0, sigma_min=.003, rho=7.0, seed=123,
                 gpu=os.environ['CUDA_VISIBLE_DEVICES'])
(Path(arg('--save_dir')) / 'requested_config.json').write_text(json.dumps(requested, indent=2))
runpy.run_path('eval_fast/run.py', run_name='__main__')
expected = int(sys.argv[sys.argv.index('--steps') + 1])
assert len(actual) == 1 and len(actual[0]) == expected
assert abs(actual[0][0] - 1.0) < 1e-6 and abs(actual[0][-1] - .003) < 1e-6
out = Path(sys.argv[sys.argv.index('--save_dir') + 1]) / 'evaluate'
(out / 'dps_sigma_schedule.json').write_text(json.dumps(actual[0], indent=2))
print(f'DPS schedule verified: {expected} levels, 1 -> 0.003', flush=True)
PY_SIGMA
    done

    python - "$group" "$mask" "$steps" "$samples" "$model" "$GPU" "$inner" "$model_label" <<'PY'
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

group = Path(sys.argv[1])
mask, steps = map(int, sys.argv[2:4])
indices = [int(i) for i in sys.argv[4].split(',')]
model, gpu = sys.argv[5:7]
inner = int(sys.argv[7])
model_label = sys.argv[8]
keys = ('psnr', 'ssim', 'nrmse', 'reconstruction_seconds')
rows = []
for i in indices:
    evaluate = group / f'sample_{i}/evaluate'
    m = json.loads((evaluate / 'comp_plots/results.json').read_text())['per_image'][str(i)]
    timing = json.loads((evaluate / 'timing.json').read_text())
    t = timing.get('reconstruction_seconds')
    if t is None:
        t = timing['recon_seconds_per_sample'][str(i)]
    rows.append([i, *(float(m[k]) for k in keys[:3]), float(t)])
with (group / 'per_image.csv').open('w', newline='') as f:
    csv.writer(f).writerows([['sample', *keys], *rows])
summary = dict(dataset='T1FLAIR', mask=mask, model=model_label, checkpoint=model,
               sampler='pure_dps', steps=steps, inner_loops=inner, total_updates=steps*inner,
               sigma_max=1.0, sigma_min=.003, rho=7.0, seed=123, samples=len(indices), gpu=gpu)
lines = [f'{model_label} / T1-FLAIR R{mask} / pure DPS {steps}×{inner} / σmax=1 / N={len(indices)}',
         '', '| PSNR (dB) | SSIM | NRMSE | 秒/张 |', '|---:|---:|---:|---:|']
cells = []
for col, (key, digits) in enumerate(zip(keys, (4, 5, 5, 2)), 1):
    values = [r[col] for r in rows]
    summary[key + '_mean'] = mean(values)
    summary[key + '_std'] = pstdev(values)
    cells.append(f'{mean(values):.{digits}f}' if key=='reconstruction_seconds'
                 else f'{mean(values):.{digits}f} ± {pstdev(values):.{digits}f}')
lines.append('| ' + ' | '.join(cells) + ' |')
text = '\n'.join(lines) + '\n'
(group / 'summary.md').write_text(text, encoding='utf-8')
(group / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
with (group / 'summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary))
    writer.writeheader()
    writer.writerow(summary)
print(text, flush=True)
print(f'逐张指标和时间：{group / "per_image.csv"}', flush=True)
PY
done
printf '\n完成。结果目录：%s\n' "$root"
