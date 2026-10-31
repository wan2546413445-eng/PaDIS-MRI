#!/usr/bin/env bash
set -Eeuo pipefail

cd /mnt/SSD/wsy/projects/PaDIS-MRI-main
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD:$PWD/train/padis-mri:$PWD/eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB"
MODEL_D0="${TRAIN_ROOT}/overnight_overlap_independent_control_directional_lam0p0_active64_bgpu1_s16s32s64_p020305_b4_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam0p0-p64-align-directional/network-snapshot-005000.pkl"
MODEL_D1="${TRAIN_ROOT}/overnight_overlap_independent_center_directional_lam0p5_active64_bgpu1_s16s32s64_p020305_b4_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam0p5-p64-align-directional/network-snapshot-005000.pkl"
VAL_DIR="/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB"
RESULT_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/D0_D1_all32_pad64_b4_new3T_R4_s78x10_005000"
D2_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/D2_symmetric_all32_pad64_b4_new3T_R4_s78x10_005000"

[[ -f eval/run.py ]] || { echo '找不到 eval/run.py'; exit 1; }
[[ -d "$VAL_DIR" ]] || { echo "找不到验证集：$VAL_DIR"; exit 1; }
for model in "$MODEL_D0" "$MODEL_D1"; do
    [[ -s "$model" ]] || { echo "找不到模型：$model"; exit 1; }
done
for idx in {0..31}; do
    [[ -s "$D2_ROOT/sample_${idx}/evaluate/comp_plots/results.json" ]] || { echo "缺少 D2 sample_${idx} 指标，请检查 D2_ROOT"; exit 1; }
done

for group in D0 D1; do
    if [[ "$group" == D0 ]]; then model="$MODEL_D0"; else model="$MODEL_D1"; fi
    for idx in {0..31}; do
        out="$RESULT_ROOT/$group/sample_${idx}"
        mkdir -p "$out"
        echo "$group sample=$idx GPU=$CUDA_VISIBLE_DEVICES 输出=$out"
        python -u eval/run.py \
            --run_evaluate --algo padis --model_path "$model" --val_dir "$VAL_DIR" \
            --image_size 384 --pad 64 --psize 64 --mask_select 4 --val_count 1 \
            --sample_indices "$idx" --seed 123 --fixed_seed_per_sample \
            --zeta 3.0 --steps 78 --inner_loops 10 --report_every 1 \
            --save_dir "$out" --gpus 0 2>&1 | tee "$out/run.log"
        [[ -s "$out/evaluate/recons/recon_patch_${idx}.npy" ]] || { echo "缺少重建：$group sample=$idx"; exit 1; }
        [[ -s "$out/evaluate/comp_plots/results.json" ]] || { echo "缺少指标：$group sample=$idx"; exit 1; }
        printf '%s\n' "$model" > "$out/model_path.txt"
    done
done

python - "$RESULT_ROOT" "$D2_ROOT" <<'PY'
import json, math, statistics, sys
from pathlib import Path
root, d2 = map(Path, sys.argv[1:])
groups = {'D0': root / 'D0', 'D1': root / 'D1', 'D2': d2}
metrics = ('psnr', 'ssim', 'nrmse')
comparisons = (('D1', 'D0'), ('D2', 'D0'), ('D2', 'D1'))
rows = []
for idx in range(32):
    row = {'sample_index': idx}
    for group, base in groups.items():
        path = base / f'sample_{idx}' / 'evaluate/comp_plots/results.json'
        values = json.loads(path.read_text(encoding='utf-8'))['per_image'][str(idx)]
        row[group] = {k: float(values[k]) for k in metrics}
        if not all(math.isfinite(v) for v in row[group].values()):
            raise ValueError(f'指标包含非有限值：{path}')
    for a, b in comparisons:
        row[f'{a}_minus_{b}'] = {k: row[a][k] - row[b][k] for k in metrics}
    rows.append(row)
summary = {}
print('\n=========== D0 / D1 / D2：32 样本配对评估 ===========')
for metric in metrics:
    entry = {g: {'mean': statistics.mean(r[g][metric] for r in rows),
                 'std_population': statistics.pstdev(r[g][metric] for r in rows)} for g in groups}
    for a, b in comparisons:
        name = f'{a}_minus_{b}'
        diffs = [r[name][metric] for r in rows]
        wins = sum(d < 0 if metric == 'nrmse' else d > 0 for d in diffs)
        entry[name] = {'mean_paired_delta': statistics.mean(diffs), 'improved_count': wins,
                       'equal_count': sum(d == 0 for d in diffs), 'total_count': len(rows)}
    summary[metric] = entry
    print(metric.upper(), json.dumps(entry, ensure_ascii=False))
output = root / 'all32_D0_D1_D2_paired_metrics.json'
output.write_text(json.dumps({'sample_indices': list(range(32)), 'result_roots': {g: str(p) for g, p in groups.items()},
                              'summary': summary, 'per_image': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
print('汇总保存至：', output)
PY