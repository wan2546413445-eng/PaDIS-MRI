#!/usr/bin/env bash
set -Eeuo pipefail
cd "${PROJECT_DIR:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD:$PWD/train/padis-mri:$PWD/eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VAL_T2="${VAL_T2:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t2/32dB}"
TRAIN_ROOT=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB
MODEL_D2="${MODEL_D2:-${TRAIN_ROOT}/overnight_overlap_independent_center_symmetric_lam0p5_active64_bgpu1_s16s32s64_p020305_b4_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam0p5-p64-align-symmetric/network-snapshot-005000.pkl}"
OUT_ROOT="${OUT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/OPA_E0_D2_T1FLAIR32_T2_50_R4_s78x10}"
run_group() {
    local contrast="$1" group="$2" model="$3" val="$4" count="$5"
    local idx out
    for ((idx=0;idx<count;idx++)); do
        out="$OUT_ROOT/$contrast/$group/sample_${idx}"
        mkdir -p "$out"
        echo "Running $contrast $group sample=$idx GPU=$CUDA_VISIBLE_DEVICES"
        python -u eval/run.py \
            --run_evaluate --algo padis --model_path "$model" --val_dir "$val" \
            --image_size 384 --pad 64 --psize 64 --mask_select 4 --val_count 1 \
            --sample_indices "$idx" --seed 123 --fixed_seed_per_sample \
            --zeta 3.0 --steps 78 --inner_loops 10 --report_every 1 \
            --save_dir "$out" --gpus 0 2>&1 | tee "$out/run.log"
    done
}
run_group T2 D2_OPA "$MODEL_D2" "$VAL_T2" 50

python - "$OUT_ROOT/T2/D2_OPA" 50 <<'PY_STATS'
import csv,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); n=int(sys.argv[2])
rows=[]
for i in range(n):
    p=root/f'sample_{i}/evaluate/comp_plots/results.json'
    v=json.loads(p.read_text())['per_image'][str(i)]
    rows.append({'sample_index':i,**{k:float(v[k]) for k in ('psnr','ssim','nrmse')}})
summary={k:{'mean':statistics.mean(r[k] for r in rows),
            'std_population':statistics.pstdev(r[k] for r in rows)} for k in ('psnr','ssim','nrmse')}
for k,v in summary.items():
    print(f"{k.upper()}: {v['mean']:.6f} +/- {v['std_population']:.6f} (N={n})")
(root/'metrics_summary.json').write_text(json.dumps({'summary':summary,'per_image':rows},indent=2))
with (root/'per_image_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
PY_STATS
