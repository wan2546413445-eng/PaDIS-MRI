#!/usr/bin/env bash
set -euo pipefail
cd /mnt/SSD/wsy/projects/PaDIS-MRI-main
export CUDA_VISIBLE_DEVICES=${GPU:-4}
export PADIS_PATCH_BATCH=${PADIS_PATCH_BATCH:-7}
export PADIS_PATCH_CHECKPOINT=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/eval:$PWD:$PWD/train/padis-mri${PYTHONPATH:+:$PYTHONPATH}"

RUNS=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB
TRAIN=resume10k_overlap_independent_center_symmetric_lam2p5_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform
JOB=00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam2p5-p64-align-symmetric-geom-identical-spatial-uniform
MODEL=${MODEL:-$RUNS/$TRAIN/$JOB/network-snapshot-010000.pkl}
VAL=${VAL:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}
MASK=${MASK:-4}
SAMPLE=${SAMPLE:-26}
BASE=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/OPA10k_fixed47_partition_zoom
PLAN_ONLY=${PLAN_ONLY:-0}

if [[ "$PLAN_ONLY" == 1 ]]; then
    OUT="$BASE/T1FLAIR_R${MASK}_sample${SAMPLE}_seed123_PLAN"
else
    test -s "$MODEL" || { echo "找不到权重：$MODEL"; exit 1; }
    test -s "$VAL/sample_${SAMPLE}.pt" || { echo "找不到样本：$VAL/sample_${SAMPLE}.pt"; exit 1; }
    mkdir -p "$BASE"
    OUT=$(mktemp -d "$BASE/T1FLAIR_R${MASK}_sample${SAMPLE}_seed123_XXXXXX")
fi
printf '本次总输出目录：%s\n' "$OUT"

COMMON=(
    --model_path "$MODEL" --val_dir "$VAL"
    --sample_indices "$SAMPLE" --mask_select "$MASK"
    --image_size 384 --pad 64 --psize 64
    --score_mode patch --grid_mode cycle_k
    --seed 123 --grid_seed 123
    --pula_convention padis_real --pula_variant baseline
    --num_noise_levels 47 --langevin_steps 5
    --sigma_max 1.0 --sigma_min 0.003
    --gamma 0.5 --cg_iters 10 --cg_rtol 1e-6
    --dps_base_steps 47 --dps_inner_loops 5
    --dps_sigma_max 1.0 --dps_sigma_min 0.003 --dps_rho 7 --dps_zeta 3
    --fixed47_patch_policy shared_cycle_k
)
[[ -z "${NOISE_DIR:-}" ]] || COMMON+=(--noise_dir "$NOISE_DIR")
[[ -z "${NOISE_COMPLEX_VAR:-}" ]] || COMMON+=(--noise_complex_var "$NOISE_COMPLEX_VAR")
[[ -z "${NOISE_COVARIANCE:-}" ]] || COMMON+=(--noise_covariance "$NOISE_COVARIANCE")

run_case() {
    local switch=$1 name=$2 case_dir="$OUT/$2"
    echo "========== $name =========="
    if [[ "$PLAN_ONLY" == 1 ]]; then
        python -u tools/run_fixed47_partition_zoom.py "${COMMON[@]}" \
            --fixed47_switch "$switch" --out_root "$case_dir" --dry_run
    else
        mkdir -p "$case_dir"
        python -u tools/run_fixed47_partition_zoom.py "${COMMON[@]}" \
            --fixed47_switch "$switch" --out_root "$case_dir" \
            2>&1 | tee "$case_dir/terminal.log"
    fi
}

run_case 47 pure_pULA47x5
run_case 0 pure_DPS47x5
run_case 23 pULA23x5_DPS24x5

if [[ "$PLAN_ONLY" != 1 ]]; then
    echo "三组第12层："
    find "$OUT" -name 'layer_012_*.png' | sort
    echo "全部输出：$OUT"
fi
