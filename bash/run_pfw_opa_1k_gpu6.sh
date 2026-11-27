#!/usr/bin/env bash
set -euo pipefail
cd /mnt/SSD/wsy/projects/PaDIS-MRI-main
base=/mnt/SSD2/wsy/PaDIS-MRI
runs="$base/PaDIS-MRI-runs"
out="$base/PaDIS-MRI-recon/new_data/OPA_PFW_vs_OPA_C128_1k_T1FLAIR_R4_40x6_sigma1"
suffix=p64-align-symmetric-geom-identical-spatial-uniform
prefix=00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center
pfw="$runs/OPA_PFW/C128_OPA1p25_Wmax2.0_train_5000kimg/$prefix-lam1p25-$suffix/network-snapshot-001004.pkl"
opa="$runs/overlap_independent_training-runs/brain/32dB/overnight_overlap_independent_center_symmetric_lam2p5_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform/$prefix-lam2p5-$suffix/network-snapshot-001016.pkl"
for file in "$pfw" "$opa" eval/post_eval_espirit.py eval/utils_espirit.py; do
    [[ -r "$file" ]] || { echo "Missing $file"; exit 1; }
done
for method in OPA_PFW OPA; do
    model="$pfw"; k=1004
    if [[ "$method" == OPA ]]; then model="$opa"; k=1016; fi
    GPU=6 METHOD="$method" CKPT_KIMG="$k" MODEL_PATH="$model" OUT_ROOT="$out/$method" \
        SAMPLES=0,4,8,13,18,22,26,31 MASKS=4 DPS_STEPS=40 DPS_INNER=6 \
        bash bash/eval_paired_sigma1_core.sh
done
