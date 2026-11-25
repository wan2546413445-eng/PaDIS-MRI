#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"
ROOT=/mnt/SSD2/wsy/PaDIS-MRI
RUN=resume10k_overlap_independent_center_symmetric_lam2p5_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform
JOB=00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam2p5-p64-align-symmetric-geom-identical-spatial-uniform
MODEL=${MODEL:-$ROOT/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB/$RUN/$JOB/network-snapshot-010000.pkl}
COV_CACHE=${COV_CACHE:-$ROOT/PaDIS-MRI-recon/new_data/pULA_DPS_ablation/Flex_Quality_OPA10k/calibration/OPA10k_train16_a10.npz}
ARGS=()
[[ -z ${TRAIN_PT:-} ]] || ARGS+=(--train_pt "$TRAIN_PT")
CUDA_VISIBLE_DEVICES="${GPU:-6}" PADIS_PATCH_BATCH="${PADIS_PATCH_BATCH:-40}" PADIS_PATCH_CHECKPOINT=0 \
python -u eval_pula/calibrate_patch_cov.py --model_path "$MODEL" --out "$COV_CACHE" \
 --samples "${CAL_SAMPLES:-16}" --anchors "${CAL_ANCHORS:-10}" --sigma_min .003 --sigma_max .45 "${ARGS[@]}"
