#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}/train/padis-mri:${REPO_ROOT}/eval:${REPO_ROOT}/eval_gcc_mri:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PADIS_PATCH_BATCH || true
unset PADIS_PATCH_CHECKPOINT || true

MODEL_PATH="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/agg_overlap_simplified_longtrain_10k/brain/32dB/oc0p5_rho0p65_gamma1p0_oc0p5_pad64_seed123_resume5k_to10k/00001-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-agg-overlap-rho0p65_gamma1p0_oc0p5-p64/network-snapshot-010000.pkl"
# Correct the accidental underscore above if a local override is not used.
MODEL_PATH="${MODEL_PATH_OVERRIDE:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/agg_overlap_simplified_longtrain_10k/brain/32dB/oc0p5_rho0p65_gamma1p0_oc0p5_pad64_seed123_resume5k_to10k/00001-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-agg-overlap-rho0p65-gamma1p0-oc0p5-p64/network-snapshot-010000.pkl}"
VAL_DIR="/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB"
SAVE_DIR="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/gcc_padis/s200_10k_sample0_v3_dualpath"

echo "============================================================"
echo "GCC-PaDIS v3: baseline DPS + proximal global score + holdout TTA"
echo "sample/seed          : 0 / 123"
echo "DPS zeta             : 3"
echo "Global proximal CG   : gamma=1, CG=5, every inner"
echo "TTA                  : [40,30,20,10], 5/event, lr=1e-5"
echo "Output               : ${SAVE_DIR}"
echo "============================================================"

python -u eval_gcc_mri/run_gcc.py \
    --model_path "${MODEL_PATH}" \
    --val_dir "${VAL_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --sample_indices 0 \
    --seed 123 \
    --gpus 0 \
    --fixed_seed_per_sample
