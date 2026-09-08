#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# PaDIS-MRI Scan-TTA late-only diagnostic
# S200 long-train 10k checkpoint
# sample 0 / GPU 0
#
# Keep only TTA events:
#   diffusion index 40, 30, 20, 10
#
# Baseline is NOT rerun here.
# Existing paired baseline:
#   PSNR 33.53 / SSIM 0.8705 / NRMSE 0.0740
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}/train/padis-mri:${REPO_ROOT}/eval:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

GPU=0
LOGICAL_GPU=0
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset PADIS_PATCH_BATCH || true
unset PADIS_PATCH_CHECKPOINT || true

MODEL_PATH="${MODEL_PATH:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/agg_overlap_simplified_longtrain_10k/brain/32dB/oc0p5_rho0p65_gamma1p0_oc0p5_pad64_seed123_resume5k_to10k/00001-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-agg-overlap-rho0p65-gamma1p0-oc0p5-p64/network-snapshot-010000.pkl}"

VAL_DIR="${VAL_DIR:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"

IMAGE_SIZE=384
PAD=64
PSIZE=64
MASK_SELECT=7
ZETA=3.0
STEPS=78
INNER_LOOPS=10
SEED=123

TTA_INTERVAL=10
TTA_MAX_DIFFUSION=40
REFINEMENT_ITERS=5
CG_ITERS=5
TTA_LR=1e-5

SAVE_DIR="${SAVE_DIR:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/scan_tta/s200_10k_sample0_paired_gpu0/scan_tta_late40}"
mkdir -p "${SAVE_DIR}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "找不到 checkpoint: ${MODEL_PATH}" >&2
    exit 2
fi

if [[ ! -f "${VAL_DIR}/sample_${SAMPLE_INDEX}.pt" ]]; then
    echo "找不到 validation sample: ${VAL_DIR}/sample_${SAMPLE_INDEX}.pt" >&2
    exit 2
fi

cat <<EOF
============================================================
PaDIS-MRI Scan-TTA late-only diagnostic

GPU               : ${GPU}
Checkpoint        : ${MODEL_PATH}
Sample            : ${SAMPLE_INDEX}
Mask / seed       : ${MASK_SELECT} / ${SEED}
Steps x inner     : ${STEPS} x ${INNER_LOOPS}

TTA interval      : ${TTA_INTERVAL}
TTA max diffusion : ${TTA_MAX_DIFFUSION}
Expected events   : [40, 30, 20, 10]
Refinement/event  : ${REFINEMENT_ITERS}
CG/refinement     : ${CG_ITERS}
Expected TTA BP   : 20
Expected CG iters : 100
TTA lr            : ${TTA_LR}

Output            : ${SAVE_DIR}
============================================================
EOF

python -u eval_tta_mri/run_tta.py \
    --model_path "${MODEL_PATH}" \
    --val_dir "${VAL_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --image_size "${IMAGE_SIZE}" \
    --pad "${PAD}" \
    --psize "${PSIZE}" \
    --mask_select "${MASK_SELECT}" \
    --val_count 1 \
    --sample_indices "${SAMPLE_INDEX}" \
    --seed "${SEED}" \
    --zeta "${ZETA}" \
    --steps "${STEPS}" \
    --inner_loops "${INNER_LOOPS}" \
    --gpus "${LOGICAL_GPU}" \
    --fixed_seed_per_sample \
    --tta-interval "${TTA_INTERVAL}" \
    --tta-max-diffusion "${TTA_MAX_DIFFUSION}" \
    --refinement-iters "${REFINEMENT_ITERS}" \
    --cg-iters "${CG_ITERS}" \
    --tta-lr "${TTA_LR}"

echo
echo "============================================================"
echo "Late-only Scan-TTA finished"
echo "Result: ${SAVE_DIR}"
echo "Compare against baseline: 33.53 / 0.8705 / 0.0740"
echo "============================================================"
