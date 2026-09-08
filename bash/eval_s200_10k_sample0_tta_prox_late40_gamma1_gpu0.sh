#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# PaDIS-MRI differentiable proximal Scan-TTA
# S200 long-train 10k / sample0 / GPU0
#
# Keep the best timing found so far:
#   diffusion indices [40,30,20,10]
#
# Change ONLY the TTA objective:
#
# Old:
#   stop-gradient CG target
#   loss = ||D_theta - stopgrad(z_CG)||^2
#
# New:
#   z_cond = argmin_z 0.5||z-D_theta||^2
#                      +0.5*gamma||Az-y||^2
#   finite CG is differentiable
#   loss = mean |A z_cond - y|^2
#
# gamma = 1.0
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
CG_GAMMA=1.0
TTA_LR=1e-5

SAVE_DIR="${SAVE_DIR:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/scan_tta/s200_10k_sample0_paired_gpu0/scan_tta_prox_late40_gamma1}"

mkdir -p "${SAVE_DIR}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "找不到 checkpoint: ${MODEL_PATH}" >&2
    exit 2
fi

if [[ ! -f "${VAL_DIR}/sample_${SAMPLE_INDEX}.pt" ]]; then
    echo "找不到 validation sample: ${VAL_DIR}/sample_${SAMPLE_INDEX}.pt" >&2
    exit 2
fi

echo "============================================================"
echo "PaDIS-MRI differentiable proximal Scan-TTA"
echo
echo "GPU               : ${GPU}"
echo "Sample            : ${SAMPLE_INDEX}"
echo "Mask / seed       : ${MASK_SELECT} / ${SEED}"
echo "Steps x inner     : ${STEPS} x ${INNER_LOOPS}"
echo
echo "Objective         : differentiable_prox_measurement"
echo "TTA events        : [40, 30, 20, 10]"
echo "Refinement/event  : ${REFINEMENT_ITERS}"
echo "CG/refinement     : ${CG_ITERS}"
echo "CG gamma          : ${CG_GAMMA}"
echo "TTA lr            : ${TTA_LR}"
echo
echo "Reference baseline: 33.53 / 0.8705 / 0.0740"
echo "Old late40 target : 33.69 / 0.8804 / 0.0727"
echo "Output            : ${SAVE_DIR}"
echo "============================================================"

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
    --cg-gamma "${CG_GAMMA}" \
    --tta-lr "${TTA_LR}"

echo
echo "============================================================"
echo "Differentiable proximal Scan-TTA finished"
echo "Result: ${SAVE_DIR}"
echo "Compare:"
echo "  baseline          : 33.53 / 0.8705 / 0.0740"
echo "  old projected TTA : 33.69 / 0.8804 / 0.0727"
echo "============================================================"
