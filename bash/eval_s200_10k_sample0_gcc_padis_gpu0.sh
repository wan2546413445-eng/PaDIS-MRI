#!/usr/bin/env bash
set -Eeuo pipefail

# GCC-PaDIS formal sample0 evaluation. This script is intentionally not run
# during implementation review.

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
SAVE_DIR="${SAVE_DIR:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/gcc_padis/s200_10k_sample0}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"

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
echo "Global Cross-Conditioned PaDIS"
echo "GPU / sample / seed : ${GPU} / ${SAMPLE_INDEX} / 123"
echo "Image/pad/patch     : 384 / 64 / 64"
echo "Mask               : 7 (54 columns, ACS24, holdout11)"
echo "Formal updates      : 78 x 10"
echo "TTA events          : [40, 30, 20, 10]"
echo "TTA refine/lr       : 5 / 1e-5"
echo "Proximal CG         : gamma=1, iterations=5"
echo "Budget              : 800 denoisers, 20 backprops, 4000 CG iterations"
echo "Output              : ${SAVE_DIR}"
echo "============================================================"

python -u eval_gcc_mri/run_gcc.py \
    --model_path "${MODEL_PATH}" \
    --val_dir "${VAL_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --image_size 384 \
    --pad 64 \
    --psize 64 \
    --mask_select 7 \
    --val_count 1 \
    --sample_indices "${SAMPLE_INDEX}" \
    --seed 123 \
    --steps 78 \
    --inner_loops 10 \
    --gpus "${LOGICAL_GPU}" \
    --fixed_seed_per_sample \
    --refinement-iters 5 \
    --cg-iters 5 \
    --gamma 1 \
    --tta-lr 1e-5

echo "GCC-PaDIS sample${SAMPLE_INDEX} finished: ${SAVE_DIR}"
