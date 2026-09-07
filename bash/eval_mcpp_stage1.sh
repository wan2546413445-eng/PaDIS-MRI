#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the PaDIS-MRI repository root}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the AGG+OC network snapshot}"
VAL_DIR="${VAL_DIR:?Set VAL_DIR to the validation 32dB directory}"
SAVE_DIR="${SAVE_DIR:?Set SAVE_DIR to the MCPP result directory}"
GPU_ID="${GPU_ID:-0}"
MCPP_LR="${MCPP_LR:-1e-4}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0,4,8,12,16,20,24,28}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCPP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PADIS_REPO_ROOT="${REPO_ROOT}"
cd "${REPO_ROOT}"

python "${MCPP_ROOT}/eval_mcpp/run_mcpp.py" \
  --model_path "${MODEL_PATH}" \
  --val_dir "${VAL_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --gpus "${GPU_ID}" \
  --sample_indices "${SAMPLE_INDICES}" \
  --fixed_seed_per_sample \
  --image_size 384 \
  --seed 123 \
  --mask_select 7 \
  --zeta 3 \
  --steps 78 \
  --inner_loops 10 \
  --psize 64 \
  --pad 64 \
  --mcpp-lr "${MCPP_LR}" \
  --save-mcpp-diagnostics
