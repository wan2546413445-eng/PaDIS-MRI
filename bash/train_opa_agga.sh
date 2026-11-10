#!/usr/bin/env bash
set -euo pipefail

# Environment variables are converted to explicit entry-point arguments.
CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-dryrun}
if (( $# )); then shift; fi
case "$MODE" in dryrun|smoke|train) ;; *) echo 'Mode must be dryrun, smoke or train' >&2; exit 2 ;; esac
DATA_DIR=${DATA_DIR:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB}
OUTDIR=${OUTDIR:-$CODE_ROOT/PaDIS-MRI-runs/opa_agga_training-runs}
args=(--mode "$MODE" --data "$DATA_DIR" --outdir "$OUTDIR"
      --gpu "${GPU:-0}" --opa-weight "${OPA_WEIGHT:-1.0}"
      --rho "${RHO:-0.65}" --gamma "${GAMMA:-1.0}"
      --score-mode "${SCORE_MODE:-multiply}" --seed "${SEED:-123}"
      --workers "${WORKERS:-4}")
if [[ -n ${TOTAL_KIMG:-} ]]; then args+=(--total-kimg "$TOTAL_KIMG"); fi
if [[ -n ${RESUME:-} ]]; then args+=(--resume "$RESUME"); fi
if [[ -n ${TRANSFER:-} ]]; then args+=(--transfer "$TRANSFER"); fi
cd "$CODE_ROOT"
if [[ $MODE == dryrun ]]; then
    exec "${PYTHON:-python}" train/padis-mri/train_opa_agga.py "${args[@]}" "$@"
fi
# Disable external tracking by default; existing local training logs are retained.
export WANDB_MODE=${WANDB_MODE:-disabled}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONPATH="$CODE_ROOT/train/padis-mri:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node=1 \
    train/padis-mri/train_opa_agga.py "${args[@]}" "$@"
