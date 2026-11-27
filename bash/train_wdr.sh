#!/usr/bin/env bash
set -euo pipefail
cd "${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"
mode=${1:-train}
case "$mode" in
    smoke) kimg=1 ;;
    train) kimg=${KIMG:-500} ;;
    *) echo "usage: $0 smoke|train" >&2; exit 2 ;;
esac
data=${DATA_DIR:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB}
out=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}/PaDIS-MRI-runs/WDR/${VARIANT:-wdr}_${mode}_${kimg}k
args=()
case ${VARIANT:-wdr} in
    wdr) args+=(--wdr --stage-depths=3,4,4) ;;
    depth) args+=(--stage-depths=3,4,4) ;;
    opa) ;;
    *) exit 2 ;;
esac
if [[ -n ${RESUME_PATH:-} ]]; then args+=(--resume="$RESUME_PATH"); fi
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}} WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="$PWD/train/padis-mri:$PWD${PYTHONPATH:+:$PYTHONPATH}"
duration=$(awk -v n="$kimg" 'BEGIN { print n / 1000 }')
torchrun --standalone --nproc_per_node=1 train/padis-mri/train_overlap.py \
    --outdir="$out" --data="$data" "${args[@]}" \
    --cond=0 --arch=ddpmpp --backbone=songunet --precond=pedm --cbase=128 --cres=2,2,2 \
    --overlap-mode=independent --overlap-alignment=symmetric --overlap-geometry=identical \
    --overlap-spatial-weight=uniform --lambda-overlap=1.25 --batch=4 --batch-gpu=1 \
    --lr=1e-4 --ema=0.5 --dropout=0.05 --augment=0 --real_p=0.5 --padding=1 --pad_width=64 \
    --patch-list=16,32,64 --patch-probs=0.2,0.3,0.5 --duration="$duration" \
    --tick=10 --snap=50 --dump=50 --workers=4 --seed=123
