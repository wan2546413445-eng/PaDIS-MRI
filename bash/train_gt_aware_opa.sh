#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-train}
case "$mode" in
    debug) kimg=${KIMG:-1} ;;
    train) kimg=${KIMG:-500} ;;
    *) echo 'Expected debug or train' >&2; exit 2 ;;
esac
[[ "$kimg" =~ ^[1-9][0-9]*$ ]]
lambda=${LAMBDA_OVERLAP:-1.0}
duration=$(python -c 'import sys; print(int(sys.argv[1])/1000)' "$kimg")
root=${OUTDIR:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/OPA_GT_AWARE}
out="$root/C128_GT-aware_lam${lambda}_${kimg}kimg_${mode}"
export PYTHONPATH="$PWD:$PWD/train/padis-mri:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PADIS_GT_SMOKE=0
if [[ "$mode" == debug ]]; then export PADIS_GT_SMOKE=1; fi
torchrun --standalone --nproc_per_node=1 train/padis-mri/train_overlap.py \
    --outdir="$out" --data="${DATA_DIR:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB}" \
    --cond=0 --arch=ddpmpp --precond=pedm --cbase=128 --cres=2,2,2 \
    --overlap-mode=independent --overlap-geometry=identical --overlap-alignment=symmetric \
    --overlap-spatial-weight=uniform --overlap-consistency-weight=gt_aware --lambda-overlap="$lambda" \
    --batch=4 --batch-gpu=1 --lr=1e-4 --dropout=0.05 --augment=0 --real_p=0.5 \
    --padding=1 --pad_width=64 --patch-list=16,32,64 --patch-probs=0.2,0.3,0.5 \
    --duration="$duration" --ema=0.5 --tick=5 --snap=200 --dump=400 --workers=4 --seed=123
