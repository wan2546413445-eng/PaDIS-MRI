#!/bin/bash
set -e
set -o pipefail

# A0 inherits the validated OPA C96 settings while selecting the independent
# efficient backbone and an isolated result directory.
GPU=${GPU:-0}
KIMG=${KIMG:-1016}
BATCH=${BATCH:-4}
BATCH_GPU=${BATCH_GPU:-1}
SEED=${SEED:-123}
OPA_LAMBDA=${OPA_LAMBDA:-1.25}
DROPOUT=${DROPOUT:-0.05}
WORKERS=${WORKERS:-4}
CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

DURATION_MIMG=$(awk -v kimg="$KIMG" 'BEGIN { printf "%.12g", kimg / 1000 }')
LAMBDA_TAG=${OPA_LAMBDA//./p}
OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/OPA_A0/OPA_A0_C96_L${LAMBDA_TAG}_${KIMG}kimg
DATA_DIR=$ROOT_DATA/32dB

mkdir -p "$OUTDIR"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone \
    --nproc_per_node=1 \
    train/padis-mri/train_overlap.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --backbone=efficient_songunet \
    --precond=pedm \
    --cbase=96 \
    --cres=2,2,2 \
    --overlap-mode=independent \
    --overlap-alignment=symmetric \
    --overlap-geometry=identical \
    --overlap-spatial-weight=uniform \
    --lambda-overlap="$OPA_LAMBDA" \
    --batch="$BATCH" \
    --batch-gpu="$BATCH_GPU" \
    --lr=1e-4 \
    --ema=0.5 \
    --dropout="$DROPOUT" \
    --augment=0 \
    --real_p=0.5 \
    --padding=1 \
    --pad_width=64 \
    --patch-list=16,32,64 \
    --patch-probs=0.2,0.3,0.5 \
    --duration="$DURATION_MIMG" \
    --tick=5 \
    --snap=400 \
    --dump=400 \
    --workers="$WORKERS" \
    --seed="$SEED"

