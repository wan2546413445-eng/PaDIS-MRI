#!/bin/bash
set -e
set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

GPU=${GPU:-2}
KIMG=${KIMG:-1016}
BATCH=${BATCH:-4}
BATCH_GPU=${BATCH_GPU:-1}
SEED=${SEED:-123}
OPA_LAMBDA=${OPA_LAMBDA:-1.25}
LR=${LR:-1e-4}
DROPOUT=${DROPOUT:-0.05}
EMA_MIMG=${EMA_MIMG:-0.5}
WORKERS=${WORKERS:-4}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-false}

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

DURATION_MIMG=$(awk -v kimg="$KIMG" 'BEGIN { printf "%.12g", kimg / 1000 }')
LAMBDA_TAG=${OPA_LAMBDA//./p}
OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/OPA_NC_Restormer/OPA_NC_Restormer_L${LAMBDA_TAG}_${KIMG}kimg
DATA_DIR=$ROOT_DATA/32dB
LOG_DIR=$RESULT_ROOT/results_record/logs
LOG_FILE=$LOG_DIR/train_OPA_NC_Restormer_${KIMG}kimg_gpu${GPU}_$(date +"%Y%m%d_%H%M%S").log

mkdir -p "$OUTDIR" "$LOG_DIR"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export WANDB_MODE=disabled

{
    echo "=================================================="
    echo "PaDIS-MRI OPA Noise-Conditioned Restormer"
    echo "GPU=$GPU"
    echo "KIMG=$KIMG BATCH=$BATCH BATCH_GPU=$BATCH_GPU SEED=$SEED"
    echo "OPA_LAMBDA=$OPA_LAMBDA LR=$LR DROPOUT=$DROPOUT EMA_MIMG=$EMA_MIMG"
    echo "PATCH_SIZES=16,32,64 PATCH_PROBS=0.2,0.3,0.5 PAD=64 FP32=true"
    echo "ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING"
    echo "OUTDIR=$OUTDIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "=================================================="
} | tee "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone \
    --nproc_per_node=1 \
    train/padis-mri/train_overlap.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --backbone=nc_restormer \
    --precond=pedm \
    --overlap-mode=independent \
    --overlap-alignment=symmetric \
    --overlap-geometry=identical \
    --overlap-spatial-weight=uniform \
    --lambda-overlap="$OPA_LAMBDA" \
    --batch="$BATCH" \
    --batch-gpu="$BATCH_GPU" \
    --lr="$LR" \
    --ema="$EMA_MIMG" \
    --dropout="$DROPOUT" \
    --activation-checkpointing="$ACTIVATION_CHECKPOINTING" \
    --fp16=false \
    --augment=0 \
    --real_p=0.5 \
    --padding=1 \
    --pad_width=64 \
    --patch-list=16,32,64 \
    --patch-probs=0.2,0.3,0.5 \
    --duration="$DURATION_MIMG" \
    --tick=5 \
    --snap=200 \
    --dump=400 \
    --workers="$WORKERS" \
    --seed="$SEED" \
    2>&1 | tee -a "$LOG_FILE"
