#!/usr/bin/env bash
set -euo pipefail

# 单卡、单 seed 训练示例：
# GPU=0 SEED=123 bash bash/train_agg_overlap_simplified.sh

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

GPU=${GPU:-0}
SEED=${SEED:-123}
CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

ANATOMY=brain
SNR=32dB
PATCH_SIZES=16,32,64
PATCH_PROBS=0.2,0.3,0.5

# AGG-overlap 仅保留三个方法参数。
AGG_RHO=${AGG_RHO:-0.65}
AGG_GAMMA=${AGG_GAMMA:-1.0}
LAMBDA_OC=${LAMBDA_OC:-0.3}

BATCH_SIZE=2
BATCH_GPU=1
LR=1e-4
DROPOUT=0.05
REAL_P=0.5
PAD_WIDTH=64
WORKERS=4
DURATION=15
TICK=100
SNAP=10
DUMP=20

ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/agg_overlap_simplified_training-runs
LOG_DIR=$RESULT_ROOT/results_record/logs
RHO_TAG=${AGG_RHO//./p}
GAMMA_TAG=${AGG_GAMMA//./p}
OC_TAG=${LAMBDA_OC//./p}
RUN_NAME=main_agg_overlap_rho${RHO_TAG}_gamma${GAMMA_TAG}_oc${OC_TAG}_pad${PAD_WIDTH}_seed${SEED}
OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_DIR=$ROOT_DATA/$SNR

mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export WANDB_NAME="$RUN_NAME"

{
    echo "PaDIS-MRI simplified AGG + overlap consistency"
    echo "GPU=$GPU SEED=$SEED"
    echo "PATCH_SIZES=$PATCH_SIZES PATCH_PROBS=$PATCH_PROBS"
    echo "AGG_RHO=$AGG_RHO AGG_GAMMA=$AGG_GAMMA LAMBDA_OC=$LAMBDA_OC"
    echo "OUTDIR=$OUTDIR"
    echo "DATA_DIR=$DATA_DIR"
} | tee "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPU" torchrun \
    --standalone \
    --nproc_per_node=1 \
    train/padis-mri/train_overlap_agg.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=pedm \
    --agg-rho="$AGG_RHO" \
    --agg-gamma="$AGG_GAMMA" \
    --lambda-oc="$LAMBDA_OC" \
    --batch="$BATCH_SIZE" \
    --batch-gpu="$BATCH_GPU" \
    --lr="$LR" \
    --dropout="$DROPOUT" \
    --augment=0 \
    --real_p="$REAL_P" \
    --padding=1 \
    --pad_width="$PAD_WIDTH" \
    --patch-list="$PATCH_SIZES" \
    --patch-probs="$PATCH_PROBS" \
    --duration="$DURATION" \
    --tick="$TICK" \
    --snap="$SNAP" \
    --dump="$DUMP" \
    --workers="$WORKERS" \
    --seed="$SEED" \
    2>&1 | tee -a "$LOG_FILE"
