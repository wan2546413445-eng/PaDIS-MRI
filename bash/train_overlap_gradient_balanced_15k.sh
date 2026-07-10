#!/bin/bash
set -e
set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Balanced Sobel-Gradient Overlap
# - fixed lambda_overlap=0.3
# - pixel/structure mixture=0.5
# - total target=15000 kimg
# - network snapshot approximately every 1008 kimg
# - training state approximately every 2016 kimg

GPU=${GPU:-5}
NPROC=1

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/overlap_structure_training-runs
LOG_DIR=$RESULT_ROOT/results_record/logs
ANATOMY=brain
SNR=32dB

BATCH_SIZE=2
BATCH_GPU=1
LR=1e-4
DROPOUT=0.05
REAL_P=0.5
PADDING=1
PAD_WIDTH=96
WORKERS=4
SEED=${SEED:-123}

PATCH_SIZES=16,32,64
PATCH_PROBS=0.2,0.3,0.5
PATCH_TAG=s16s32s64_p020305

LAMBDA_OVERLAP=${LAMBDA_OVERLAP:-0.3}
STRUCTURE_MODE=gradient
STRUCTURE_MIX=${STRUCTURE_MIX:-0.5}
STRUCTURE_SCALE_MIN=${STRUCTURE_SCALE_MIN:-0.1}
STRUCTURE_SCALE_MAX=${STRUCTURE_SCALE_MAX:-10.0}

DURATION=15.0
TICK=5
SNAP=200
DUMP=400

safe_tag() {
    echo "$1" | sed 's/-/m/g; s/\./p/g'
}

LAMBDA_TAG=$(safe_tag "$LAMBDA_OVERLAP")
MIX_TAG=$(safe_tag "$STRUCTURE_MIX")
EXP_NAME=overlap_structure_gradient_mix${MIX_TAG}_lam${LAMBDA_TAG}_active64_bgpu1_${PATCH_TAG}_b2_seed${SEED}
RUN_NAME=train15k_${EXP_NAME}

OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_DIR=$ROOT_DATA/$SNR
mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"

{
    echo "=================================================="
    echo "PaDIS-MRI Balanced Sobel-Gradient Overlap"
    echo "GPU=$GPU"
    echo "SEED=$SEED"
    echo "LAMBDA_OVERLAP=$LAMBDA_OVERLAP"
    echo "STRUCTURE_MODE=$STRUCTURE_MODE"
    echo "STRUCTURE_MIX=$STRUCTURE_MIX"
    echo "STRUCTURE_SCALE_MIN=$STRUCTURE_SCALE_MIN"
    echo "STRUCTURE_SCALE_MAX=$STRUCTURE_SCALE_MAX"
    echo "ACTIVE_PATCH_SIZE=64"
    echo "PATCH_SIZES=$PATCH_SIZES"
    echo "PATCH_PROBS=$PATCH_PROBS"
    echo "SAME_SIGMA_WITHIN_PAIR=true"
    echo "INDEPENDENT_NOISE_REALIZATION=true"
    echo "INTERIOR_TEACHER_DETACHED=true"
    echo "TARGET_DURATION_MIMG=$DURATION"
    echo "KIMG_PER_TICK=$TICK"
    echo "SNAPSHOT_EVERY_TICKS=$SNAP"
    echo "STATE_DUMP_EVERY_TICKS=$DUMP"
    echo "EXPECTED_SNAPSHOT_INTERVAL_KIMG≈1008"
    echo "EXPECTED_STATE_INTERVAL_KIMG≈2016"
    echo "OUTDIR=$OUTDIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "=================================================="
} | tee "$LOG_FILE"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone \
    --nproc_per_node=$NPROC \
    train/padis-mri/train_overlap_structure_balanced.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=pedm \
    --overlap-mode=independent \
    --lambda-overlap="$LAMBDA_OVERLAP" \
    --structure-mode="$STRUCTURE_MODE" \
    --structure-mix="$STRUCTURE_MIX" \
    --structure-scale-min="$STRUCTURE_SCALE_MIN" \
    --structure-scale-max="$STRUCTURE_SCALE_MAX" \
    --batch="$BATCH_SIZE" \
    --batch-gpu="$BATCH_GPU" \
    --lr="$LR" \
    --dropout="$DROPOUT" \
    --augment=0 \
    --real_p="$REAL_P" \
    --padding="$PADDING" \
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
