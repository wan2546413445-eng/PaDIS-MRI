#!/bin/bash
set -e
set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# 用法：
# 原始随机 overlap pair 采样：
# GPU=4 VARIANT=center ACTIVE_PATCH_SIZES=64 OVERLAP_SAMPLING_MODE=uniform /bin/bash bash/train_overlap_independent_gsam.sh debug
#
# 高梯度优先的 mixed overlap pair 采样：
# GPU=4 VARIANT=center ACTIVE_PATCH_SIZES=64 OVERLAP_SAMPLING_MODE=mixed OVERLAP_GRADIENT_ALPHA=0.7 /bin/bash bash/train_overlap_independent_gsam.sh train10k

GPU=${GPU:-4}
VARIANT=${VARIANT:-center}
MODE=${1:-debug}
NPROC=1

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/overlap_independent_training-runs
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
SEED=123
RESUME_ARGS=()
RESUME_PATH=${RESUME_PATH:-}

ACTIVE_PATCH_SIZES=${ACTIVE_PATCH_SIZES:-64}
ACTIVE_TAG=${ACTIVE_PATCH_SIZES//,/p}

OVERLAP_SAMPLING_MODE=${OVERLAP_SAMPLING_MODE:-uniform}
OVERLAP_GRADIENT_ALPHA=${OVERLAP_GRADIENT_ALPHA:-0.7}
SAMPLING_ALPHA_TAG=${OVERLAP_GRADIENT_ALPHA//./p}
SAMPLING_TAG=sample${OVERLAP_SAMPLING_MODE}_a${SAMPLING_ALPHA_TAG}

case "$VARIANT" in
    control)
        LAMBDA_OVERLAP=0.0
        ;;
    center)
        LAMBDA_OVERLAP=${LAMBDA_OVERLAP:-0.3}
        ;;
    *)
        echo "Unknown VARIANT=$VARIANT，必须使用 control 或 center"
        exit 1
        ;;
esac

case "$MODE" in
    debug)
        DURATION=0.001
        TICK=1
        SNAP=1
        DUMP=1
        PATCH_SIZES=16,32,64
        PATCH_PROBS=0.2,0.3,0.5
        PATCH_TAG=s16s32s64_p020305
        ;;
    overnight)
        DURATION=8.064
        TICK=5
        SNAP=200
        DUMP=1000
        PATCH_SIZES=16,32,64
        PATCH_PROBS=0.2,0.3,0.5
        PATCH_TAG=s16s32s64_p020305
        ;;
    train10k)
        DURATION=10.08
        TICK=5
        SNAP=200
        DUMP=1000
        PATCH_SIZES=16,32,64
        PATCH_PROBS=0.2,0.3,0.5
        PATCH_TAG=s16s32s64_p020305
        ;;
    main)
        DURATION=200
        TICK=5
        SNAP=200
        DUMP=1000
        PATCH_SIZES=16,32,64
        PATCH_PROBS=0.2,0.3,0.5
        PATCH_TAG=s16s32s64_p020305
        ;;
    *)
        echo "Unknown MODE=$MODE，必须使用 debug、overnight、train10k 或 main"
        exit 1
        ;;
esac

LAMBDA_TAG=${LAMBDA_OVERLAP//./p}
EXP_NAME=overlap_independent_${VARIANT}_lam${LAMBDA_TAG}_active${ACTIVE_TAG}_${SAMPLING_TAG}_bgpu1_${PATCH_TAG}_b2_seed123
RUN_NAME=${MODE}_${EXP_NAME}

OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_DIR=$ROOT_DATA/$SNR
mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"

{
    echo "=================================================="
    echo "PaDIS-MRI Independent-Noise Center-Guided Gradient-Sampling Overlap Training"
    echo "MODE=$MODE"
    echo "VARIANT=$VARIANT"
    echo "GPU=$GPU"
    echo "LAMBDA_OVERLAP=$LAMBDA_OVERLAP"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "BATCH_GPU=$BATCH_GPU"
    echo "ACTIVE_PATCH_SIZES=$ACTIVE_PATCH_SIZES"
    echo "OVERLAP_SAMPLING_MODE=$OVERLAP_SAMPLING_MODE"
    echo "OVERLAP_GRADIENT_ALPHA=$OVERLAP_GRADIENT_ALPHA"
    echo "PATCH_SIZES=$PATCH_SIZES"
    echo "PATCH_PROBS=$PATCH_PROBS"
    echo "SAME_SIGMA=true"
    echo "INDEPENDENT_NOISE=true"
    echo "CENTER_TO_BOUNDARY=true"
    echo "GT_GATE=false"
    echo "NOISE_RATIO_GATE=false"
    echo "OUTDIR=$OUTDIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "RESUME_PATH=${RESUME_PATH:-none}"
    echo "TARGET_DURATION_MIMG=$DURATION"
    echo "=================================================="
} | tee "$LOG_FILE"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone \
    --nproc_per_node=$NPROC \
    train/padis-mri/train_overlap_active_selective_gsam.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=pedm \
    --overlap-mode=independent \
    --lambda-overlap="$LAMBDA_OVERLAP" \
    --active-patch-sizes="$ACTIVE_PATCH_SIZES" \
    --overlap-sampling-mode="$OVERLAP_SAMPLING_MODE" \
    --overlap-gradient-alpha="$OVERLAP_GRADIENT_ALPHA" \
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
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"
