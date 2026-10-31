#!/bin/bash
set -e
set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# 用法：
# GPU=4 VARIANT=control     /bin/bash bash/train_overlap_independent.sh debug
# GPU=5 VARIANT=center      /bin/bash bash/train_overlap_independent.sh debug
# GPU=4 VARIANT=control     /bin/bash bash/train_overlap_independent.sh main
# GPU=5 VARIANT=center      /bin/bash bash/train_overlap_independent.sh main

GPU=${GPU:-0}
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

BATCH_SIZE=4
BATCH_GPU=1
LR=1e-4
DROPOUT=0.05
REAL_P=0.5
PADDING=1
PAD_WIDTH=64
WORKERS=4
SEED=123
RESUME_ARGS=()
RESUME_PATH=${RESUME_PATH:-}
OC_NOISE_MODE=${OC_NOISE_MODE:-independent}
OC_PAIR_GEOMETRY=${OC_PAIR_GEOMETRY:-shifted}
OC_SPATIAL_WEIGHT_MODE=${OC_SPATIAL_WEIGHT_MODE:-distance}
OC_ALIGNMENT_MODE=${OC_ALIGNMENT_MODE:-directional}

case "$OC_ALIGNMENT_MODE" in
    directional|symmetric) ;;
    *)
        echo "Invalid OC_ALIGNMENT_MODE=$OC_ALIGNMENT_MODE"
        exit 1
        ;;
esac

case "$OC_NOISE_MODE" in
    independent|same_center)
        ;;
    *)
        echo "Invalid OC_NOISE_MODE=$OC_NOISE_MODE"
        exit 1
        ;;
esac
case "$VARIANT" in
    control)
        LAMBDA_OVERLAP=0.0
        ;;
    center)
        LAMBDA_OVERLAP=${LAMBDA_OVERLAP:-0.5}
        ;;
    *)
        echo "Unknown VARIANT=$VARIANT，必须使用 control 或 center"
        exit 1
        ;;
esac

case "$MODE" in
    debug)
        # 强制抽取 64 patch，确保 debug 真正测试到最坏显存分支。
        DURATION=0.001
        TICK=1
        SNAP=1
        DUMP=1
        PATCH_SIZES=64
        PATCH_PROBS=1.0
        PATCH_TAG=p64only
        ;;
    overnight)
        DURATION=5
        TICK=5
        SNAP=200
        DUMP=1000
        PATCH_SIZES=16,32,64
        PATCH_PROBS=0.2,0.3,0.5
        PATCH_TAG=s16s32s64_p020305
        ;;
    resume15k)

        DURATION=15.12
        TICK=5
        SNAP=200
        DUMP=1000

        PATCH_SIZES=16,32,64
        PATCH_PROBS=0.2,0.3,0.5
        PATCH_TAG=s16s32s64_p020305

        RESUME_PATH=${RESUME_PATH:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB/main_overlap_independent_center_lam0p3_active64_bgpu1_s16s32s64_p020305_b2_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-overlap-independent-center-lam0p3-p64/training-state-005040.pt}

        if [ ! -f "$RESUME_PATH" ]; then
            echo "找不到恢复文件：$RESUME_PATH"
            exit 1
        fi

        RESUME_ARGS=(--resume="$RESUME_PATH")
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
        echo "Unknown MODE=$MODE，必须使用 debug、overnight、resume15k 或 main"
        exit 1
        ;;
esac

LAMBDA_TAG=${LAMBDA_OVERLAP//./p}

EXP_NAME=overlap_${OC_NOISE_MODE}_${VARIANT}_${OC_ALIGNMENT_MODE}_lam${LAMBDA_TAG}_active64_bgpu${BATCH_GPU}_${PATCH_TAG}_b${BATCH_SIZE}_seed${SEED}

if [[ "$OC_PAIR_GEOMETRY" != shifted || "$OC_SPATIAL_WEIGHT_MODE" != distance ]]; then
    EXP_NAME=${EXP_NAME}_geom${OC_PAIR_GEOMETRY}_spatial${OC_SPATIAL_WEIGHT_MODE}
fi

RUN_NAME=${MODE}_${EXP_NAME}

OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_DIR=$ROOT_DATA/$SNR
mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"

{
    echo "=================================================="
    echo "PaDIS-MRI Independent-Noise Center-Guided Overlap Training"
    echo "MODE=$MODE"
    echo "VARIANT=$VARIANT"
    echo "GPU=$GPU"
    echo "LAMBDA_OVERLAP=$LAMBDA_OVERLAP"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "BATCH_GPU=$BATCH_GPU"
    echo "ACTIVE_PATCH_SIZE=64"
    echo "PATCH_SIZES=$PATCH_SIZES"
    echo "PATCH_PROBS=$PATCH_PROBS"
    echo "SAME_SIGMA=true"
    echo "OC_NOISE_MODE=$OC_NOISE_MODE"
    echo "OC_PAIR_GEOMETRY=$OC_PAIR_GEOMETRY"
    echo "OC_SPATIAL_WEIGHT_MODE=$OC_SPATIAL_WEIGHT_MODE"
    echo "OC_ALIGNMENT_MODE=$OC_ALIGNMENT_MODE"
    echo "INDEPENDENT_NOISE=$([ "$OC_NOISE_MODE" = independent ] && echo true || echo false)"
    echo "CENTER_TO_BOUNDARY=$([ "$OC_ALIGNMENT_MODE" = directional ] && echo true || echo false)"
    echo "GT_GATE=false"
    echo "NOISE_RATIO_GATE=false"
    echo "OUTDIR=$OUTDIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "RESUME_PATH=${RESUME_PATH:-none}"
    echo "TARGET_DURATION_MIMG=$DURATION"
    echo "=================================================="
} | tee "$LOG_FILE"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone \
    --nproc_per_node=$NPROC \
    train/padis-mri/train_overlap.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=pedm \
    --overlap-mode="$OC_NOISE_MODE" \
    --overlap-alignment="$OC_ALIGNMENT_MODE" \
    --overlap-geometry="$OC_PAIR_GEOMETRY" \
    --overlap-spatial-weight="$OC_SPATIAL_WEIGHT_MODE" \
    --lambda-overlap="$LAMBDA_OVERLAP" \
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
