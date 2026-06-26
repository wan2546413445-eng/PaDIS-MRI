#!/bin/bash
set -e
set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# 用法：
# GPU=5 /bin/bash bash/train_detail_residual.sh debug
# GPU=5 DETAIL_HIDDEN=48 DETAIL_ETA=0.15 /bin/bash bash/train_detail_residual.sh train10k

GPU=2
MODE=${1:-debug}
NPROC=1

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/detail_residual_training-runs
LOG_DIR=$RESULT_ROOT/results_record/logs
ANATOMY=brain
SNR=32dB

BATCH_SIZE=${BATCH_SIZE:-2}
BATCH_GPU=${BATCH_GPU:-1}
LR=${LR:-1e-4}
DROPOUT=${DROPOUT:-0.05}
REAL_P=${REAL_P:-0.5}
PADDING=${PADDING:-1}
PAD_WIDTH=${PAD_WIDTH:-96}
WORKERS=${WORKERS:-4}
SEED=${SEED:-123}
RESUME_ARGS=()
RESUME_PATH=${RESUME_PATH:-}

DETAIL_HIDDEN=${DETAIL_HIDDEN:-48}
DETAIL_ETA=${DETAIL_ETA:-0.15}
DETAIL_DILATIONS=${DETAIL_DILATIONS:-1,2,5}
DETAIL_GATE_BIAS=${DETAIL_GATE_BIAS:--1.0}
DETAIL_INIT_SCALE=${DETAIL_INIT_SCALE:-0.001}
DETAIL_DETACH_BASE=${DETAIL_DETACH_BASE:-1}
DETAIL_USE_POS=${DETAIL_USE_POS:-1}

LAMBDA_RESIDUAL=${LAMBDA_RESIDUAL:-0.2}
LAMBDA_GRADIENT=${LAMBDA_GRADIENT:-0.1}
LAMBDA_EDGE=${LAMBDA_EDGE:-0.1}
EDGE_ALPHA=${EDGE_ALPHA:-2.0}
DETAIL_USE_SIGMA_WEIGHT=${DETAIL_USE_SIGMA_WEIGHT:-1}

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
    resume10k|resume15k)
        DURATION=${DURATION:-15.12}
        TICK=5
        SNAP=200
        DUMP=1000
        PATCH_SIZES=16,32,64
        PATCH_PROBS=0.2,0.3,0.5
        PATCH_TAG=s16s32s64_p020305
        if [ -z "$RESUME_PATH" ] || [ ! -f "$RESUME_PATH" ]; then
            echo "RESUME_PATH is required for $MODE and must point to training-state-*.pt"
            exit 1
        fi
        RESUME_ARGS=(--resume="$RESUME_PATH")
        ;;
    *)
        echo "Unknown MODE=$MODE，必须使用 debug、overnight、train10k、main、resume10k 或 resume15k"
        exit 1
        ;;
esac

ETA_TAG=${DETAIL_ETA//./p}
RES_TAG=${LAMBDA_RESIDUAL//./p}
GRAD_TAG=${LAMBDA_GRADIENT//./p}
EDGE_TAG=${LAMBDA_EDGE//./p}
DIL_TAG=${DETAIL_DILATIONS//,/p}
EXP_NAME=detail_h${DETAIL_HIDDEN}_eta${ETA_TAG}_res${RES_TAG}_grad${GRAD_TAG}_edge${EDGE_TAG}_dil${DIL_TAG}_bgpu${BATCH_GPU}_${PATCH_TAG}_b${BATCH_SIZE}_seed${SEED}
RUN_NAME=${MODE}_${EXP_NAME}

OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_DIR=$ROOT_DATA/$SNR
mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"

{
    echo "=================================================="
    echo "PaDIS-MRI Detail Residual Prior Training"
    echo "MODE=$MODE"
    echo "GPU=$GPU"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "BATCH_GPU=$BATCH_GPU"
    echo "DETAIL_HIDDEN=$DETAIL_HIDDEN"
    echo "DETAIL_ETA=$DETAIL_ETA"
    echo "DETAIL_DILATIONS=$DETAIL_DILATIONS"
    echo "DETAIL_GATE_BIAS=$DETAIL_GATE_BIAS"
    echo "DETAIL_INIT_SCALE=$DETAIL_INIT_SCALE"
    echo "DETAIL_DETACH_BASE=$DETAIL_DETACH_BASE"
    echo "LAMBDA_RESIDUAL=$LAMBDA_RESIDUAL"
    echo "LAMBDA_GRADIENT=$LAMBDA_GRADIENT"
    echo "LAMBDA_EDGE=$LAMBDA_EDGE"
    echo "EDGE_ALPHA=$EDGE_ALPHA"
    echo "DETAIL_USE_SIGMA_WEIGHT=$DETAIL_USE_SIGMA_WEIGHT"
    echo "PATCH_SIZES=$PATCH_SIZES"
    echo "PATCH_PROBS=$PATCH_PROBS"
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
    train/padis-mri/train_detail_residual.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=pedm \
    --detail-hidden="$DETAIL_HIDDEN" \
    --detail-eta="$DETAIL_ETA" \
    --detail-dilations="$DETAIL_DILATIONS" \
    --detail-use-pos="$DETAIL_USE_POS" \
    --detail-gate-bias="$DETAIL_GATE_BIAS" \
    --detail-init-scale="$DETAIL_INIT_SCALE" \
    --detail-detach-base="$DETAIL_DETACH_BASE" \
    --lambda-residual="$LAMBDA_RESIDUAL" \
    --lambda-gradient="$LAMBDA_GRADIENT" \
    --lambda-edge="$LAMBDA_EDGE" \
    --edge-alpha="$EDGE_ALPHA" \
    --detail-use-sigma-weight="$DETAIL_USE_SIGMA_WEIGHT" \
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
