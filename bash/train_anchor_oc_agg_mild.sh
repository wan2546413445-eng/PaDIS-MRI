#!/usr/bin/env bash
set -euo pipefail

# Formal/smoke training for:
#   Original AGGA-mild + primary-only EDM + fixed New Anchor-OC
#
# This script DOES NOT use git.
#
# Expected files:
#   train/padis-mri/training/patch_anchor_overlap_consistency_agg_mild_loss.py
#   train/padis-mri/train_anchor_oc_agg_mild.py
#
# Examples:
#   MODE=smoke GPU=0 LAMBDA_OC=0.3 bash bash/train_anchor_oc_agg_mild.sh
#   MODE=train GPU=0 LAMBDA_OC=0.3 bash bash/train_anchor_oc_agg_mild.sh
#   MODE=train GPU=1 LAMBDA_OC=0.5 bash bash/train_anchor_oc_agg_mild.sh

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODE=${MODE:-smoke}
GPU=${GPU:-0}
SEED=${SEED:-123}

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

ANATOMY=${ANATOMY:-brain}
SNR=${SNR:-32dB}

PATCH_SIZES=${PATCH_SIZES:-16,32,64}
PATCH_PROBS=${PATCH_PROBS:-0.2,0.3,0.5}
AGG_SAMPLE_PROB=0.65
AGG_THR_RATIO=0.05
AGG_BASE=0.08
AGG_RADIUS_SCALE=1.15
AGG_TAU_SCALE=0.30
AGG_MIN_PIXELS=64
AGG_GRAD_ALPHA=1.0
AGG_GRAD_CLIP_Q=0.95
AGG_GATE_BASE=0.25
LAMBDA_OC=${LAMBDA_OC:-0.3}

BATCH_SIZE=${BATCH_SIZE:-4}
BATCH_GPU=${BATCH_GPU:-1}
LR=${LR:-1e-4}
DROPOUT=${DROPOUT:-0.05}
REAL_P=${REAL_P:-0.5}
PAD_WIDTH=${PAD_WIDTH:-64}
WORKERS=${WORKERS:-4}

SMOKE_MIMG=${SMOKE_MIMG:-0.01}  # 10 kimg
TRAIN_MIMG=${TRAIN_MIMG:-10}     # 10000 kimg

if [[ "$MODE" == "smoke" ]]; then
    DURATION_MIMG=$SMOKE_MIMG
    TICK=${TICK:-1}
    SNAP=${SNAP:-5}
    DUMP=${DUMP:-10}
elif [[ "$MODE" == "train" ]]; then
    DURATION_MIMG=$TRAIN_MIMG
    TICK=${TICK:-100}
    SNAP=${SNAP:-10}
    DUMP=${DUMP:-10}
else
    echo "Unknown MODE=$MODE; expected smoke or train" >&2
    exit 2
fi

LAMBDA_TAG=${LAMBDA_OC//./p}

RUN_NAME=agg_mild_anchor_oc_oc${LAMBDA_TAG}_pad${PAD_WIDTH}_seed${SEED}_${MODE}
ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/agg_mild_anchor_oc_training-runs
OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_PATH=${DATA_PATH:-$ROOT_DATA/$SNR}

LOG_DIR=$RESULT_ROOT/results_record/logs
mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"

LOSS_FILE=train/padis-mri/training/patch_anchor_overlap_consistency_agg_mild_loss.py
ENTRY_POINT=train/padis-mri/train_anchor_oc_agg_mild.py

if [[ ! -f "$LOSS_FILE" ]]; then
    echo "Missing $LOSS_FILE" >&2
    exit 3
fi
if [[ ! -f "$ENTRY_POINT" ]]; then
    echo "Missing $ENTRY_POINT" >&2
    echo "Run: python install_anchor_oc_training.py" >&2
    exit 3
fi
if [[ ! -e "$DATA_PATH" ]]; then
    echo "Missing data path: $DATA_PATH" >&2
    exit 3
fi

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export WANDB_NAME="$RUN_NAME"

COMMON_ARGS=(
    --outdir="$OUTDIR"
    --data="$DATA_PATH"
    --cond=0
    --arch=ddpmpp
    --precond=pedm
    --agg-sample-prob="$AGG_SAMPLE_PROB"
    --agg-thr-ratio="$AGG_THR_RATIO"
    --agg-base="$AGG_BASE"
    --agg-radius-scale="$AGG_RADIUS_SCALE"
    --agg-tau-scale="$AGG_TAU_SCALE"
    --agg-min-pixels="$AGG_MIN_PIXELS"
    --agg-grad-alpha="$AGG_GRAD_ALPHA"
    --agg-grad-clip-q="$AGG_GRAD_CLIP_Q"
    --agg-gate-base="$AGG_GATE_BASE"
    --lambda-oc="$LAMBDA_OC"
    --batch="$BATCH_SIZE"
    --batch-gpu="$BATCH_GPU"
    --lr="$LR"
    --dropout="$DROPOUT"
    --augment=0
    --real_p="$REAL_P"
    --padding=1
    --pad_width="$PAD_WIDTH"
    --patch-list="$PATCH_SIZES"
    --patch-probs="$PATCH_PROBS"
    --workers="$WORKERS"
    --seed="$SEED"
    --desc="$MODE"
)

EXTRA_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    EXTRA_ARGS+=(--resume="$RESUME")
fi
if [[ -n "${TRANSFER:-}" ]]; then
    EXTRA_ARGS+=(--transfer="$TRANSFER")
fi

{
    echo "============================================================"
    echo "PaDIS-MRI Original AGGA-mild + New Anchor-OC training"
    echo "============================================================"
    echo "MODE          : $MODE"
    echo "GPU           : $GPU"
    echo "SEED          : $SEED"
    echo "DURATION_MIMG : $DURATION_MIMG"
    echo "PATCH_SIZES   : $PATCH_SIZES"
    echo "PATCH_PROBS   : $PATCH_PROBS"
    echo "AGG_SAMPLE_PROB : $AGG_SAMPLE_PROB"
    echo "AGG_THR_RATIO   : $AGG_THR_RATIO"
    echo "AGG_BASE        : $AGG_BASE"
    echo "AGG_RADIUS_SCALE: $AGG_RADIUS_SCALE"
    echo "AGG_TAU_SCALE   : $AGG_TAU_SCALE"
    echo "AGG_MIN_PIXELS  : $AGG_MIN_PIXELS"
    echo "AGG_GRAD_ALPHA  : $AGG_GRAD_ALPHA"
    echo "AGG_GRAD_CLIP_Q : $AGG_GRAD_CLIP_Q"
    echo "AGG_GATE_BASE   : $AGG_GATE_BASE"
    echo "LAMBDA_OC       : $LAMBDA_OC"
    echo "BATCH           : $BATCH_SIZE / GPU=$BATCH_GPU"
    echo "LR            : $LR"
    echo "DROPOUT       : $DROPOUT"
    echo "PAD_WIDTH     : $PAD_WIDTH"
    echo "DATA_PATH     : $DATA_PATH"
    echo "OUTDIR        : $OUTDIR"
    echo "============================================================"
} | tee "$LOG_FILE"

# Static checks before allocating a formal run.
python -m py_compile \
    train/padis-mri/training/patch_anchor_overlap_consistency_agg_mild_loss.py \
    train/padis-mri/train_anchor_oc_agg_mild.py

CUDA_VISIBLE_DEVICES="$GPU" torchrun \
    --standalone \
    --nproc_per_node=1 \
    "$ENTRY_POINT" \
    "${COMMON_ARGS[@]}" \
    --duration="$DURATION_MIMG" \
    --tick="$TICK" \
    --snap="$SNAP" \
    --dump="$DUMP" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"

