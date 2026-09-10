#!/usr/bin/env bash
set -euo pipefail

# Modes:
#   MODE=audit  : verify one-patch sampling and write sampling heatmaps.
#   MODE=smoke  : short training run (default 10 kimg; keep <=1000 kimg).
#   MODE=train  : formal 10000-kimg training.
#   MODE=paired : formal 10000-kimg paired control using existing AGG-OC code
#                 with LAMBDA_OC=0.
#
# Examples:
#   MODE=audit GPU=0 bash train_agg_simplified.sh
#   MODE=smoke GPU=0 SMOKE_MIMG=0.01 bash train_agg_simplified.sh
#   MODE=train GPU=0 bash train_agg_simplified.sh
#   MODE=paired GPU=1 bash train_agg_simplified.sh

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODE=${MODE:-smoke}
GPU=${GPU:-0}
SEED=${SEED:-123}
CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

ANATOMY=${ANATOMY:-brain}
SNR=${SNR:-32dB}
PATCH_SIZES=${PATCH_SIZES:-16,32,64}
PATCH_PROBS=${PATCH_PROBS:-0.2,0.3,0.5}
AGG_RHO=${AGG_RHO:-0.65}
AGG_GAMMA=${AGG_GAMMA:-1.0}

BATCH_SIZE=${BATCH_SIZE:-4}
BATCH_GPU=${BATCH_GPU:-1}
LR=${LR:-1e-4}
DROPOUT=${DROPOUT:-0.05}
REAL_P=${REAL_P:-0.5}
PAD_WIDTH=${PAD_WIDTH:-64}
WORKERS=${WORKERS:-4}

SMOKE_MIMG=${SMOKE_MIMG:-0.01}  # 0.01 Mimg = 10 kimg.
TRAIN_MIMG=${TRAIN_MIMG:-10}     # 10 Mimg = 10000 kimg.
PAIRED_MIMG=${PAIRED_MIMG:-10}   # Match the formal single-patch run.
AUDIT_IMAGES=${AUDIT_IMAGES:-8}
AUDIT_DRAWS=${AUDIT_DRAWS:-200}

LOG_DIR=$RESULT_ROOT/results_record/logs
RHO_TAG=${AGG_RHO//./p}
GAMMA_TAG=${AGG_GAMMA//./p}
if [[ "$MODE" == "paired" ]]; then
    ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/agg_overlap_simplified_training-runs
    RUN_NAME=agg_paired_control_rho${RHO_TAG}_gamma${GAMMA_TAG}_oc0_pad${PAD_WIDTH}_seed${SEED}
else
    ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/agg_simplified_single_training-runs
    RUN_NAME=agg_single_rho${RHO_TAG}_gamma${GAMMA_TAG}_pad${PAD_WIDTH}_seed${SEED}_${MODE}
fi
OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_PATH=${DATA_PATH:-$ROOT_DATA/$SNR}
AUDIT_DIR=$RESULT_ROOT/results_record/agg_sampler_audit/$RUN_NAME

mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export WANDB_NAME="$RUN_NAME"

COMMON_ARGS=(
    --outdir="$OUTDIR"
    --data="$DATA_PATH"
    --cond=0
    --arch=ddpmpp
    --precond=pedm
    --agg-rho="$AGG_RHO"
    --agg-gamma="$AGG_GAMMA"
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

if [[ "$MODE" == "audit" ]]; then
    mkdir -p "$AUDIT_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" python \
        train/padis-mri/train_agg_simplified.py \
        "${COMMON_ARGS[@]}" \
        --sampler-audit-dir="$AUDIT_DIR" \
        --sampler-audit-images="$AUDIT_IMAGES" \
        --sampler-audit-draws="$AUDIT_DRAWS"
    echo "Sampler audit saved to: $AUDIT_DIR"
    exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
    DURATION_MIMG=$SMOKE_MIMG
    ENTRY_POINT=train/padis-mri/train_agg_simplified.py
    MODE_ARGS=()
    TICK=${TICK:-1}
    SNAP=${SNAP:-5}
    DUMP=${DUMP:-10}
elif [[ "$MODE" == "train" ]]; then
    DURATION_MIMG=$TRAIN_MIMG
    ENTRY_POINT=train/padis-mri/train_agg_simplified.py
    MODE_ARGS=()
    TICK=${TICK:-100}
    SNAP=${SNAP:-10}
    DUMP=${DUMP:-10}
elif [[ "$MODE" == "paired" ]]; then
    DURATION_MIMG=$PAIRED_MIMG
    ENTRY_POINT=train/padis-mri/train_overlap_agg_v2.py
    MODE_ARGS=(--lambda-oc=0)
    TICK=${TICK:-100}
    SNAP=${SNAP:-10}
    DUMP=${DUMP:-10}
else
    echo "Unknown MODE=$MODE; expected audit, smoke, train, or paired" >&2
    exit 2
fi

EXTRA_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    EXTRA_ARGS+=(--resume="$RESUME")
fi
if [[ -n "${TRANSFER:-}" ]]; then
    EXTRA_ARGS+=(--transfer="$TRANSFER")
fi

{
    echo "PaDIS-MRI AGG ablation training"
    echo "MODE=$MODE GPU=$GPU SEED=$SEED DURATION_MIMG=$DURATION_MIMG"
    echo "PATCH_SIZES=$PATCH_SIZES PATCH_PROBS=$PATCH_PROBS"
    echo "AGG_RHO=$AGG_RHO AGG_GAMMA=$AGG_GAMMA"
    echo "OUTDIR=$OUTDIR"
    echo "DATA_PATH=$DATA_PATH"
} | tee "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPU" torchrun \
    --standalone \
    --nproc_per_node=1 \
    "$ENTRY_POINT" \
    "${COMMON_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    --duration="$DURATION_MIMG" \
    --tick="$TICK" \
    --snap="$SNAP" \
    --dump="$DUMP" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"
