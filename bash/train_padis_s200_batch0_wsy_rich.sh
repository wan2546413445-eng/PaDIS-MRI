#!/bin/bash
set -e
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Train PaDIS baseline on FastMRI brain dataset.
# This script keeps the original baseline training arguments unchanged,
# and only adds mode selection, structured logging, and clearer output naming.

GPU=1
NPROC=1

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI
ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/baseline_training-runs_sigma16_full
ROOT_DATA=/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200
LOG_DIR=$RESULT_ROOT/results_record/logs

ANATOMY=brain
SNR=32dB

MAIN_BATCH_SIZE=2

BATCH_SIZE=$MAIN_BATCH_SIZE
LR=1e-4
DROPOUT=0.05
REAL_P=0.5
PADDING=1
PAD_WIDTH=96
PATCH_SIZES=16,32,64
PROBS=0.2,0.3,0.5
WORKERS=4
SEED=123
SIGMA_DATA=0.16
P_MEAN=-2.34
P_STD=1.2

# Optional continuation controls. Leave both empty for a fresh run.
# Exact resume:
#   RESUME_STATE=/path/to/training-state-001000.pt bash bash/train_padis_s200_batch0_wsy.sh main
# Weight-only warm-start:
#   TRANSFER_PKL=/path/to/network-snapshot-001000.pkl bash bash/train_padis_s200_batch0_wsy.sh main
RESUME_STATE=${RESUME_STATE:-}
TRANSFER_PKL=${TRANSFER_PKL:-}

mkdir -p "$ROOT_OUTDIR"
mkdir -p "$LOG_DIR"

MODE=${1:-debug}

if [ "$MODE" = "debug" ]; then
  DURATION=0.01
  TICK=1
  SNAP=1
  DUMP=1
elif [ "$MODE" = "probe" ]; then
  DURATION=1
  TICK=1
  SNAP=10
  DUMP=10
elif [ "$MODE" = "main" ]; then
  DURATION=200
  TICK=5
  SNAP=200
  DUMP=1000
else
  echo "Unknown mode: $MODE (use debug|probe|main)"
  exit 1
fi

EXP_NAME=baseline_sigma16_full_s16s32s64_p020305_b${BATCH_SIZE}_seed${SEED}
RUN_NAME=${MODE}_${EXP_NAME}
OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_DIR=$ROOT_DATA/$SNR
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_baseline_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"

RESUME_ARGS=()
if [ -n "$RESUME_STATE" ]; then
  RESUME_ARGS+=(--resume=$RESUME_STATE)
elif [ -n "$TRANSFER_PKL" ]; then
  RESUME_ARGS+=(--transfer=$TRANSFER_PKL)
fi

GIT_BRANCH=$(git branch --show-current 2>/dev/null || true)
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || true)


{
  echo "=================================================="
  echo "Baseline PaDIS-MRI Training"
  echo "MODE=$MODE"
  echo "EXP_NAME=$EXP_NAME"
  echo "RUN_NAME=$RUN_NAME"
  echo "GPU=$GPU"
  echo "NPROC=$NPROC"
  echo "DURATION=$DURATION"
  echo "SNAP=$SNAP"
  echo "TICK=$TICK"
  echo "DUMP=$DUMP"
  echo "RESUME_STATE=$RESUME_STATE"
  echo "TRANSFER_PKL=$TRANSFER_PKL"
  echo "CODE_ROOT=$CODE_ROOT"
  echo "SIGMA_DATA=$SIGMA_DATA"
  echo "P_MEAN=$P_MEAN"
  echo "P_STD=$P_STD"
  echo "RESULT_ROOT=$RESULT_ROOT"
  echo "LOG_FILE=$LOG_FILE"
  echo "OUTDIR=$OUTDIR"
  echo "DATA=$DATA_DIR"
  echo "ANATOMY=$ANATOMY"
  echo "SNR=$SNR"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "LR=$LR"
  echo "DROPOUT=$DROPOUT"
  echo "REAL_P=$REAL_P"
  echo "PADDING=$PADDING"
  echo "PAD_WIDTH=$PAD_WIDTH"
  echo "PATCH_SIZES=$PATCH_SIZES"
  echo "PROBS=$PROBS"
  echo "WORKERS=$WORKERS"
  echo "SEED=$SEED"
  echo "GIT_BRANCH=$GIT_BRANCH"
  echo "GIT_COMMIT=$GIT_COMMIT"

  echo "=================================================="
} | tee "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --nproc_per_node=$NPROC train/padis-mri/train_sigma016.py \
  --outdir=$OUTDIR \
  --data=$DATA_DIR \
  --cond=0 \
  --arch=ddpmpp \
  --batch=$BATCH_SIZE \
  --lr=$LR \
  --dropout=$DROPOUT \
  --augment=0 \
  --real_p=$REAL_P \
  --padding=$PADDING \
  --tick=$TICK \
  --snap=$SNAP \
  --dump=$DUMP \
  --sigma-data=$SIGMA_DATA \
  --p-mean=$P_MEAN \
  --p-std=$P_STD \
  --precond=pedm \
  --seed=$SEED \
  --pad_width=$PAD_WIDTH \
  "${RESUME_ARGS[@]}" \
  --patch-list=$PATCH_SIZES \
  --patch-probs=$PROBS \
  --duration=$DURATION \
  --workers=$WORKERS \
  2>&1 | tee -a "$LOG_FILE"
