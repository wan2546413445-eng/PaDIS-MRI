#!/bin/bash
set -e
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

GPU=${GPU:-0}
NPROC=1
CODE_ROOT=${CODE_ROOT:-/workspace/PaDIS-MRI}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/overlap_same_training-runs
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}
LOG_DIR=$RESULT_ROOT/results_record/logs
ANATOMY=brain
SNR=32dB
BATCH_SIZE=2
LR=1e-4
DROPOUT=0.05
REAL_P=0.5
PADDING=1
PAD_WIDTH=96
PATCH_SIZES=16,32,64
PROBS=0.2,0.3,0.5
WORKERS=4
SEED=123
EXP_NAME=overlap_same_lam001_s16s32s64_p020305_b2_seed123
MODE=${1:-debug}
if [ "$MODE" = "debug" ]; then DURATION=0.01; TICK=1; SNAP=1; DUMP=1;
elif [ "$MODE" = "overnight" ]; then DURATION=8.064; TICK=5; SNAP=200; DUMP=1000;
elif [ "$MODE" = "main" ]; then DURATION=200; TICK=5; SNAP=200; DUMP=1000;
else echo "Unknown mode: $MODE (use debug|overnight|main)"; exit 1; fi
RUN_NAME=${MODE}_${EXP_NAME}
OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_DIR=$ROOT_DATA/$SNR
mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_overlap_same_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log
cd "$CODE_ROOT"
{
  echo "Overlap Same-Noise PaDIS-MRI Training"; echo "MODE=$MODE"; echo "RUN_NAME=$RUN_NAME"; echo "GPU=$GPU"; echo "OUTDIR=$OUTDIR"; echo "DATA=$DATA_DIR";
} | tee "$LOG_FILE"
CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --nproc_per_node=$NPROC train/padis-mri/train_overlap.py \
  --outdir=$OUTDIR --data=$DATA_DIR --cond=0 --arch=ddpmpp --precond=pedm \
  --overlap-mode=same --lambda-overlap=0.01 \
  --batch=$BATCH_SIZE --lr=$LR --dropout=$DROPOUT --augment=0 --real_p=$REAL_P \
  --padding=$PADDING --pad_width=$PAD_WIDTH --patch-list=$PATCH_SIZES --patch-probs=$PROBS \
  --duration=$DURATION --tick=$TICK --snap=$SNAP --dump=$DUMP --workers=$WORKERS --seed=$SEED \
  2>&1 | tee -a "$LOG_FILE"
