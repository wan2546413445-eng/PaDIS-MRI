#!/bin/bash
set -e
set -o pipefail

GPU=2
NPROC=1

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI
ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/training-runs-cross
ROOT_DATA=/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200
LOG_DIR=$RESULT_ROOT/results_record/logs

SNR=32dB

mkdir -p $ROOT_OUTDIR
mkdir -p $LOG_DIR

MODE=${1:-debug}

EXP_NAME=cp64_k8_l3_g4_cbase96_b4_fp32


if [ "$MODE" = "debug" ]; then
  RUN_NAME=debug_${EXP_NAME}
  DURATION=0.01
  SNAP=1
  TICK=1
elif [ "$MODE" = "short" ]; then
  RUN_NAME=short_${EXP_NAME}
  DURATION=1
  SNAP=10
  TICK=1
else
  RUN_NAME=full_${EXP_NAME}
  DURATION=200
  SNAP=50
  TICK=1
fi

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_cross_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

echo "==================================================" | tee $LOG_FILE
echo "Cross-Patch PaDIS-MRI Training" | tee -a $LOG_FILE
echo "MODE=$MODE" | tee -a $LOG_FILE
echo "EXP_NAME=$EXP_NAME" | tee -a $LOG_FILE
echo "RUN_NAME=$RUN_NAME" | tee -a $LOG_FILE
echo "GPU=$GPU" | tee -a $LOG_FILE
echo "DURATION=$DURATION" | tee -a $LOG_FILE
echo "CODE_ROOT=$CODE_ROOT" | tee -a $LOG_FILE
echo "RESULT_ROOT=$RESULT_ROOT" | tee -a $LOG_FILE
echo "LOG_FILE=$LOG_FILE" | tee -a $LOG_FILE
echo "OUTDIR=$ROOT_OUTDIR/$RUN_NAME" | tee -a $LOG_FILE
echo "DATA=$ROOT_DATA/$SNR" | tee -a $LOG_FILE
echo "==================================================" | tee -a $LOG_FILE

cd $CODE_ROOT

CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --nproc_per_node=$NPROC train/padis-mri/cross_train.py \
  --outdir=$ROOT_OUTDIR/$RUN_NAME \
  --data=$ROOT_DATA/$SNR \
  --cond=0 \
  --arch=ddpmpp \
  --precond=pedm \
  --batch=4 \
  --batch-gpu=4 \
  --cbase=96 \
  --lr=1e-4 \
  --dropout=0.05 \
  --augment=0 \
  --padding=1 \
  --pad_width=96 \
  --duration=$DURATION \
  --tick=$TICK \
  --snap=$SNAP \
  --seed=2025 \
  --cp_k=8 \
  --cp_local_k=3 \
  --cp_global_k=4 \
  --cp_patch_size=64 \
  --cp_depth=2 \
  --cp_num_heads=4 \
  --fp16=0 \
  2>&1 | tee -a $LOG_FILE