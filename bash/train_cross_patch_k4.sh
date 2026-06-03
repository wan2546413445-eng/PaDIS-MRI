#!/bin/bash
set -e
set -o pipefail

GPU=0
NPROC=1

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI
ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/training-runs-cross
ROOT_DATA=/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200
LOG_DIR=$RESULT_ROOT/results_record/logs

SNR=32dB

# Optional continuation controls. Leave both empty for a fresh run.
# Exact resume, same as original PaDIS-MRI style:
#   RESUME_STATE=/path/to/training-state-001000.pt bash bash/train_cross_patch_k4.sh main
# Weight-only warm-start:
#   TRANSFER_PKL=/path/to/network-snapshot-001000.pkl bash bash/train_cross_patch_k4.sh main
RESUME_STATE=${RESUME_STATE:-}
TRANSFER_PKL=${TRANSFER_PKL:-}

mkdir -p $ROOT_OUTDIR
mkdir -p $LOG_DIR

MODE=${1:-debug}

EXP_NAME=cross_patch_s16s32s64_k4_l1g2_d1h2ffn2_cbase96_b6_fp32

if [ "$MODE" = "debug" ]; then
  RUN_NAME=debug_${EXP_NAME}
  DURATION=0.01
  SNAP=1
  TICK=1
  DUMP=1
elif [ "$MODE" = "probe" ]; then
  RUN_NAME=probe_${EXP_NAME}
  DURATION=1
  SNAP=100
  TICK=10
  DUMP=100
elif [ "$MODE" = "main" ]; then
  RUN_NAME=main_${EXP_NAME}
  DURATION=5
  TICK=1
  SNAP=100
  DUMP=500
elif [ "$MODE" = "full" ]; then
  RUN_NAME=full_${EXP_NAME}
  DURATION=15
  SNAP=100
  TICK=10
  DUMP=100
else
  echo "Unknown mode: $MODE (use debug|probe|main|full)"
  exit 1
fi

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_cross_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

RESUME_ARGS=()
if [ -n "$RESUME_STATE" ]; then
  RESUME_ARGS+=(--resume=$RESUME_STATE)
elif [ -n "$TRANSFER_PKL" ]; then
  RESUME_ARGS+=(--transfer=$TRANSFER_PKL)
fi

echo "==================================================" | tee $LOG_FILE
echo "Cross-Patch PaDIS-MRI Training" | tee -a $LOG_FILE
echo "MODE=$MODE" | tee -a $LOG_FILE
echo "EXP_NAME=$EXP_NAME" | tee -a $LOG_FILE
echo "RUN_NAME=$RUN_NAME" | tee -a $LOG_FILE
echo "GPU=$GPU" | tee -a $LOG_FILE
echo "DURATION=$DURATION" | tee -a $LOG_FILE
echo "SNAP=$SNAP" | tee -a $LOG_FILE
echo "TICK=$TICK" | tee -a $LOG_FILE
echo "DUMP=$DUMP" | tee -a $LOG_FILE
echo "RESUME_STATE=$RESUME_STATE" | tee -a $LOG_FILE
echo "TRANSFER_PKL=$TRANSFER_PKL" | tee -a $LOG_FILE
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
  --batch=12 \
  --batch-gpu=12 \
  --cbase=96 \
  --lr=1e-4 \
  --dropout=0.05 \
  --augment=0 \
  --padding=1 \
  --pad_width=96 \
  --duration=$DURATION \
  --tick=$TICK \
  --snap=$SNAP \
  --dump=$DUMP \
  --seed=2025 \
  "${RESUME_ARGS[@]}" \
  --cp_k=4 \
  --cp_local_k=1 \
  --cp_global_k=2 \
  --cp_patch_size=64 \
  --patch-list=16,32,64 \
  --patch-probs=0.2,0.3,0.5 \
  --cp_depth=1 \
  --cp_num_heads=2 \
  --cp_ffn_mult=2 \
  --fp16=0 \
  2>&1 | tee -a $LOG_FILE

#结构名：cross_patch_s16s32s64_k4_l1g2_d1h2ffn2_cbase96_b6_fp32
 #K 配置：
 #cp_k = 4
 #cp_local_k = 1
 #cp_global_k = 2
 #
 #Transformer 配置：
 #cp_depth = 1
 #cp_num_heads = 2
 #cp_ffn_mult = 2
 #
 #训练配置：
 #batch = 6
 #batch_gpu = 6
 #patch-list = 16,32,64
 #patch-probs = 0.2,0.3,0.5
 #fp16 = 0