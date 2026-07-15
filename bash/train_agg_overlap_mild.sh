#!/usr/bin/env bash
set -euo pipefail

# 单卡、单 seed 的 AGG-mild + independent-overlap 训练。
#
# 示例：
#   GPU=0 SEED=123  bash bash/train_agg_overlap_mild.sh
#   GPU=1 SEED=2025 bash bash/train_agg_overlap_mild.sh
#
# 两卡并行：
#   GPU=0 SEED=123  nohup bash bash/train_agg_overlap_mild.sh > agg_overlap_s123.out 2>&1 &
#   GPU=1 SEED=2025 nohup bash bash/train_agg_overlap_mild.sh > agg_overlap_s2025.out 2>&1 &

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

GPU=${GPU:-0}
SEED=${SEED:-123}
NPROC=1

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/agg_overlap_mild_training-runs
LOG_DIR=$RESULT_ROOT/results_record/logs

ANATOMY=brain
SNR=32dB

BATCH_SIZE=2
BATCH_GPU=1
LR=1e-4
DROPOUT=0.05
REAL_P=0.5
PADDING=1
PAD_WIDTH=64
WORKERS=4

PATCH_SIZES=16,32,64
PATCH_PROBS=0.2,0.3,0.5

LAMBDA_OVERLAP=0.3
ACTIVE_PATCH_SIZE=64

AGG_SAMPLE_PROB=0.65
AGG_THR_RATIO=0.05
AGG_BASE=0.08
AGG_RADIUS_SCALE=1.15
AGG_TAU_SCALE=0.30
AGG_MIN_PIXELS=64
AGG_GRAD_ALPHA=1.0
AGG_GRAD_CLIP_Q=0.95
AGG_GATE_BASE=0.25

# train_overlap.py 中 duration 的单位是 MIMG：
# 15 MIMG = 15000 kimg。
DURATION=15

# 保存逻辑：
#   每 100 kimg 形成一个 tick；
#   每 10 tick 保存 network snapshot  -> 每 1000 kimg；
#   每 20 tick 保存 training state     -> 每 2000 kimg；
#   训练达到 15000 kimg 时 done=True，会额外保存最终 snapshot/state 并退出。
TICK=100
SNAP=10
DUMP=20

PATCH_TAG=s16s32s64_p020305
LAMBDA_TAG=${LAMBDA_OVERLAP//./p}
EXP_NAME=agg_mild_overlap_independent_center_lam${LAMBDA_TAG}_active${ACTIVE_PATCH_SIZE}_pad${PAD_WIDTH}_bgpu${BATCH_GPU}_${PATCH_TAG}_b${BATCH_SIZE}_seed${SEED}
RUN_NAME=main_${EXP_NAME}

OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_DIR=$ROOT_DATA/$SNR

mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export WANDB_NAME="$RUN_NAME"

# 新 loss 完全使用 CLI/loss_kwargs 参数，不读取旧 AGG 环境变量。
unset PADIS_AGG_ENABLE PADIS_AGG_SAMPLE_PROB PADIS_AGG_THR_RATIO \
      PADIS_AGG_BASE PADIS_AGG_RADIUS_SCALE PADIS_AGG_TAU_SCALE \
      PADIS_AGG_MIN_PIXELS PADIS_AGG_GRAD_ALPHA PADIS_AGG_GRAD_CLIP_Q \
      PADIS_AGG_GATE_BASE || true

{
    echo "=================================================="
    echo "PaDIS-MRI AGG-mild + Independent Overlap Training"
    echo "START_FROM_ZERO=true"
    echo "GPU=$GPU"
    echo "SEED=$SEED"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "BATCH_GPU=$BATCH_GPU"
    echo "LR=$LR"
    echo "DROPOUT=$DROPOUT"
    echo "PAD_WIDTH=$PAD_WIDTH"
    echo "PATCH_SIZES=$PATCH_SIZES"
    echo "PATCH_PROBS=$PATCH_PROBS"
    echo "LAMBDA_OVERLAP=$LAMBDA_OVERLAP"
    echo "ACTIVE_PATCH_SIZE=$ACTIVE_PATCH_SIZE"
    echo "AGG_SAMPLE_PROB=$AGG_SAMPLE_PROB"
    echo "AGG_THR_RATIO=$AGG_THR_RATIO"
    echo "AGG_BASE=$AGG_BASE"
    echo "AGG_RADIUS_SCALE=$AGG_RADIUS_SCALE"
    echo "AGG_TAU_SCALE=$AGG_TAU_SCALE"
    echo "AGG_MIN_PIXELS=$AGG_MIN_PIXELS"
    echo "AGG_GRAD_ALPHA=$AGG_GRAD_ALPHA"
    echo "AGG_GRAD_CLIP_Q=$AGG_GRAD_CLIP_Q"
    echo "AGG_GATE_BASE=$AGG_GATE_BASE"
    echo "TARGET_KIMG=15000"
    echo "SNAPSHOT_EVERY_KIMG=1000"
    echo "STATE_DUMP_EVERY_KIMG=2000"
    echo "OUTDIR=$OUTDIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "LOG_FILE=$LOG_FILE"
    echo "=================================================="
} | tee "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPU" torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    train/padis-mri/train_overlap.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=pedm \
    --lambda-overlap="$LAMBDA_OVERLAP" \
    --active-patch-size="$ACTIVE_PATCH_SIZE" \
    --agg-sample-prob="$AGG_SAMPLE_PROB" \
    --agg-thr-ratio="$AGG_THR_RATIO" \
    --agg-base="$AGG_BASE" \
    --agg-radius-scale="$AGG_RADIUS_SCALE" \
    --agg-tau-scale="$AGG_TAU_SCALE" \
    --agg-min-pixels="$AGG_MIN_PIXELS" \
    --agg-grad-alpha="$AGG_GRAD_ALPHA" \
    --agg-grad-clip-q="$AGG_GRAD_CLIP_Q" \
    --agg-gate-base="$AGG_GATE_BASE" \
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
    --lambda-overlap=0.3 \
    --active-patch-size=64 \
    --agg-sample-prob=0.65 \
    --agg-thr-ratio=0.05 \
    --agg-base=0.08 \
    --agg-radius-scale=1.15 \
    --agg-tau-scale=0.30 \
    --agg-min-pixels=64 \
    --agg-grad-alpha=1.0 \
    --agg-grad-clip-q=0.95 \
    --agg-gate-base=0.25
    2>&1 | tee -a "$LOG_FILE"
