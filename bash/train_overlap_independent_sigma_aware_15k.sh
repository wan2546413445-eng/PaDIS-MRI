#!/bin/bash
set -e
set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# PaDIS-MRI sigma-aware overlap consistency:
# - 总训练目标：15 Mimg，即 15000 kimg
# - network snapshot：每 200 ticks 保存一次
# - training state：每 400 ticks 保存一次
#
# 当前 training_loop 使用 TICK=5 kimg，但由于混合 patch 的平均 batch multiplier，
# 每个实际 tick 约为 5.04 kimg。因此文件名通常表现为：
# network-snapshot-001008.pkl, 002016.pkl, 003024.pkl, ...
# training-state-002016.pt, 004032.pt, 006048.pt, ...
# 训练结束时还会额外保存最终的 015000 文件。
#
# 前台启动：
# GPU=4 /bin/bash bash/train_overlap_independent_sigma_aware_15k.sh
#
# 后台启动示例：
# nohup env GPU=4 /bin/bash bash/train_overlap_independent_sigma_aware_15k.sh \
#   > sigma_aware_15k_launcher.log 2>&1 &

GPU=${GPU:-4}
NPROC=1

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

ROOT_OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/overlap_sigma_aware_training-runs
LOG_DIR=$RESULT_ROOT/results_record/logs
ANATOMY=brain
SNR=32dB

# 与原 CG-OL only64 正式实验保持一致。
BATCH_SIZE=2
BATCH_GPU=1
LR=1e-4
DROPOUT=0.05
REAL_P=0.5
PADDING=1
PAD_WIDTH=96
WORKERS=4
SEED=${SEED:-123}

PATCH_SIZES=16,32,64
PATCH_PROBS=0.2,0.3,0.5
PATCH_TAG=s16s32s64_p020305

# Sigma-aware overlap 参数。
LAMBDA_OVERLAP=${LAMBDA_OVERLAP:-0.3}
OVERLAP_WEIGHT_MODE=${OVERLAP_WEIGHT_MODE:-sigmoid}
OVERLAP_SIGMA_CENTER=${OVERLAP_SIGMA_CENTER:-2.0}
OVERLAP_SIGMA_SLOPE=${OVERLAP_SIGMA_SLOPE:-2.0}
OVERLAP_SIGMA_LOW=${OVERLAP_SIGMA_LOW:-0.1}
OVERLAP_SIGMA_HIGH=${OVERLAP_SIGMA_HIGH:-0.5}

# 训练与保存设置。
DURATION=15.0
TICK=5
SNAP=200
DUMP=400

case "$OVERLAP_WEIGHT_MODE" in
    fixed|sigmoid|piecewise)
        ;;
    *)
        echo "Unknown OVERLAP_WEIGHT_MODE=$OVERLAP_WEIGHT_MODE"
        echo "必须使用 fixed、sigmoid 或 piecewise"
        exit 1
        ;;
esac

safe_tag() {
    echo "$1" | sed 's/-/m/g; s/\./p/g'
}

LAMBDA_TAG=$(safe_tag "$LAMBDA_OVERLAP")
CENTER_TAG=$(safe_tag "$OVERLAP_SIGMA_CENTER")
SLOPE_TAG=$(safe_tag "$OVERLAP_SIGMA_SLOPE")
LOW_TAG=$(safe_tag "$OVERLAP_SIGMA_LOW")
HIGH_TAG=$(safe_tag "$OVERLAP_SIGMA_HIGH")

case "$OVERLAP_WEIGHT_MODE" in
    fixed)
        WEIGHT_TAG=fixed
        ;;
    sigmoid)
        WEIGHT_TAG=sigmoid_c${CENTER_TAG}_k${SLOPE_TAG}_bins${LOW_TAG}_${HIGH_TAG}
        ;;
    piecewise)
        WEIGHT_TAG=piecewise_s${LOW_TAG}_${HIGH_TAG}
        ;;
esac

EXP_NAME=overlap_independent_${WEIGHT_TAG}_lam${LAMBDA_TAG}_active64_bgpu1_${PATCH_TAG}_b2_seed${SEED}
RUN_NAME=train15k_${EXP_NAME}

OUTDIR=$ROOT_OUTDIR/$ANATOMY/$SNR/$RUN_NAME
DATA_DIR=$ROOT_DATA/$SNR
mkdir -p "$ROOT_OUTDIR" "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/train_${RUN_NAME}_gpu${GPU}_${TIME_TAG}.log

cd "$CODE_ROOT"

{
    echo "=================================================="
    echo "PaDIS-MRI Sigma-Aware Independent Overlap Training"
    echo "MODE=train15k"
    echo "GPU=$GPU"
    echo "LAMBDA_OVERLAP=$LAMBDA_OVERLAP"
    echo "OVERLAP_WEIGHT_MODE=$OVERLAP_WEIGHT_MODE"
    echo "OVERLAP_SIGMA_CENTER=$OVERLAP_SIGMA_CENTER"
    echo "OVERLAP_SIGMA_SLOPE=$OVERLAP_SIGMA_SLOPE"
    echo "OVERLAP_SIGMA_LOW=$OVERLAP_SIGMA_LOW"
    echo "OVERLAP_SIGMA_HIGH=$OVERLAP_SIGMA_HIGH"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "BATCH_GPU=$BATCH_GPU"
    echo "ACTIVE_PATCH_SIZE=64"
    echo "PATCH_SIZES=$PATCH_SIZES"
    echo "PATCH_PROBS=$PATCH_PROBS"
    echo "SAME_SIGMA_WITHIN_PAIR=true"
    echo "INDEPENDENT_NOISE_REALIZATION=true"
    echo "CENTER_TO_BOUNDARY=true"
    echo "TARGET_DURATION_MIMG=$DURATION"
    echo "KIMG_PER_TICK=$TICK"
    echo "SNAPSHOT_EVERY_TICKS=$SNAP"
    echo "STATE_DUMP_EVERY_TICKS=$DUMP"
    echo "EXPECTED_SNAPSHOT_INTERVAL_KIMG≈1008"
    echo "EXPECTED_STATE_INTERVAL_KIMG≈2016"
    echo "OUTDIR=$OUTDIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "=================================================="
} | tee "$LOG_FILE"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone \
    --nproc_per_node=$NPROC \
    train/padis-mri/train_overlap_sigma_aware.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=pedm \
    --overlap-mode=independent \
    --lambda-overlap="$LAMBDA_OVERLAP" \
    --overlap-weight-mode="$OVERLAP_WEIGHT_MODE" \
    --overlap-sigma-center="$OVERLAP_SIGMA_CENTER" \
    --overlap-sigma-slope="$OVERLAP_SIGMA_SLOPE" \
    --overlap-sigma-low="$OVERLAP_SIGMA_LOW" \
    --overlap-sigma-high="$OVERLAP_SIGMA_HIGH" \
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
    2>&1 | tee -a "$LOG_FILE"
