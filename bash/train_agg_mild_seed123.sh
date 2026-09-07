#!/usr/bin/env bash
set -Eeuo pipefail

# PaDIS-MRI AGG-mild 单独训练：
# - 从零开始训练
# - train seed 固定为 123
# - 训练到 10000 kimg
# - 不包含 overlap loss
#
# 默认使用物理 GPU 1；可临时切换：
#   GPU=0 bash train_agg_mild_seed123.sh
#
# 后台运行示例：
#   nohup bash train_agg_mild_seed123.sh \
#     > train_agg_mild_seed123_launcher.log 2>&1 &

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${GPU:-1}"
NPROC=1
SEED=123

CODE_ROOT="${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}"
ROOT_DATA="${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}"

ROOT_OUTDIR="${RESULT_ROOT}/PaDIS-MRI-runs/agg_mild_seed123_to10k"
LOG_DIR="${RESULT_ROOT}/results_record/logs"

ANATOMY=brain
SNR=32dB

BATCH_SIZE=2
LR=1e-4
DROPOUT=0.05
REAL_P=0.5
PADDING=1
PAD_WIDTH=64
PATCH_SIZES=16,32,64
PATCH_PROBS=0.2,0.3,0.5
WORKERS=4

# train_agg.py 中 duration 的单位是 MIMG。
# 10 MIMG = 10000 kimg。
DURATION=10

# 复刻原 AGG-mild 0~10000 kimg 阶段的日志/保存节奏：
# tick=10 kimg；每 1000 tick 保存 snapshot；每 2000 tick 保存训练状态。
TICK=10
SNAP=100
DUMP=200

# 原 AGG-mild 参数。
export PADIS_AGG_ENABLE=1
export PADIS_AGG_SAMPLE_PROB=0.65
export PADIS_AGG_THR_RATIO=0.05
export PADIS_AGG_BASE=0.08
export PADIS_AGG_RADIUS_SCALE=1.15
export PADIS_AGG_TAU_SCALE=0.30
export PADIS_AGG_MIN_PIXELS=64
export PADIS_AGG_GRAD_ALPHA=1.0
export PADIS_AGG_GRAD_CLIP_Q=0.95
export PADIS_AGG_GATE_BASE=0.25

EXP_NAME="d384_pad64_AGG_mild_s123_to10k"
RUN_ROOT="${ROOT_OUTDIR}/${ANATOMY}/${SNR}"
DATA_DIR="${ROOT_DATA}/${SNR}"

TIME_TAG="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/train_${EXP_NAME}_gpu${GPU}_${TIME_TAG}.log"

CURRENT_STAGE="startup"
trap 'status=$?; echo; echo "[ERROR] stage=${CURRENT_STAGE}, line=${LINENO}, exit=${status}" >&2; echo "日志：${LOG_FILE}" >&2; exit ${status}' ERR

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

if [[ ! -d "${CODE_ROOT}" ]]; then
    echo "找不到代码目录：${CODE_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${CODE_ROOT}/train/padis-mri/train_agg.py" ]]; then
    echo "找不到训练入口：${CODE_ROOT}/train/padis-mri/train_agg.py" >&2
    exit 2
fi
if [[ ! -f "${DATA_DIR}/noisy.pt" ]]; then
    echo "找不到训练数据：${DATA_DIR}/noisy.pt" >&2
    exit 2
fi

cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}/train/padis-mri:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_NAME="${EXP_NAME}"

GIT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
GIT_DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
{
    echo "============================================================"
    echo "PaDIS-MRI AGG-mild standalone training"
    echo "START_FROM_ZERO=true"
    echo "OVERLAP_ENABLE=false"
    echo "GPU=${GPU}"
    echo "SEED=${SEED}"
    echo "TARGET_KIMG=10000"
    echo "CODE_ROOT=${CODE_ROOT}"
    echo "DATA_DIR=${DATA_DIR}"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "LR=${LR}"
    echo "DROPOUT=${DROPOUT}"
    echo "PAD_WIDTH=${PAD_WIDTH}"
    echo "PATCH_SIZES=${PATCH_SIZES}"
    echo "PATCH_PROBS=${PATCH_PROBS}"
    echo "TICK=${TICK}"
    echo "SNAP=${SNAP}"
    echo "DUMP=${DUMP}"
    echo "PADIS_AGG_SAMPLE_PROB=${PADIS_AGG_SAMPLE_PROB}"
    echo "PADIS_AGG_THR_RATIO=${PADIS_AGG_THR_RATIO}"
    echo "PADIS_AGG_BASE=${PADIS_AGG_BASE}"
    echo "PADIS_AGG_RADIUS_SCALE=${PADIS_AGG_RADIUS_SCALE}"
    echo "PADIS_AGG_TAU_SCALE=${PADIS_AGG_TAU_SCALE}"
    echo "PADIS_AGG_MIN_PIXELS=${PADIS_AGG_MIN_PIXELS}"
    echo "PADIS_AGG_GRAD_ALPHA=${PADIS_AGG_GRAD_ALPHA}"
    echo "PADIS_AGG_GRAD_CLIP_Q=${PADIS_AGG_GRAD_CLIP_Q}"
    echo "PADIS_AGG_GATE_BASE=${PADIS_AGG_GATE_BASE}"
    echo "GIT_BRANCH=${GIT_BRANCH}"
    echo "GIT_COMMIT=${GIT_COMMIT}"
    echo "GIT_DIRTY_FILE_COUNT=${GIT_DIRTY}"
    echo "============================================================"
} | tee "${LOG_FILE}"

CURRENT_STAGE="syntax_check"
python -m py_compile \
    train/padis-mri/train_agg.py \
    train/padis-mri/training/training_loop_agg.py \
    train/padis-mri/training/patch_loss_agg.py \
    2>&1 | tee -a "${LOG_FILE}"

CURRENT_STAGE="training"
CUDA_VISIBLE_DEVICES="${GPU}" torchrun \
    --standalone \
    --nproc_per_node="${NPROC}" \
    train/padis-mri/train_agg.py \
    --outdir="${RUN_ROOT}" \
    --data="${DATA_DIR}" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=pedm \
    --batch="${BATCH_SIZE}" \
    --lr="${LR}" \
    --dropout="${DROPOUT}" \
    --augment=0 \
    --real_p="${REAL_P}" \
    --padding="${PADDING}" \
    --pad_width="${PAD_WIDTH}" \
    --patch-list="${PATCH_SIZES}" \
    --patch-probs="${PATCH_PROBS}" \
    --duration="${DURATION}" \
    --tick="${TICK}" \
    --snap="${SNAP}" \
    --dump="${DUMP}" \
    --seed="${SEED}" \
    --workers="${WORKERS}" \
    --desc="${EXP_NAME}" \
    2>&1 | tee -a "${LOG_FILE}"

CURRENT_STAGE="done"
echo
echo "训练结束。"
echo "运行根目录：${RUN_ROOT}"
echo "日志：${LOG_FILE}"
