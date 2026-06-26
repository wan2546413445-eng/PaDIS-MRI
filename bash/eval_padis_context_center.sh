#!/bin/bash
set -e
set -o pipefail

GPU=1
CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI
LOG_DIR=$RESULT_ROOT/results_record/logs

MODEL_PATH=${MODEL_PATH:?Please set MODEL_PATH}
VAL_DIR=${VAL_DIR:?Please set VAL_DIR}

SAVE_DIR=${SAVE_DIR:-$RESULT_ROOT/PaDIS-MRI-runs/context_center_eval_default}

CONTEXT_MARGIN=16
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_context_center_m${CONTEXT_MARGIN}_gpu${GPU}_${TIME_TAG}.log

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

cd "$CODE_ROOT"

CUDA_VISIBLE_DEVICES=$GPU python eval/run_context_center.py \
    --run_evaluate \
    --algo padis \
    --model_path "$MODEL_PATH" \
    --val_dir "$VAL_DIR" \
    --image_size 384 \
    --pad 64 \
    --psize 64 \
    --context_margin $CONTEXT_MARGIN \
    --mask_select 7 \
    --val_count 32 \
    --seed 123 \
    --zeta 3.0 \
    --steps 78 \
    --report_every 1 \
    --gpus 0 \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$LOG_FILE"
