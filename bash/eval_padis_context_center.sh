#!/bin/bash
set -e
set -o pipefail

GPU=${GPU:-1}
CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI
LOG_DIR=$RESULT_ROOT/results_record/logs

MODEL_PATH=${MODEL_PATH:?Please set MODEL_PATH}
VAL_DIR=${VAL_DIR:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB}

CONTEXT_MARGIN=${CONTEXT_MARGIN:-16}
CHUNK=${CHUNK:-8}
SAMPLE_INDICES=${SAMPLE_INDICES:-18}
VAL_COUNT=${VAL_COUNT:-1}

SAVE_DIR=${SAVE_DIR:-$RESULT_ROOT/PaDIS-MRI-recon/context_center_m${CONTEXT_MARGIN}_eval}

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_context_center_m${CONTEXT_MARGIN}_gpu${GPU}_${TIME_TAG}.log

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

cd "$CODE_ROOT"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=================================================="
echo "PaDIS-MRI Context-Center Eval"
echo "GPU=$GPU"
echo "MODEL_PATH=$MODEL_PATH"
echo "VAL_DIR=$VAL_DIR"
echo "SAVE_DIR=$SAVE_DIR"
echo "CONTEXT_MARGIN=$CONTEXT_MARGIN"
echo "CHUNK=$CHUNK"
echo "SAMPLE_INDICES=$SAMPLE_INDICES"
echo "VAL_COUNT=$VAL_COUNT"
echo "=================================================="

CUDA_VISIBLE_DEVICES=$GPU python eval/run_context_center.py \
    --run_evaluate \
    --algo padis \
    --model_path "$MODEL_PATH" \
    --val_dir "$VAL_DIR" \
    --image_size 384 \
    --pad 64 \
    --psize 64 \
    --context_margin "$CONTEXT_MARGIN" \
    --chunk "$CHUNK" \
    --mask_select 7 \
    --val_count "$VAL_COUNT" \
    --sample_indices "$SAMPLE_INDICES" \
    --seed 123 \
    --zeta 3.0 \
    --steps 78 \
    --inner_loops 10 \
    --report_every 1 \
    --gpus 0 \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$LOG_FILE"
