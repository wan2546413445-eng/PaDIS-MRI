#!/usr/bin/env bash
set -euo pipefail

# Full 32-sample PaDIS evaluation with the same seed reset before every sample.
#
# Required:
#   MODEL_PATH=/absolute/path/to/network-snapshot-xxxxxx.pkl
#   EXP_NAME=baseline_ckpt010047_fixed_seed123_32
#
# Optional:
#   GPU=0
#   CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
#   RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI
#   VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB
#   PAD=64
#   PSIZE=64
#   STEPS=78
#   INNER_LOOPS=10
#   ZETA=3.0
#
# Example:
#   GPU=0 \
#   MODEL_PATH=/path/to/baseline/network-snapshot-010047.pkl \
#   EXP_NAME=old_pad96_ckpt010047_fixed_seed123_32 \
#   nohup bash bash/eval_32_fixed_seed123.sh \
#   > baseline_fixed_seed123_32.out 2>&1 &
#
#   GPU=1 \
#   MODEL_PATH=/path/to/agg/network-snapshot-010000.pkl \
#   EXP_NAME=AGG_mild_ckpt010000_fixed_seed123_32 \
#   nohup bash bash/eval_32_fixed_seed123.sh \
#   > agg_fixed_seed123_32.out 2>&1 &

MODEL_PATH="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/baseline_seed123_to20k/brain/32dB/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-baseline-pad96-seed123-to20k/network-snapshot-010200.pkl"
EXP_NAME=${EXP_NAME:-baseline_ckpt010200_trainseed123_valseed123_all32}

GPU=2
SEED=123

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
VAL_DIR=${VAL_DIR:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB}

IMAGE_SIZE=${IMAGE_SIZE:-384}
PAD=${PAD:-64}
PSIZE=${PSIZE:-64}
MASK_SELECT=${MASK_SELECT:-7}
ZETA=${ZETA:-3.0}
STEPS=${STEPS:-78}
INNER_LOOPS=${INNER_LOOPS:-10}
#SAMPLE_INDICES=${SAMPLE_INDICES:-3}
SAMPLE_INDICES=${SAMPLE_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}

SAVE_DIR=$RESULT_ROOT/PaDIS-MRI-recon/$EXP_NAME

LOG_DIR="$SAVE_DIR/logs"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_${EXP_NAME}_gpu${GPU}_${TIME_TAG}.log

{
    echo "=================================================="
    echo "PaDIS-MRI full-32 fixed-per-sample-seed evaluation"
    echo "RNG_POLICY=fixed_per_sample"
    echo "SEED=$SEED"
    echo "GPU=$GPU"
    echo "MODEL_PATH=$MODEL_PATH"
    echo "VAL_DIR=$VAL_DIR"
    echo "SAVE_DIR=$SAVE_DIR"
    echo "IMAGE_SIZE=$IMAGE_SIZE"
    echo "PAD=$PAD"
    echo "PSIZE=$PSIZE"
    echo "MASK_SELECT=$MASK_SELECT"
    echo "ZETA=$ZETA"
    echo "STEPS=$STEPS"
    echo "INNER_LOOPS=$INNER_LOOPS"
    echo "TOTAL_UPDATES=$((STEPS * INNER_LOOPS))"
    echo "SAMPLE_INDICES=$SAMPLE_INDICES"
    echo "=================================================="
} | tee "$LOG_FILE"

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# CUDA_VISIBLE_DEVICES exposes the selected physical card as logical cuda:0.
# Therefore eval/run.py must receive --gpus 0, not the physical GPU number.
CUDA_VISIBLE_DEVICES="$GPU" python eval/run.py \
    --run_evaluate \
    --algo padis \
    --model_path "$MODEL_PATH" \
    --val_dir "$VAL_DIR" \
    --image_size "$IMAGE_SIZE" \
    --pad "$PAD" \
    --psize "$PSIZE" \
    --mask_select "$MASK_SELECT" \
    --val_count 1 \
    --sample_indices "$SAMPLE_INDICES" \
    --seed "$SEED" \
    --fixed_seed_per_sample \
    --zeta "$ZETA" \
    --steps "$STEPS" \
    --inner_loops "$INNER_LOOPS" \
    --report_every 1 \
    --gpus 0 \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$LOG_FILE"
