#!/bin/bash
set -e
set -o pipefail

GPU=7
SAMPLE_INDEX=1

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI

MODEL_PATH=$RESULT_ROOT/PaDIS-MRI-runs/checkpoints/00003-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32/network-snapshot-014994.pkl
VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB

EXP_NAME=baseline_ckpt014994_shifted_uniform_s78_f520_shift32_sample${SAMPLE_INDEX}
SAVE_DIR=$RESULT_ROOT/PaDIS-MRI-recon/$EXP_NAME
LOG_DIR=$RESULT_ROOT/results_record/logs

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_${EXP_NAME}_gpu${GPU}_${TIME_TAG}.log

echo "==================================================" | tee "$LOG_FILE"
echo "PaDIS-MRI Baseline Checkpoint + Shifted Uniform Fusion Eval" | tee -a "$LOG_FILE"
echo "EXP_NAME=$EXP_NAME" | tee -a "$LOG_FILE"
echo "GPU=$GPU" | tee -a "$LOG_FILE"
echo "SAMPLE_INDEX=$SAMPLE_INDEX" | tee -a "$LOG_FILE"
echo "MODEL_PATH=$MODEL_PATH" | tee -a "$LOG_FILE"
echo "VAL_DIR=$VAL_DIR" | tee -a "$LOG_FILE"
echo "SAVE_DIR=$SAVE_DIR" | tee -a "$LOG_FILE"
echo "STEPS=78" | tee -a "$LOG_FILE"
echo "INNER_LOOPS=10" | tee -a "$LOG_FILE"
echo "TOTAL_UPDATES=780" | tee -a "$LOG_FILE"
echo "FUSION_START_UPDATE=520" | tee -a "$LOG_FILE"
echo "SHIFT=32" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

cd "$CODE_ROOT"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$GPU python eval/run_shifted_fusion.py \
    --model_path "$MODEL_PATH" \
    --val_dir "$VAL_DIR" \
    --image_size 384 \
    --pad 64 \
    --psize 64 \
    --mask_select 7 \
    --sample_indices "$SAMPLE_INDEX" \
    --seed 123 \
    --zeta 3.0 \
    --steps 78 \
    --inner_loops 10 \
    --fusion_start_update 520 \
    --shift 32 \
    --report_every 1 \
    --gpus 0 \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$LOG_FILE"
