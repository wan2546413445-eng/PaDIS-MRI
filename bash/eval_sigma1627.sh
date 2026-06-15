#!/bin/bash
set -e
set -o pipefail

GPU=3

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI

MODEL_PATH=$RESULT_ROOT/PaDIS-MRI-runs/baseline_training-runs_sigma1627_full/brain/32dB/main_baseline_sigma1627_full_s16s32s64_p020305_b2_seed123/00001-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32/network-snapshot-010080.pkl

VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB

SAVE_DIR=$RESULT_ROOT/PaDIS-MRI-recon/baseline_sigma1627_full_ckpt010080_s78_all32
LOG_DIR=$RESULT_ROOT/results_record/logs

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_baseline_sigma1627_full_ckpt010080_s78_all32_gpu${GPU}_${TIME_TAG}.log

echo "==================================================" | tee "$LOG_FILE"
echo "Original PaDIS-MRI Eval: sigma_data=0.1627" | tee -a "$LOG_FILE"
echo "GPU=$GPU" | tee -a "$LOG_FILE"
echo "MODEL_PATH=$MODEL_PATH" | tee -a "$LOG_FILE"
echo "VAL_DIR=$VAL_DIR" | tee -a "$LOG_FILE"
echo "SAVE_DIR=$SAVE_DIR" | tee -a "$LOG_FILE"
echo "STEPS=78" | tee -a "$LOG_FILE"
echo "VAL_COUNT=32" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

cd "$CODE_ROOT"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$GPU python eval/run.py \
    --run_evaluate \
    --algo padis \
    --model_path "$MODEL_PATH" \
    --val_dir "$VAL_DIR" \
    --image_size 384 \
    --pad 64 \
    --psize 64 \
    --mask_select 7 \
    --val_count 32 \
    --seed 123 \
    --zeta 3.0 \
    --steps 78 \
    --report_every 1 \
    --gpus 0 \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$LOG_FILE"