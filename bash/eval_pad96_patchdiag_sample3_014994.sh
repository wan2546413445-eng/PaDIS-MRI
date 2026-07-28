#!/bin/bash
set -e
set -o pipefail

GPU=7

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/00_CURRENT_MAIN/mask_sensitivity/strict_sample3_compare

MODEL_PATH=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/baseline_seed123_to20k/brain/32dB/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-baseline-pad96-seed123-to20k/network-snapshot-010200.pkl

RANDOM_VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB_sample3_only

EQUI_VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp_sample3_equispaced_strict/32dB

mkdir -p "$RESULT_ROOT"

cd "$CODE_ROOT"

export PYTHONPATH=$CODE_ROOT:$CODE_ROOT/train/padis-mri:$CODE_ROOT/eval:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


run_eval () {
    NAME=$1
    VAL_DIR=$2

    SAVE_DIR="$RESULT_ROOT/$NAME"
    LOG_DIR="$SAVE_DIR/logs"

    mkdir -p "$LOG_DIR"

    TIME_TAG=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/eval_${NAME}_gpu${GPU}_${TIME_TAG}.log"

    echo "==================================================" | tee "$LOG_FILE"
    echo "Strict mask sensitivity evaluation" | tee -a "$LOG_FILE"
    echo "NAME=$NAME" | tee -a "$LOG_FILE"
    echo "GPU=$GPU" | tee -a "$LOG_FILE"
    echo "MODEL_PATH=$MODEL_PATH" | tee -a "$LOG_FILE"
    echo "VAL_DIR=$VAL_DIR" | tee -a "$LOG_FILE"
    echo "SAVE_DIR=$SAVE_DIR" | tee -a "$LOG_FILE"
    echo "SAMPLE=3" | tee -a "$LOG_FILE"
    echo "MASK_SELECT=7" | tee -a "$LOG_FILE"
    echo "STEPS=78" | tee -a "$LOG_FILE"
    echo "INNER_LOOPS=10" | tee -a "$LOG_FILE"
    echo "TOTAL_UPDATES=780" | tee -a "$LOG_FILE"
    echo "==================================================" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU python eval/run.py \
      --run_evaluate \
      --algo padis \
      --model_path "$MODEL_PATH" \
      --val_dir "$VAL_DIR" \
      --image_size 384 \
      --pad 64 \
      --psize 64 \
      --mask_select 7 \
      --val_count 1 \
      --sample_indices 0 \
      --seed 123 \
      --fixed_seed_per_sample \
      --zeta 3.0 \
      --steps 78 \
      --inner_loops 10 \
      --report_every 1 \
      --save_dir "$SAVE_DIR" \
      --gpus 0 \
      2>&1 | tee -a "$LOG_FILE"
}


run_eval \
  baseline_ckpt014994_sample3_random_mask7_seed123 \
  "$RANDOM_VAL_DIR"

run_eval \
  baseline_ckpt014994_sample3_equispaced_strict_mask7_seed123 \
  "$EQUI_VAL_DIR"