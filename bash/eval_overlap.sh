#!/bin/bash
set -e
set -o pipefail


GPU=1



SAMPLE_INDICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI

MODEL_PATH=${MODEL_PATH:-$RESULT_ROOT/PaDIS-MRI-runs/overlap_same_training-runs/brain/32dB/main_overlap_same_lam001_s16s32s64_p020305_b2_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-overlap-same-lam0p01/network-snapshot-005040.pkl}

VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB
EXP_NAME=${EXP_NAME:-overlap_independent_center_lam0p3_ckpt005040_s78_sample1_seed123}
SAVE_DIR=$RESULT_ROOT/PaDIS-MRI-recon/$EXP_NAME
LOG_DIR=$RESULT_ROOT/results_record/logs

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_${EXP_NAME}_gpu${GPU}_${TIME_TAG}.log

echo "==================================================" | tee "$LOG_FILE"
echo "PaDIS-MRI Different-noise Overlap Training Eval" | tee -a "$LOG_FILE"
echo "GPU=$GPU" | tee -a "$LOG_FILE"
echo "SAMPLE_INDICES=$SAMPLE_INDICES" | tee -a "$LOG_FILE"
echo "MODEL_PATH=$MODEL_PATH" | tee -a "$LOG_FILE"
echo "VAL_DIR=$VAL_DIR" | tee -a "$LOG_FILE"
echo "SAVE_DIR=$SAVE_DIR" | tee -a "$LOG_FILE"
echo "STEPS=78" | tee -a "$LOG_FILE"
echo "INNER_LOOPS=10" | tee -a "$LOG_FILE"
echo "TOTAL_UPDATES=780" | tee -a "$LOG_FILE"
echo "SAMPLER=Original PaDIS single-partition sampler" | tee -a "$LOG_FILE"
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
    --sample_indices "$SAMPLE_INDICES" \
    --seed 123 \
    --zeta 3.0 \
    --steps 78 \
    --inner_loops 10 \
    --report_every 1 \
    --gpus 0 \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$LOG_FILE"
