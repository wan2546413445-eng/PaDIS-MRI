#!/bin/bash
set -e
set -o pipefail


GPU=3



SAMPLE_INDICES=${SAMPLE_INDICES:-3}
VAL_COUNT=${VAL_COUNT:-1}

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI

MODEL_PATH=${MODEL_PATH:-$RESULT_ROOT/PaDIS-MRI-runs/agg_overlap_simplified_training-runs/brain/32dB/main_agg_overlap_rho0p65_gamma1p0_oc0p3_pad64_seed123/00001-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-agg-overlap-rho0p65-gamma1p0-oc0p3-p64/network-snapshot-010006.pkl}

VAL_DIR=${VAL_DIR:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB}
EXP_NAME=${EXP_NAME:-AGG_OC_simply_ckpt010006_s78_10_sample3_seed123}
SAVE_DIR=$RESULT_ROOT/PaDIS-MRI-recon/03_MODULE_TRIALS/AGG/$EXP_NAME
LOG_DIR=$RESULT_ROOT/PaDIS-MRI-recon/03_MODULE_TRIALS/AGG/$EXP_NAME/logs

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_${EXP_NAME}_gpu${GPU}_${TIME_TAG}.log

echo "==================================================" | tee "$LOG_FILE"
echo "PaDIS-MRI AGG_OC Training Eval" | tee -a "$LOG_FILE"
echo "GPU=$GPU" | tee -a "$LOG_FILE"
echo "SAMPLE_INDICES=$SAMPLE_INDICES" | tee -a "$LOG_FILE"
echo "MODEL_PATH=$MODEL_PATH" | tee -a "$LOG_FILE"
echo "VAL_DIR=$VAL_DIR" | tee -a "$LOG_FILE"
echo "SAVE_DIR=$SAVE_DIR" | tee -a "$LOG_FILE"
echo "STEPS=78" | tee -a "$LOG_FILE"
echo "INNER_LOOPS=10" | tee -a "$LOG_FILE"
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
