#!/bin/bash
set -e
set -o pipefail


GPU=7



SAMPLE_INDICES=${SAMPLE_INDICES:-3}
VAL_COUNT=${VAL_COUNT:-1}

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI

MODEL_PATH=${MODEL_PATH:-$RESULT_ROOT/PaDIS-MRI-runs/detail_residual_training-runs/brain/32dB/train10k_detail_h48_eta0p15_res0p2_grad0p1_edge0p1_dil1p2p5_bgpu1_s16s32s64_p020305_b2_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-detail-h48-eta0p15-res0p2-grad0p1-edge0p1-dil1p2p5/network-snapshot-005040.pkl}

VAL_DIR=${VAL_DIR:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB_sample3_center320}
EXP_NAME=${EXP_NAME:-detail_base_ckpt005040_s78_sample3_seed123}
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
