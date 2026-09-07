#!/bin/bash
set -e
set -o pipefail

GPU=3

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI

MODEL_PATH=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/agg_overlap_simplified_training-runs/brain/32dB/main_agg_overlap_rho0p65_gamma1p0_oc0p3_pad64_seed123/00001-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-agg-overlap-rho0p65-gamma1p0-oc0p3-p64/network-snapshot-010006.pkl

VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB
#VAL_DIR=/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB
#VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t2/32dB
EXP_NAME=AGG_oc_ckpt010006_T1_flair_trainseed123_valseed123_all32_s78
SAVE_DIR="$RESULT_ROOT/PaDIS-MRI-recon/T1_FLAIR/03_MODULE_TRIALS/AGG/$EXP_NAME"
LOG_DIR="$SAVE_DIR/logs"

mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/eval_${EXP_NAME}_gpu${GPU}_${TIME_TAG}.log"

echo "==================================================" | tee $LOG_FILE
echo "PaDIS-MRI AGG OC  Eval all" | tee -a $LOG_FILE
echo "GPU=$GPU" | tee -a $LOG_FILE
echo "MODEL_PATH=$MODEL_PATH" | tee -a $LOG_FILE
echo "VAL_DIR=$VAL_DIR" | tee -a $LOG_FILE
echo "SAVE_DIR=$SAVE_DIR" | tee -a $LOG_FILE
echo "==================================================" | tee -a $LOG_FILE


cd $CODE_ROOT


# 关键：加入training模块路径
export PYTHONPATH=$CODE_ROOT:$CODE_ROOT/train/padis-mri:$PYTHONPATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


CUDA_VISIBLE_DEVICES=$GPU python eval/run.py \
  --run_evaluate \
  --algo padis \
  --model_path $MODEL_PATH \
  --val_dir $VAL_DIR \
  --image_size 384 \
  --pad 64 \
  --psize 64 \
  --mask_select 7 \
  --val_count 32 \
  --sample_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --seed 123 \
  --fixed_seed_per_sample \
  --zeta 3.0 \
  --steps 78 \
  --inner_loops 10 \
  --report_every 1 \
  --save_dir $SAVE_DIR \
  --gpus 0 \
  2>&1 | tee -a $LOG_FILE