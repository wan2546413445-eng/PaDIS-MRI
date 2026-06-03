#!/bin/bash
set -e
set -o pipefail

GPU=0

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI

MODEL_PATH=$RESULT_ROOT/PaDIS-MRI-runs/training-runs-cross/main_cross_patch_s16s32s64_k8_cbase96_b4_fp32/00003-cross_patch_s16s32s64_k8_l3g4_d2h4ffn4_cbase96_b6_fp32-aapm_3-uncond-ddpmpp-pedm-gpus1/network-snapshot-005000.pkl

VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB

SAVE_DIR=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/cross_patch_k8_5000k_s78_i10_sample1
LOG_DIR=$RESULT_ROOT/results_record/logs
mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_cross_patch_k8_5000k_s78_i10_sample1_gpu${GPU}_${TIME_TAG}.log

echo "==================================================" | tee $LOG_FILE
echo "Cross-Patch PaDIS-MRI Eval Debug" | tee -a $LOG_FILE
echo "GPU=$GPU" | tee -a $LOG_FILE
echo "MODEL_PATH=$MODEL_PATH" | tee -a $LOG_FILE
echo "VAL_DIR=$VAL_DIR" | tee -a $LOG_FILE
echo "SAVE_DIR=$SAVE_DIR" | tee -a $LOG_FILE
echo "LOG_FILE=$LOG_FILE" | tee -a $LOG_FILE
echo "==================================================" | tee -a $LOG_FILE

cd $CODE_ROOT
export PYTHONPATH=$CODE_ROOT/train/padis-mri:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$GPU python eval/cross_run.py \
  --run_evaluate \
  --algo cross_padis \
  --model_path $MODEL_PATH \
  --val_dir $VAL_DIR \
  --image_size 384 \
  --pad 96 \
  --psize 64 \
  --mask_select 7 \
  --val_count 1 \
  --sample_indices 1 \
  --zeta 3.0 \
  --steps 78 \
  --inner_loops 10 \
  --cp_k 8 \
  --cp_local_k 3 \
  --cp_global_k 4 \
  --cp_eval_batch_size 64 \
  --memory_safe_eval \
  --save_dir $SAVE_DIR \
  2>&1 | tee -a $LOG_FILE