#!/bin/bash
set -e
set -o pipefail

GPU=4

CODE_ROOT=/mnt/SSD/wsy/projects/PaDIS-MRI-main
RESULT_ROOT=/mnt/SSD2/wsy/PaDIS-MRI

# 改成你实际训练得到的 snapshot。
MODEL_PATH=$RESULT_ROOT/PaDIS-MRI-runs/training-runs-agg-overlap/main_agg_overlap_fixed64_ov16_lam005_cbase128_b64_fp32/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch64-fp32-agg_overlap_fixed64_ov16_lam005_cbase128_b64_fp32-agg_overlap-ov16-lam0.05/network-snapshot-005000.pkl

VAL_DIR=/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB

SAVE_DIR=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/agg_overlap_fixed64_lam005_5000k_s78_i10_val18
LOG_DIR=$RESULT_ROOT/results_record/logs

mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_agg_overlap_fixed64_lam005_5000k_s78_i10_val18_gpu${GPU}_${TIME_TAG}.log

echo "==================================================" | tee $LOG_FILE
echo "Aggregation-Aware Overlap PaDIS-MRI Eval" | tee -a $LOG_FILE
echo "GPU=$GPU" | tee -a $LOG_FILE
echo "MODEL_PATH=$MODEL_PATH" | tee -a $LOG_FILE
echo "VAL_DIR=$VAL_DIR" | tee -a $LOG_FILE
echo "SAVE_DIR=$SAVE_DIR" | tee -a $LOG_FILE
echo "LOG_FILE=$LOG_FILE" | tee -a $LOG_FILE
echo "==================================================" | tee -a $LOG_FILE

cd $CODE_ROOT
export PYTHONPATH=$CODE_ROOT/train/padis-mri:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$GPU python eval/run.py \
  --run_evaluate \
  --algo padis \
  --model_path $MODEL_PATH \
  --val_dir $VAL_DIR \
  --image_size 384 \
  --pad 96 \
  --psize 64 \
  --mask_select 7 \
  --val_count 1 \
  --sample_indices 18 \
  --zeta 3.0 \
  --steps 78 \
  --inner_loops 10 \
  --save_dir $SAVE_DIR \
  2>&1 | tee -a $LOG_FILE