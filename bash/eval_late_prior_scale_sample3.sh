#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
BETA=${BETA:-0.5}
LATE_STEPS=${LATE_STEPS:-10}
SAMPLE_INDICES=${SAMPLE_INDICES:-3}

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
VAL_DIR=${VAL_DIR:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB}
MODEL_PATH=${MODEL_PATH:-$RESULT_ROOT/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB/main_overlap_independent_center_lam0p3_active64_bgpu1_s16s32s64_p020305_b2_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-overlap-independent-center-lam0p3-p64/network-snapshot-005040.pkl}

BETA_TAG=$(echo "$BETA" | sed 's/\./p/g')
SAVE_DIR=${SAVE_DIR:-$RESULT_ROOT/PaDIS-MRI-recon/late_prior_scale/sample${SAMPLE_INDICES}_last${LATE_STEPS}_beta${BETA_TAG}}

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/train/padis-mri:$CODE_ROOT:$CODE_ROOT/eval:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python eval/run_late_prior_scale.py \
  --run_evaluate \
  --algo padis \
  --model_path "$MODEL_PATH" \
  --val_dir "$VAL_DIR" \
  --image_size 384 \
  --pad 64 \
  --psize 64 \
  --mask_select 7 \
  --sample_indices "$SAMPLE_INDICES" \
  --seed 123 \
  --zeta 3.0 \
  --steps 78 \
  --inner_loops 10 \
  --late_prior_steps "$LATE_STEPS" \
  --late_prior_scale "$BETA" \
  --save_scale_diagnostics \
  --report_every 1 \
  --gpus 0 \
  --save_dir "$SAVE_DIR"
