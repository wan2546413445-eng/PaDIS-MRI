#!/bin/bash
set -euo pipefail

# Run one-sample LMPC mechanism check using the Fixed CG-OL checkpoint.
#
# Usage:
#   cd /mnt/SSD/wsy/projects/PaDIS-MRI-main
#   chmod +x bash/eval_lmpc_sample3.sh
#   GPU_ID=0 bash bash/eval_lmpc_sample3.sh
#
# OOM fallback:
#   GPU_ID=0 PADIS_PATCH_BATCH=7 PADIS_PATCH_CHECKPOINT=1 \
#     bash bash/eval_lmpc_sample3.sh

GPU_ID=${GPU_ID:-0}
SAMPLE_INDICES=${SAMPLE_INDICES:-3}
LATE_STEPS=${LATE_STEPS:-10}
CONSENSUS_K=${CONSENSUS_K:-4}
OFFSET_MODE=${OFFSET_MODE:-random_complementary}

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
VAL_DIR=${VAL_DIR:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB}

MODEL_PATH=${MODEL_PATH:-$RESULT_ROOT/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB/main_overlap_independent_center_lam0p3_active64_bgpu1_s16s32s64_p020305_b2_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-overlap-independent-center-lam0p3-p64/network-snapshot-005040.pkl}

SAVE_DIR=${SAVE_DIR:-$RESULT_ROOT/PaDIS-MRI-recon/lmpc_fixed_cgol/sample${SAMPLE_INDICES}_last${LATE_STEPS}_k${CONSENSUS_K}_${OFFSET_MODE}}

cd "$CODE_ROOT"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PADIS_PATCH_BATCH=${PADIS_PATCH_BATCH:-49}
export PADIS_PATCH_CHECKPOINT=${PADIS_PATCH_CHECKPOINT:-0}

CUDA_VISIBLE_DEVICES="$GPU_ID" python eval/run_lmpc.py \
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
  --late_consensus_steps "$LATE_STEPS" \
  --late_consensus_k "$CONSENSUS_K" \
  --late_consensus_offset_mode "$OFFSET_MODE" \
  --save_consensus_diagnostics \
  --report_every 1 \
  --gpus 0 \
  --save_dir "$SAVE_DIR"
