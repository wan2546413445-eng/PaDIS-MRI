#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Specify MODEL=/absolute/path/to/network-snapshot-xxxxxx.pkl}"
CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
VAL=${VAL:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}
OUT=${OUT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/JointCT_C96_pilot8}
GPU=${GPU:-0}
MASK=${MASK:-4}
STEPS=${STEPS:-40}
INNER_LOOPS=${INNER_LOOPS:-6}
PATCH_BATCH=${PATCH_BATCH:-49}
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/train/padis-mri:$CODE_ROOT/eval:${PYTHONPATH:-}"
export PADIS_PATCH_BATCH="$PATCH_BATCH"
export CUDA_VISIBLE_DEVICES="$GPU"
python -u eval/run.py \
    --algo=padis --run_evaluate \
    --model_path="$MODEL" --val_dir="$VAL" \
    --image_size=384 --pad=64 --psize=64 --mask_select="$MASK" \
    --sample_indices=0,4,8,13,18,22,26,31 \
    --fixed_seed_per_sample --seed=123 --gpus 0 \
    --steps="$STEPS" --inner_loops="$INNER_LOOPS" --zeta=3 \
    --report_every=1 --save_dir="$OUT"
