#!/usr/bin/env bash
set -euo pipefail
cd "${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"
GPU=${GPU:-0}; KIMG=${KIMG:-1000}; BATCH_GPU=${BATCH_GPU:-4}
ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
DATA=${DATA_DIR:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB}
OUT="$ROOT/PaDIS-MRI-runs/JointCT/CTPaper2D_VP64_${KIMG}kimg"
mkdir -p "$ROOT/results_record/logs" "$OUT"
export PYTHONPATH="$PWD/train/padis-mri:$PWD:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="$GPU" python -u train/padis-mri/train_ct_paper_2d.py \
    --data "$DATA" --outdir "$OUT" --kimg "$KIMG" --batch 64 \
    --batch-gpu "$BATCH_GPU" --lr 2e-5 --ema 0.999 --dropout 0.05 \
    --seed 123 --snap-kimg 100 \
    2>&1 | tee "$ROOT/results_record/logs/train_CTPaper2D_${KIMG}k_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
