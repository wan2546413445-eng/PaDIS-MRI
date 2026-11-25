#!/usr/bin/env bash
set -euo pipefail
cd "${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"
GPU=${GPU:-5}; KIMG=${KIMG:-500}; BATCH_GPU=${BATCH_GPU:-1}
CBASE=${CBASE:-64}; CRES=${CRES:-1,2,4,4}; NUM_BLOCKS=${NUM_BLOCKS:-2}; ATTN_RESOLUTIONS=${ATTN_RESOLUTIONS:-8}; OPA_LAMBDA=${OPA_LAMBDA:-1.25}
NOISE_EMBEDDING=${NOISE_EMBEDDING:-fourier}; INIT_MODE=${INIT_MODE:-edm}
LR=${LR:-1e-4}; LR_RAMPUP_KIMG=${LR_RAMPUP_KIMG:-2644.628099173554}; EMA_KIMG=${EMA_KIMG:-500}
SEED=${SEED:-123}; TICK=${TICK:-5}; SNAP=${SNAP:-20}; DUMP=${DUMP:-20}
M_TAG=${CRES//,/-}; A_TAG=${ATTN_RESOLUTIONS//,/-}; LAM_TAG=${OPA_LAMBDA//./p}
TAG="Adapt_C${CBASE}_M${M_TAG}_B${NUM_BLOCKS}_A${A_TAG}_EDM_OPA${LAM_TAG}_p64_${NOISE_EMBEDDING}_${INIT_MODE}_lr${LR}_wu${LR_RAMPUP_KIMG}_ema${EMA_KIMG}_s${SEED}_${KIMG}kimg"
ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
DATA=${DATA_DIR:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB}
OUT="$ROOT/PaDIS-MRI-runs/JointCT/CTPaper2D_${TAG}"
mkdir -p "$ROOT/results_record/logs" "$OUT"
export PYTHONPATH="$PWD:$PWD/train/padis-mri:$PWD/eval:${PYTHONPATH:-}"
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES="$GPU" torchrun --standalone --nproc_per_node=1 train/padis-mri/train_ct_paper_opa.py \
    --data "$DATA" --outdir "$OUT" --kimg "$KIMG" --batch-gpu "$BATCH_GPU" --seed "$SEED" --cbase "$CBASE" --cres "$CRES" --num-blocks "$NUM_BLOCKS" --attn-resolutions "$ATTN_RESOLUTIONS" --opa-lambda "$OPA_LAMBDA" \
    --noise-embedding "$NOISE_EMBEDDING" --init-mode "$INIT_MODE" --lr "$LR" --lr-rampup-kimg "$LR_RAMPUP_KIMG" --ema-kimg "$EMA_KIMG" --tick "$TICK" --snap "$SNAP" --dump "$DUMP" \
    2>&1 | tee "$ROOT/results_record/logs/train_CTPaper_${TAG}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
