#!/bin/bash
set -e
set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ========== Experiment Configuration ==========

GPU=${GPU:-0}

# Network
CBASE=${CBASE:-96}
CHANNEL_MULT=${CHANNEL_MULT:-"2,2,2"}

# OPA
# Paper-definition lambda: pass directly to --lambda-overlap.
# Requires symmetric OPA loss WITHOUT the old internal 0.5 factor.
# Old lambda=2.5 (with 0.5) is equivalent to new lambda=1.25.
OPA_LAMBDA=${OPA_LAMBDA:-1.25}

# Training
KIMG=${KIMG:-5000}
BATCH=${BATCH:-4}
BATCH_GPU=${BATCH_GPU:-1}
SEED=${SEED:-123}
PAD=${PAD:-64}
DROPOUT=${DROPOUT:-0.05}
LR=${LR:-1e-4}
EMA_MIMG=${EMA_MIMG:-0.5}
WORKERS=${WORKERS:-4}

# Patch
PATCH_SIZES=${PATCH_SIZES:-"16,32,64"}
PATCH_PROBS=${PATCH_PROBS:-"0.2,0.3,0.5"}

# Paths
CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}

# Saving cadence: TICK is kimg, SNAP/DUMP are tick counts.
# Defaults save both network snapshot and resumable training state
# approximately every 2000 kimg, plus at the final step.
TICK=${TICK:-5}
SNAP=${SNAP:-200}
DUMP=${DUMP:-400}

# Optional continuation: point to an existing training-state-XXXXXX.pt.
# Its matching network-snapshot-XXXXXX.pkl must exist beside it.
# Keep CBASE, OPA_LAMBDA, and the other training settings unchanged.
RESUME_PATH=${RESUME_PATH:-}
RESUME_ARGS=()
if [[ -n "$RESUME_PATH" ]]; then
    RESUME_ARGS=(--resume="$RESUME_PATH")
fi

# ==============================================

case "$CBASE" in
    128|96|64) ;;
    *)
        echo "CBASE must be one of: 128, 96, 64"
        exit 1
        ;;
esac

DURATION_MIMG=$(awk -v kimg="$KIMG" 'BEGIN { printf "%.12g", kimg / 1000 }')
LAMBDA_TAG=${OPA_LAMBDA//./p}
EXP_NAME=OPA_C${CBASE}_L${LAMBDA_TAG}_${KIMG}kimg

OUTDIR=$RESULT_ROOT/PaDIS-MRI-runs/OPA_compact/$EXP_NAME
LOG_DIR=$RESULT_ROOT/results_record/logs
DATA_DIR=$ROOT_DATA/32dB
LOG_FILE=$LOG_DIR/train_${EXP_NAME}_gpu${GPU}_$(date +"%Y%m%d_%H%M%S").log

mkdir -p "$OUTDIR" "$LOG_DIR"
cd "$CODE_ROOT"

{
    echo "=================================================="
    echo "PaDIS-MRI OPA Width and Lambda Ablation"
    echo "GPU=$GPU"
    echo "CBASE=$CBASE"
    echo "CHANNEL_MULT=$CHANNEL_MULT"
    echo "OPA_LAMBDA=$OPA_LAMBDA"
    echo "KIMG=$KIMG"
    echo "BATCH=$BATCH"
    echo "BATCH_GPU=$BATCH_GPU"
    echo "SEED=$SEED"
    echo "PAD=$PAD"
    echo "DROPOUT=$DROPOUT"
    echo "PATCH_SIZES=$PATCH_SIZES"
    echo "PATCH_PROBS=$PATCH_PROBS"
    echo "OPA_NOISE_MODE=independent"
    echo "OPA_PAIR_GEOMETRY=identical"
    echo "OPA_ALIGNMENT=symmetric"
    echo "OPA_SPATIAL_WEIGHT=uniform"
    echo "OUTDIR=$OUTDIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "TICK=$TICK SNAP=$SNAP DUMP=$DUMP"
    echo "RESUME_PATH=${RESUME_PATH:-none}"
    echo "=================================================="
} | tee "$LOG_FILE"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export WANDB_MODE=disabled

CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone \
    --nproc_per_node=1 \
    train/padis-mri/train_overlap.py \
    --outdir="$OUTDIR" \
    --data="$DATA_DIR" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=pedm \
    --cbase="$CBASE" \
    --cres="$CHANNEL_MULT" \
    --overlap-mode=independent \
    --overlap-alignment=symmetric \
    --overlap-geometry=identical \
    --overlap-spatial-weight=uniform \
    --lambda-overlap="$OPA_LAMBDA" \
    --batch="$BATCH" \
    --batch-gpu="$BATCH_GPU" \
    --lr="$LR" \
    --ema="$EMA_MIMG" \
    --dropout="$DROPOUT" \
    --augment=0 \
    --real_p=0.5 \
    --padding=1 \
    --pad_width="$PAD" \
    --patch-list="$PATCH_SIZES" \
    --patch-probs="$PATCH_PROBS" \
    --duration="$DURATION_MIMG" \
    --tick="$TICK" \
    --snap="$SNAP" \
    --dump="$DUMP" \
    --workers="$WORKERS" \
    --seed="$SEED" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"
