#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
GPU=${GPU:-3}; CBASE=${CBASE:-96}; BATCH_GPU=${BATCH_GPU:-1}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
ROOT_DATA=${ROOT_DATA:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}
DATA_DIR=${DATA_DIR:-$ROOT_DATA/32dB}
LAMBDA_OVERLAP=${LAMBDA_OVERLAP:-1.25}
OC_PAIR_GEOMETRY=${OC_PAIR_GEOMETRY:-identical}
OC_SPATIAL_WEIGHT_MODE=${OC_SPATIAL_WEIGHT_MODE:-uniform}
OC_ALIGNMENT_MODE=${OC_ALIGNMENT_MODE:-symmetric}
SEED=${SEED:-123}; MODE=${1:-overnight}
case "$MODE" in
  debug) DURATION=${DURATION:-0.001}; TICK=1; SNAP=${SNAP:-1}; DUMP=1; PATCH_SIZES=64; PATCH_PROBS=1.0;;
  overnight|main) DURATION=${DURATION:-1}; TICK=5; SNAP=${SNAP:-20}; DUMP=${DUMP:-1000}; PATCH_SIZES=16,32,64; PATCH_PROBS=0.2,0.3,0.5;;
  *) echo "Usage: $0 debug|overnight|main" >&2; exit 1;;
esac
RUN="NGS_v2_${MODE}_C${CBASE}_OPA${LAMBDA_OVERLAP//./p}_${OC_PAIR_GEOMETRY}_b4_bgpu${BATCH_GPU}_seed${SEED}_dur${DURATION}"
OUTDIR="$RESULT_ROOT/PaDIS-MRI-runs/ngs_training-runs/brain/32dB/$RUN"
LOGDIR="$RESULT_ROOT/results_record/logs"
mkdir -p "$OUTDIR" "$LOGDIR"
export PYTHONPATH="$ROOT/train/padis-mri:$ROOT:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PADIS_PATCH_BATCH=${PADIS_PATCH_BATCH:-4}
export CUDA_VISIBLE_DEVICES="$GPU"
[[ -f "$DATA_DIR/noisy.pt" ]] || { echo "Missing dataset: $DATA_DIR/noisy.pt" >&2; exit 1; }
# Fresh training for v2 by default. Do not resume/transfer a v1 full snapshot:
# the projection dimensions have changed.
EXTRA=()
if [[ -n "${TRANSFER:-}" ]]; then EXTRA+=(--transfer="$TRANSFER"); fi
if [[ -n "${RESUME_PATH:-}" ]]; then EXTRA+=(--resume="$RESUME_PATH"); fi
[[ -z "${TRANSFER:-}" || -z "${RESUME_PATH:-}" ]] || { echo 'Choose TRANSFER or RESUME_PATH, not both'; exit 1; }
echo "NGS-v2 code: $ROOT; data: $DATA_DIR; output: $OUTDIR"
torchrun --standalone --nproc_per_node=1 train/padis-mri/train_ngs.py \
  --outdir="$OUTDIR" --data="$DATA_DIR" --cond=0 --arch=ddpmpp --backbone=songunet --precond=pedm \
  --overlap-mode=independent --overlap-alignment="$OC_ALIGNMENT_MODE" \
  --overlap-geometry="$OC_PAIR_GEOMETRY" --overlap-spatial-weight="$OC_SPATIAL_WEIGHT_MODE" \
  --lambda-overlap="$LAMBDA_OVERLAP" --cbase="$CBASE" --cres=2,2,2 \
  --batch=4 --batch-gpu="$BATCH_GPU" --lr=1e-4 --dropout=0.05 --augment=0 \
  --real_p=0.5 --padding=1 --pad_width=64 --patch-list="$PATCH_SIZES" --patch-probs="$PATCH_PROBS" \
  --duration="$DURATION" --tick="$TICK" --snap="$SNAP" --dump="$DUMP" --workers=4 --seed="$SEED" \
  "${EXTRA[@]}" 2>&1 | tee "$LOGDIR/train_${RUN}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
