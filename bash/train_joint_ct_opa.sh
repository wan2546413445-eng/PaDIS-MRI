#!/usr/bin/env bash
set -euo pipefail
GPU=${GPU:-0}
CBASE=${CBASE:-96}
KIMG=${KIMG:-1000}
SEED=${SEED:-123}
CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
DATA_ROOT=${DATA_ROOT:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200}
RUN_ROOT=${RUN_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/JointCT}
OUTDIR=${OUTDIR:-$RUN_ROOT/C${CBASE}_OPA125_${KIMG}kimg}
if [[ ! -f "$CODE_ROOT/train/padis-mri/training/joint_context.py" ]]; then
    echo 'ERROR: install tools/install_joint_ct.py first' >&2; exit 2
fi
if [[ ! -e "$DATA_ROOT/32dB/noisy.pt" ]]; then
    echo "ERROR: missing data: $DATA_ROOT/32dB/noisy.pt (override DATA_ROOT)" >&2; exit 2
fi
mkdir -p "$OUTDIR"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export WANDB_MODE=disabled
# Do not set a fragmentation workaround as a substitute for free GPU memory.
CUDA_VISIBLE_DEVICES="$GPU" torchrun --standalone --nproc_per_node=1 \
    train/padis-mri/train_overlap.py \
    --outdir="$OUTDIR" --data="$DATA_ROOT/32dB" \
    --cond=0 --arch=ddpmpp --backbone=songunet --precond=pedm \
    --joint-ct-context=True --cbase="$CBASE" --cres=2,2,2 \
    --overlap-mode=independent --overlap-alignment=symmetric \
    --overlap-geometry=identical --overlap-spatial-weight=uniform \
    --lambda-overlap=1.25 --batch=4 --batch-gpu=1 \
    --lr=1e-4 --ema=0.5 --dropout=0.05 --augment=0 \
    --real_p=0.5 --padding=1 --pad_width=64 \
    --patch-list=16,32,64 --patch-probs=0.2,0.3,0.5 \
    --duration="$(awk -v k="$KIMG" 'BEGIN {printf "%.12g", k/1000}')" \
    --tick=5 --snap=200 --dump=400 --workers=4 --seed="$SEED"
