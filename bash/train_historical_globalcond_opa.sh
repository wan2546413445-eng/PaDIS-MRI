#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
if [[ "${1:-}" == profile ]]; then
 for size in 16 32 64; do GC5_PROFILE=1 PROFILE_SIZE=$size bash "$0" profile-one || exit $?; done
 exit 0
fi
transfer=()
if [[ -n "${OPA_CHECKPOINT:-}" ]]; then transfer=(--transfer="$OPA_CHECKPOINT"); fi
extra=()
if [[ -n "${CBASE:-}" ]]; then extra+=(--cbase="$CBASE"); fi
case "${1:-main}" in
 profile-one) duration=0.001; sizes=$PROFILE_SIZE; probs=1; mode=profile_$sizes ;;
 debug) duration=0.001; sizes=64; probs=1; mode=debug ;;
 main) duration=${DURATION:-5}; sizes=16,32,64; probs=0.2,0.3,0.5; mode=main ;;
 *) exit 2 ;;
esac
export PYTHONPATH="$PWD:$PWD/train/padis-mri:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=${GPU:-0} torchrun --standalone --nproc_per_node=1 train/padis-mri/train_globalcond_opa.py \
 --data="${DATA_DIR:-/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB}" \
 --outdir="${OUTDIR:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/historical_globalcond_opa/$mode}" \
 --arch=ddpmpp --precond=pedm --pad_width=64 --patch-list="$sizes" --patch-probs="$probs" \
 --batch=4 --batch-gpu=1 --ema=0.5 --lr=0.0001 --dropout=0.05 --augment=0 --real_p=0.5 --seed=123 \
 --global_context_mode=magnitude_lowres --global_context_from=noisy --global_context_size=96 \
 --lambda-overlap="${LAMBDA_OVERLAP:-1.25}" --duration="$duration" \
 --tick="${TICK:-50}" --snap="${SNAP:-50}" --dump="${DUMP:-500}" \
 "${extra[@]}" "${transfer[@]}"
