#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the original five-channel EMA snapshot}"
: "${VAL_DIR:?Set VAL_DIR to the validation directory}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
python eval_globalcond/run_globalcond.py \
  --model_path "$MODEL_PATH" --val_dir "$VAL_DIR" --gpus 0 \
  --global_context_size 96 --steps "${STEPS:-78}" --inner_loops "${INNER_LOOPS:-10}" \
  --mask_select "${MASK_SELECT:-7}" --sample_indices "${SAMPLE_INDICES:-}" \
  --save_dir "${SAVE_DIR:-results/globalcond_rerun_$(date +%Y%m%d_%H%M%S)}" \
  --run_evaluate "$@"
