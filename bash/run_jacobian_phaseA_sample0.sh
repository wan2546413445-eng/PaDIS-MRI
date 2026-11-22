#!/usr/bin/env bash
# Stage A ONLY. Does not execute anything unless --dry-run or --run is passed.
set -euo pipefail
ROOT=${PADIS_REPO_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
MODE=${2:-on}
if [[ $# -lt 1 || ( "$1" != "--dry-run" && "$1" != "--run" ) || ( "$MODE" != "on" && "$MODE" != "off" ) ]]; then
    echo 'Usage: E0_10K_MODEL=/absolute/checkpoint.pkl PADIS_T1FLAIR_VAL_DIR=/absolute/val_dir bash bash/run_jacobian_phaseA_sample0.sh --dry-run|--run on|off'
    echo 'No GPU runs occur without --run. GPU 3 is exposed as cuda:0 inside Python.'
    exit 2
fi
: "${E0_10K_MODEL:?Please set E0_10K_MODEL to the VERIFIED E0 10k checkpoint absolute path}"
: "${PADIS_T1FLAIR_VAL_DIR:?Please set PADIS_T1FLAIR_VAL_DIR to the actual T1-FLAIR validation folder}"
OUT_BASE=${JACOBIAN_PHASEA_OUT_BASE:-"$ROOT/jacobian_phaseA_outputs"}
OUT_ROOT="$OUT_BASE/${MODE}_seed123_sample0"
if [[ "$1" == "--run" && -e "$OUT_ROOT" ]]; then
    echo "Refusing an existing output root: $OUT_ROOT" >&2
    exit 3
fi
ARGS=(
  --model_path "$E0_10K_MODEL"
  --val_dir "$PADIS_T1FLAIR_VAL_DIR"
  --out_root "$OUT_ROOT"
  --sample_indices 0
  --score_mode patch
  --grid_mode random_step
  --pula_convention padis_real
  --pula_variant baseline
  --grid_seed 123
  --seed 123
  --num_noise_levels 48
  --langevin_steps 4
  --dps_tail_steps 24
  --dps_inner_loops 10
)
if [[ "$MODE" == "on" ]]; then
  ARGS+=(--diagnose_dps_jacobian --jacobian_levels 0,11,23 --jacobian_inner_steps 0)
fi
if [[ "$1" == "--dry-run" ]]; then
  (cd "$ROOT" && python eval_pula/run.py "${ARGS[@]}" --dry_run)
else
  export CUDA_VISIBLE_DEVICES=3
  (cd "$ROOT" && python eval_pula/run.py "${ARGS[@]}")
fi
