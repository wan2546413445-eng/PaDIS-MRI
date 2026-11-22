#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
GPU=${GPU:-0}
MASK=${MASK:-4}
SAMPLES=${SAMPLES:-0}
TRAJECTORY_BATCH_SIZE=${TRAJECTORY_BATCH_SIZE:-1}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-10}
SEED_START=${SEED_START:-123}
GRID_MODE=${GRID_MODE:-cycle_k}
MODEL=${MODEL:-""}
# The current source commit has no OPA1.25/10k model path. Never substitute E0,
# OPA2/4 or a 5k snapshot. MODEL must be supplied until that path is synchronized.
if [[ -z "$MODEL" ]]; then
    echo 'OPA1.25 10000kimg checkpoint path is absent from the current repository Bash. Set MODEL to the exact server checkpoint.' >&2
    exit 1
fi
VAL=${VAL:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}
case "$MASK" in 4|R4|r4) MASK=4;; 7|R7|r7) MASK=7;; *) echo 'MASK must be 4 or 7' >&2; exit 1;; esac
if [[ "${BENCHMARK_MODE:-formal}" == quick ]]; then
    N=2; K=2; BUDGET=quick_N2K2
else
    N=60; K=4; BUDGET=N60K4
fi
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/Block_pULA_OPA1p25_10k_${BUDGET}_S${NUM_TRAJECTORIES}_sigma10_0p01_T1FLAIR_R${MASK}/tb${TRAJECTORY_BATCH_SIZE}_${STAMP}}
mkdir -p "$OUT_ROOT"
OUT_ROOT=$(cd "$OUT_ROOT" && pwd)
LOG="$OUT_ROOT/run.log"
echo "Absolute log: $LOG"
echo "Checkpoint: $MODEL | validation: $VAL | mask: R$MASK | samples: $SAMPLES"
echo "N=$N K=$K sigma=10->0.01 gamma=0.5 CG=10/1e-6 S=$NUM_TRAJECTORIES batch=$TRAJECTORY_BATCH_SIZE seeds=$SEED_START..$((SEED_START+NUM_TRAJECTORIES-1))"
echo "Partition strategy: $GRID_MODE | partition seeds: $SEED_START..$((SEED_START+NUM_TRAJECTORIES-1))"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 MPLBACKEND=Agg python -u eval_pula/run.py \
    --model_path "$MODEL" --val_dir "$VAL" --out_root "$OUT_ROOT" \
    --sample_indices "$SAMPLES" --mask_select "$MASK" --score_mode patch \
    --grid_mode "$GRID_MODE" --grid_seed "$SEED_START" --seed "$SEED_START" --pula_convention padis_real \
    --sigma_max 10 --sigma_min 0.01 --num_noise_levels "$N" --langevin_steps "$K" \
    --gamma 0.5 --cg_iters 10 --cg_rtol 1e-6 \
    --num_trajectories "$NUM_TRAJECTORIES" --trajectory_batch_size "$TRAJECTORY_BATCH_SIZE" \
    --trajectory_seed_start "$SEED_START" --aggregate_counts 1,2,3,5,8,10 \
    --save_each_trajectory true --measure_cuda_memory true 2>&1 | tee -a "$LOG"
