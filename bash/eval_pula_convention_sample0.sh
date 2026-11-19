#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg

VAL_T1FLAIR="${VAL_T1FLAIR:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}"
OUT_ROOT="${OUT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/pula_convention_sample0}"

# Reuse the E0 training directory from eval_E0_T1FLAIR32.sh and select its 10k snapshot.
E0_RUN='/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/消融实验/E0_agg_simplified_single_training-runs/brain/32dB/agg_single_rho0_gamma0_pad64_seed123_train'
if [[ -z "${MODEL_E0:-}" ]]; then
    shopt -s nullglob
    checkpoints=("$E0_RUN"/*/network-snapshot-010000.pkl)
    if (( ${#checkpoints[@]} != 1 )); then
        echo "Expected one E0 10k checkpoint under $E0_RUN, found ${#checkpoints[@]}. Set MODEL_E0 explicitly." >&2
        exit 1
    fi
    MODEL_E0="${checkpoints[0]}"
fi

COMMON=(
    --model_path "$MODEL_E0" --val_dir "$VAL_T1FLAIR"
    --sample_indices 0 --mask_select 4 --image_size 384 --pad 64 --psize 64
    --seed 123 --grid_seed 123 --grid_mode cycle_k
    --num_noise_levels 48 --langevin_steps 4 --gamma 0.5 --cg_iters 10 --cg_rtol 1e-6
    --dps_tail_steps 24 --dps_inner_loops 10 --save_pula_handoff
)

for convention in padis_real paper_complex; do
    python -u eval_pula/run.py "${COMMON[@]}" \
        --pula_convention "$convention" --out_root "$OUT_ROOT/$convention"
done
