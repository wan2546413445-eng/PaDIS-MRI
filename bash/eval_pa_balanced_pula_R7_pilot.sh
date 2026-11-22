#!/usr/bin/env bash
set -Eeuo pipefail
cd "${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

VARIANT="${VARIANT:-baseline}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
if (( $# )); then echo "Usage: VARIANT=baseline|noise08|balanced_k0|balanced_full bash $0 [--dry-run]" >&2; exit 2; fi
case "$VARIANT" in baseline|noise08|balanced_k0|balanced_full) ;; *) echo "Invalid VARIANT=$VARIANT" >&2; exit 2;; esac
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
PYTHON_BIN="${PYTHON:-python}"
CONVENTION="${CONVENTION:-padis_real}"
SEED="${SEED:-123}"
GRID_SEED="${GRID_SEED:-123}"
VAL_T1FLAIR="${VAL_T1FLAIR:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}"
VAL_T2="${VAL_T2:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t2/32dB}"
OUT_ROOT="${OUT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/PA_Balanced_pULA_R7_pilot}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

# Same E0 directory and 10k selection as eval_pula_convention_sample0.sh.
E0_RUN='/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/消融实验/E0_agg_simplified_single_training-runs/brain/32dB/agg_single_rho0_gamma0_pad64_seed123_train'
if [[ -z "${MODEL_E0:-}" ]]; then
    shopt -s nullglob
    checkpoints=("$E0_RUN"/*/network-snapshot-010000.pkl)
    if (( ${#checkpoints[@]} == 1 )); then
        MODEL_E0="${checkpoints[0]}"
    elif [[ "$DRY_RUN" == 1 && ${#checkpoints[@]} == 0 ]]; then
        MODEL_E0="$E0_RUN/*/network-snapshot-010000.pkl"
        echo "Dry-run: server checkpoint is unresolved; parameter validation only." >&2
    else
        echo "Expected one E0 10k checkpoint under $E0_RUN, found ${#checkpoints[@]}. Set MODEL_E0." >&2
        exit 1
    fi
fi

COMMON=(
    --model_path "$MODEL_E0" --pula_variant "$VARIANT" --pula_convention "$CONVENTION"
    --sample_indices 0,26 --mask_select 7 --image_size 384 --pad 64 --psize 64
    --seed "$SEED" --grid_seed "$GRID_SEED" --grid_mode random_step
    --sigma_max 1.0 --sigma_min 0.01 --num_noise_levels 48 --langevin_steps 4
    --gamma 0.5 --cg_iters 10 --cg_rtol 1e-6
    --dps_tail_steps 24 --dps_base_steps 78 --dps_inner_loops 10
    --dps_sigma_max 10.0 --dps_sigma_min 0.003 --dps_rho 7.0 --dps_zeta 3.0
    --pa_block_size 8 --pa_epsilon 1e-12 --save_pula_handoff
)
if [[ -n "${NOISE_DIR:-}" ]]; then COMMON+=(--noise_dir "$NOISE_DIR"); fi
if [[ "$DRY_RUN" == 1 ]]; then
    COMMON+=(--dry_run)
else
    [[ -f "$MODEL_E0" ]] || { echo "Missing checkpoint: $MODEL_E0" >&2; exit 1; }
    for val in "$VAL_T1FLAIR" "$VAL_T2"; do
        for index in 0 26; do
            [[ -f "$val/sample_$index.pt" ]] || { echo "Missing sample: $val/sample_$index.pt" >&2; exit 1; }
        done
    done
fi

for contrast in T1FLAIR T2; do
    if [[ "$contrast" == T1FLAIR ]]; then val="$VAL_T1FLAIR"; else val="$VAL_T2"; fi
    out="$OUT_ROOT/$RUN_ID/$VARIANT/$CONVENTION/seed${SEED}/$contrast"
    command=("$PYTHON_BIN" -u eval_pula/run.py "${COMMON[@]}" --val_dir "$val" --out_root "$out")
    printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
    if [[ "$DRY_RUN" == 1 ]]; then
        "${command[@]}"
    else
        mkdir -p "$(dirname "$out")"
        mkdir "$out"  # A repeated pilot never overwrites an existing run.
        "${command[@]}" 2>&1 | tee "$out/launcher.log"
    fi
done
