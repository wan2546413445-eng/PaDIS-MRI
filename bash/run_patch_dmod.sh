#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"
GPU=${GPU:-6}
MASKS=${MASKS:-"4 7"}
SAMPLES=${SAMPLES:-0,26}
T=${T:-30}; S=${S:-5}; K=${K:-1}; L=${L:-10}
SIGMA_MAX=${SIGMA_MAX:-1}; SIGMA_MIN=${SIGMA_MIN:-0.003}; RHO=${RHO:-7}
BETA_SCALE=${BETA_SCALE:-1}; SEED=${SEED:-123}; GRID_SEED=${GRID_SEED:-123}
TAG=${TAG:-patch_dmod}
[[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ && "$TAG" != . && "$TAG" != .. ]] || { echo 'Invalid TAG'; exit 2; }
ROOT=/mnt/SSD2/wsy/PaDIS-MRI
RUN=resume10k_overlap_independent_center_symmetric_lam2p5_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform
JOB=00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam2p5-p64-align-symmetric-geom-identical-spatial-uniform
MODEL=${MODEL:-$ROOT/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB/$RUN/$JOB/network-snapshot-010000.pkl}
VAL=${VAL:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}
BASE=${BASE:-$ROOT/PaDIS-MRI-recon/new_data/Patch_DMOD_OPA10k}
# Full partition (7x7) is a hard requirement; no silent fallback.
[[ ${PADIS_PATCH_BATCH:-49} == 49 && ${PADIS_PATCH_CHECKPOINT:-0} == 0 ]] || { echo 'Set PADIS_PATCH_BATCH=49 PADIS_PATCH_CHECKPOINT=0'; exit 2; }
export PADIS_PATCH_BATCH=49 PADIS_PATCH_CHECKPOINT=0 CUDA_VISIBLE_DEVICES="$GPU"
EXTRA=()
[[ -z ${NOISE_DIR:-} ]] || EXTRA+=(--noise_dir "$NOISE_DIR")
[[ -z ${NOISE_COVARIANCE:-} ]] || EXTRA+=(--noise_covariance "$NOISE_COVARIANCE")
[[ -z ${NOISE_COMPLEX_VAR:-} ]] || EXTRA+=(--noise_complex_var "$NOISE_COMPLEX_VAR")
[[ ${DRY_RUN:-0} != 1 ]] || EXTRA+=(--dry_run)
for mask in $MASKS; do
    OUT="$BASE/R${mask}/T${T}_S${S}_K${K}_L${L}_karras_smax${SIGMA_MAX}_smin${SIGMA_MIN}_rho${RHO}_beta${BETA_SCALE}_seed${SEED}_grid${GRID_SEED}_${TAG}_mb49_ck0"
    cmd=(python -u eval_pula/run_patch_dmod.py --model_path "$MODEL" --val_dir "$VAL" --out_root "$OUT"
         --sample_indices "$SAMPLES" --mask_select "$mask" --T "$T" --S "$S" --K "$K" --cg_iters "$L" --cg_rtol 1e-6
         --sigma_max "$SIGMA_MAX" --sigma_min "$SIGMA_MIN" --rho "$RHO" --beta_scale "$BETA_SCALE"
         --seed "$SEED" --grid_seed "$GRID_SEED" --image_size 384 --pad 64 --psize 64 "${EXTRA[@]}")
    echo "GPU=$GPU R$mask T=$T S=$S K=$K L=$L | $OUT"
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        "${cmd[@]}"
    else
        mkdir -p "$OUT"
        "${cmd[@]}" 2>&1 | tee -a "$OUT/run.log"
    fi
done
