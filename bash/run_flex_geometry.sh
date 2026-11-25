#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"
GPU=${GPU:-6}
GEOMETRY_MODE=${GEOMETRY_MODE:-mixed}
POLY_ORDER=${POLY_ORDER:-2}
MASKS=${MASKS:-"4 7"}
SAMPLES=${SAMPLES:-0}
N=${N:-47}
SIGMA_SCHEDULE=${SIGMA_SCHEDULE:-karras}
PULA_GRID_MODE=${PULA_GRID_MODE:-cycle_k}
DPS_GRID_MODE=${DPS_GRID_MODE:-random_step}
PADDING_MODE=${PADDING_MODE:-reset}
DPS_COV_MODE=${DPS_COV_MODE:-off}
COV_SCALE=${COV_SCALE:-1}
COV_CG_ITERS=${COV_CG_ITERS:-16}
COV_CG_RTOL=${COV_CG_RTOL:-0.001}
ROOT=/mnt/SSD2/wsy/PaDIS-MRI
RUN=resume10k_overlap_independent_center_symmetric_lam2p5_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform
JOB=00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam2p5-p64-align-symmetric-geom-identical-spatial-uniform
MODEL=${MODEL:-$ROOT/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB/$RUN/$JOB/network-snapshot-010000.pkl}
VAL=${VAL:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}
BASE=${BASE:-$ROOT/PaDIS-MRI-recon/new_data/pULA_DPS_ablation/Flex_SharedGeometry_OPA10k}
# Default pattern unless explicit schedule blocks/file are supplied.
if [[ -z ${PATTERN:-} && -z ${BLOCKS:-} && -z ${SCHEDULE_FILE:-} ]]; then BLOCKS="1-10:PPPPP;11-25:PPPDD;26-35:PPDDD;36-47:DDDDD"; TAG=${TAG:-progressive55}; fi
ARGS=()
[[ -z ${PATTERN:-} ]] || ARGS+=(--pattern "$PATTERN")
[[ -z ${BLOCKS:-} ]] || ARGS+=(--blocks "$BLOCKS")
[[ -z ${SCHEDULE_FILE:-} ]] || ARGS+=(--schedule_file "$SCHEDULE_FILE")
if [[ -n ${BLOCKS:-} || -n ${SCHEDULE_FILE:-} ]]; then
    [[ -n ${TAG:-} ]] || { echo 'For BLOCKS/SCHEDULE_FILE set TAG=<name>'; exit 2; }
fi
TAG=${TAG:-${PATTERN:-}}
if [[ ! "$TAG" =~ ^[A-Za-z0-9_.-]+$ || "$TAG" == . || "$TAG" == .. ]]; then echo 'Invalid TAG'; exit 2; fi
for mask in $MASKS; do
    OUT="$BASE/R${mask}/N${N}_${SIGMA_SCHEDULE}_${TAG}_Pg${PULA_GRID_MODE}_Dg${DPS_GRID_MODE}"
    OUT="${OUT}_geo${GEOMETRY_MODE}_m${POLY_ORDER}_pad${PADDING_MODE}_cov${DPS_COV_MODE}_mb${PADIS_PATCH_BATCH:-49}_ck${PADIS_PATCH_CHECKPOINT:-1}"
    EXTRA=(--geometry_mode "$GEOMETRY_MODE" --poly_order "$POLY_ORDER" --padding_mode "$PADDING_MODE" --dps_cov_mode "$DPS_COV_MODE" --cov_scale "$COV_SCALE" --cov_cg_iters "$COV_CG_ITERS" --cov_cg_rtol "$COV_CG_RTOL")
    [[ -z ${COV_CACHE:-} ]] || EXTRA+=(--cov_cache "$COV_CACHE")
    [[ ${DRY_RUN:-0} != 1 ]] || EXTRA+=(--dry_run)
    if [[ ${DRY_RUN:-0} != 1 ]]; then mkdir -p "$OUT"; fi
    cmd=(python -u eval_pula/run_flex_geometry.py --model_path "$MODEL" --val_dir "$VAL" --out_root "$OUT"
      --sample_indices "$SAMPLES" --mask_select "$mask" --num_noise_levels "$N"
      --sigma_schedule "$SIGMA_SCHEDULE" --sigma_max 1 --sigma_min 0.003 --rho 7
      --gamma 0.5 --dps_zeta 3 --cg_iters 10 --cg_rtol 1e-6 --seed 123 --grid_seed 123
      --grid_mode cycle_k --pula_grid_mode "$PULA_GRID_MODE" --dps_grid_mode "$DPS_GRID_MODE" --pula_convention padis_real --image_size 384 --pad 64 --psize 64
      "${ARGS[@]}" "${EXTRA[@]}")
    echo "GPU=$GPU R$mask N=$N TAG=$TAG | $OUT"
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        "${cmd[@]}"
    else
        env -u NOISE_DIR -u NOISE_COVARIANCE -u NOISE_COMPLEX_VAR CUDA_VISIBLE_DEVICES="$GPU" \
            PADIS_PATCH_BATCH="${PADIS_PATCH_BATCH:-49}" PADIS_PATCH_CHECKPOINT="${PADIS_PATCH_CHECKPOINT:-1}" "${cmd[@]}" 2>&1 | tee -a "$OUT/run.log"
    fi
done
