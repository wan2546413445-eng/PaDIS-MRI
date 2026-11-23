#!/usr/bin/env bash
# Pilot: 47 noise levels * 5 shared partitions = 235 denoiser evaluations.
# One MRI CG solve per sigma level, no DPS / no Langevin updates.
# Usage:
#   GPU=6 PLAN_ONLY=1 CONSENSUS_MODE=mean bash bash/eval_fixed47_consensus_gpu6.sh
#   GPU=6 PLAN_ONLY=0 CONSENSUS_MODE=mean bash bash/eval_fixed47_consensus_gpu6.sh
#   GPU=6 PLAN_ONLY=0 CONSENSUS_MODE=dual bash bash/eval_fixed47_consensus_gpu6.sh
set -Eeuo pipefail
cd "${PROJECT_DIR:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}"
GPU=${GPU:-6}
PLAN_ONLY=${PLAN_ONLY:-1}
CONSENSUS_MODE=${CONSENSUS_MODE:-mean}
SAMPLES=${SAMPLES:-0,8,22,26}
MASKS=${MASKS:-7}
VAL=${VAL:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}
OUT_BASE=${OUT_BASE:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/fixed47_consensus_OPA10k_${CONSENSUS_MODE}}
RUNS=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs
TRAIN=resume10k_overlap_independent_center_symmetric_lam2p5_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform
JOB=00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam2p5-p64-align-symmetric-geom-identical-spatial-uniform
OPA_MODEL=${OPA_MODEL:-$RUNS/overlap_independent_training-runs/brain/32dB/$TRAIN/$JOB/network-snapshot-010000.pkl}
case "$CONSENSUS_MODE" in mean|dual) ;; *) echo 'CONSENSUS_MODE=mean or dual' >&2; exit 2;; esac
[[ "$PLAN_ONLY" == 0 || "$PLAN_ONLY" == 1 ]] || { echo 'PLAN_ONLY=0 or 1' >&2; exit 2; }
case "$OPA_MODEL" in
  */overlap_independent_training-runs/brain/32dB/*lam2p5*/*/network-snapshot-010000.pkl) ;;
  *) echo "Wrong OPA10k model: $OPA_MODEL" >&2; exit 2;;
esac
export CUDA_VISIBLE_DEVICES="$GPU" MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/train/padis-mri:$PWD/eval${PYTHONPATH:+:$PYTHONPATH}"
export PADIS_PATCH_BATCH=${PADIS_PATCH_BATCH:-49}
export PADIS_PATCH_CHECKPOINT=${PADIS_PATCH_CHECKPOINT:-0}
unset PA_DPS_TRACE_DIR PA_DPS_TRACE_LEVELS
printf 'Consensus fixed47 | GPU=%s mode=%s plan_only=%s samples=%s masks=%s\n' \
       "$GPU" "$CONSENSUS_MODE" "$PLAN_ONLY" "$SAMPLES" "$MASKS"
printf 'Checkpoint=%s\nOutput=%s\n' "$OPA_MODEL" "$OUT_BASE"
for MASK in $MASKS; do
    case "$MASK" in 4|7) ;; *) echo "Unsupported R$MASK" >&2; exit 2;; esac
    DEST="$OUT_BASE/R${MASK}/consensus_${CONSENSUS_MODE}_47x5"
    args=(
        --model_path "$OPA_MODEL" --val_dir "$VAL" --out_root "$DEST"
        --sample_indices "$SAMPLES" --mask_select "$MASK"
        --score_mode patch --grid_mode cycle_k --pula_convention padis_real
        --pula_variant baseline --seed 123 --grid_seed 123
        --image_size 384 --pad 64 --psize 64
        --fixed47_switch 47 --fixed47_patch_policy shared_cycle_k
        --num_noise_levels 47 --langevin_steps 5 --dps_inner_loops 5
        --sigma_max 1.0 --sigma_min 0.003
        --dps_sigma_max 1.0 --dps_sigma_min 0.003 --dps_base_steps 47 --dps_rho 7.0
        --gamma 0.5 --cg_iters 10 --cg_rtol 1e-6 --dps_zeta 3.0
        --consensus_mode "$CONSENSUS_MODE"
    )
    PLAN=$(python eval_pula/run.py "${args[@]}" --dry_run)
    printf '%s' "$PLAN" | python -c '
import json,sys,hashlib
p=json.load(sys.stdin)
assert p["consensus_levels"]==47 and p["consensus_updates"]==47
assert p["denoiser_evaluations"]==235 and p["cg_solves_per_trajectory"]==48
assert p["logical_updates"]==47 and p["dps_updates"]==0
assert p["fixed47_switch"]==47
h=hashlib.sha256(json.dumps(p["fixed47_global_sigmas"],separators=(",",":")).encode()).hexdigest()
print("Verified: 47 levels, 235 denoiser evaluations, 48 CG including initialization; sigma hash:", h)
'
    if [[ "$PLAN_ONLY" == 1 ]]; then
        continue
    fi
    [[ -s "$OPA_MODEL" && -d "$VAL" ]] || { echo 'Checkpoint or validation set missing' >&2; exit 1; }
    mkdir -p "$DEST"
    printf '%s\n' "$PLAN" > "$DEST/ablation_plan.json"
    python -u eval_pula/run.py "${args[@]}" 2>&1 | tee -a "$DEST/terminal.log"
    python - "$DEST" "$MASK" "$SAMPLES" "$OUT_BASE" <<'PY_SUM'
import csv,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); mask=int(sys.argv[2]); ids=[int(i) for i in sys.argv[3].split(',')]; base=Path(sys.argv[4]); rows=[]
for index in ids:
    ev=root/'cycle_k'/f'sample_{index}'/'evaluate'
    metrics=ev/'comp_plots/results.json'; timing=ev/'timing.json'
    if not metrics.exists() or not timing.exists():
        continue
    m=json.loads(metrics.read_text())['per_image'][str(index)]
    t=json.loads(timing.read_text())
    rows.append(dict(sample=index, psnr=float(m['psnr']),ssim=float(m['ssim']),
                     nrmse=float(m['nrmse']),time_seconds=float(t['reconstruction_seconds']),
                     denoiser_evaluations=t['consensus_partition_evaluations'],
                     cg_solves=t['cg_solves'],network_forwards=t['network_forwards']))
base.mkdir(parents=True,exist_ok=True)
if rows:
    with (base/f'per_image_R{mask}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'R{mask} consensus {len(rows)}/{len(ids)}: '
          f'PSNR={statistics.mean(r["psnr"] for r in rows):.4f}, '
          f'SSIM={statistics.mean(r["ssim"] for r in rows):.5f}, '
          f'NRMSE={statistics.mean(r["nrmse"] for r in rows):.5f}, '
          f'time={statistics.mean(r["time_seconds"] for r in rows):.2f}s')
PY_SUM
done
