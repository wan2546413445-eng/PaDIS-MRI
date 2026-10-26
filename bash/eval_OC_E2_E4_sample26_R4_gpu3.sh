#!/usr/bin/env bash
set -Eeuo pipefail

# Matched OC-only validation: E2 (Uniform Anchor OC=0) vs E4 (Uniform Anchor OC=0.5).
# Inference config matches Experiment 19: NEW-3T T1-FLAIR sample 26; R4; 78x10;
# seed 123 fixed per sample; LGFC disabled; 1 physical GPU; serial runs.
# KIMG=010000 (default); optionally KIMG=005003 if BOTH files exist.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train/padis-mri:${REPO_ROOT}/eval${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU="${GPU:-2}"
export CUDA_VISIBLE_DEVICES="${GPU}"
LOGICAL_GPU=0
KIMG="${KIMG:-010000}"
[[ "${KIMG}" =~ ^[0-9]{6}$ ]] || { echo "[ERROR] KIMG must be a 6-digit snapshot identifier" >&2; exit 2; }

TRAIN_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/agg_anchor_oc_training-runs/brain/32dB"
E2_DIR="${TRAIN_ROOT}/agg_anchor_oc_rho0_gamma0_oc0_pad64_seed123_train/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-agg-anchor-oc-rho0p0-gamma0p0-oc0p0-p64-train"
E4_DIR="${TRAIN_ROOT}/agg_anchor_oc_rho0_gamma0_oc0p5_pad64_seed123_train/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-agg-anchor-oc-rho0p0-gamma0p0-oc0p5-p64-train"
E2_MODEL="${E2_DIR}/network-snapshot-${KIMG}.pkl"
E4_MODEL="${E4_DIR}/network-snapshot-${KIMG}.pkl"

VAL_DIR="/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB"
RESULT_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/实验22_E2E4_OC_sample26_R4_s78x10_${KIMG}"
EXPERIMENT_ID="EXP22_E2E4_OC_S26_R4_78x10_${KIMG}"
SAMPLE_IDX=26
IMAGE_SIZE=384
PAD=64
PSIZE=64
MASK_SELECT=4
SEED=123
ZETA=3.0
STEPS=78
INNER_LOOPS=10
REUSE_COMPLETE="${REUSE_COMPLETE:-1}"
MASTER_LOG="${RESULT_ROOT}/eval_${EXPERIMENT_ID}_gpu${GPU}.log"

# Preflight: don't run if either counterpart or evaluation implementation is missing.
[[ -f "${REPO_ROOT}/eval/run.py" ]] || { echo "[ERROR] Missing eval/run.py. Put script under repo tools/." >&2; exit 2; }
[[ -d "${VAL_DIR}" ]] || { echo "[ERROR] Missing validation dir: ${VAL_DIR}" >&2; exit 2; }
for model in "${E2_MODEL}" "${E4_MODEL}"; do
    [[ -s "${model}" ]] || { echo "[ERROR] Missing checkpoint: ${model}" >&2; exit 2; }
done
for cfg in "${E2_DIR}/training_options.json" "${E4_DIR}/training_options.json"; do
    [[ -s "${cfg}" ]] || { echo "[ERROR] Missing training config: ${cfg}" >&2; exit 2; }
done

# Check that the training groups differ ONLY in OC loss coefficient and run path.
python - "${E2_DIR}/training_options.json" "${E4_DIR}/training_options.json" <<'PY'
import json, sys
from pathlib import Path
cfgs = [json.loads(Path(p).read_text(encoding='utf-8')) for p in sys.argv[1:3]]
for cfg, expected in zip(cfgs, (0.0, 0.5)):
    loss = cfg.get('loss_kwargs', {})
    if not loss.get('class_name', '').endswith('AnchorOverlapConsistencyPatchEDMLoss'):
        raise SystemExit('[ERROR] Expected AnchorOverlapConsistencyPatchEDMLoss')
    if float(loss.get('lambda_oc', -1)) != expected:
        raise SystemExit(f'[ERROR] Expected lambda_oc={expected}')
    if float(loss.get('agg_rho', -1)) != 0.0 or float(loss.get('agg_gamma', -1)) != 0.0:
        raise SystemExit('[ERROR] AGGA must be disabled for both E2/E4')
    if int(cfg.get('total_kimg', -1)) != 10000:
        raise SystemExit('[ERROR] Expected training total_kimg=10000')
for cfg in cfgs:
    cfg.pop('run_dir', None)
    cfg['loss_kwargs'].pop('lambda_oc', None)
if cfgs[0] != cfgs[1]:
    keys = sorted(k for k in set(cfgs[0]) | set(cfgs[1]) if cfgs[0].get(k) != cfgs[1].get(k))
    raise SystemExit(f'[ERROR] Training configuration mismatch beyond OC weight: {keys}')
print('[CHECK] Matched E2/E4 training options: only lambda_oc differs (0.0 vs 0.5).')
PY

mkdir -p "${RESULT_ROOT}"

case_complete() {
    local d="$1" model="$2"
    [[ -s "${d}/evaluate/recons/recon_patch_${SAMPLE_IDX}.npy" ]] &&
    [[ -s "${d}/evaluate/comp_plots/results.json" ]] &&
    [[ -f "${d}/model_path.txt" ]] &&
    [[ "$(< "${d}/model_path.txt")" == "${model}" ]]
}

run_one() {
    local group="$1" model="$2"
    local sample_dir="${RESULT_ROOT}/${group}/sample_${SAMPLE_IDX}"
    local log_file="${sample_dir}/run.log"
    mkdir -p "${sample_dir}"
    if [[ "${REUSE_COMPLETE}" == "1" ]] && case_complete "${sample_dir}" "${model}"; then
        echo "[REUSE] ${group} sample ${SAMPLE_IDX} (${model})"
        return 0
    fi
    echo "========================================================================"
    echo "Group=${group} | physical GPU=${GPU} | logical GPU=${LOGICAL_GPU}"
    echo "Model=${model}"
    echo "Sample=${SAMPLE_IDX} | R=${MASK_SELECT} | updates=${STEPS}x${INNER_LOOPS} | seed=${SEED}"
    echo "No AGGA. No LGFC. Output=${sample_dir}"
    echo "========================================================================"

    CUDA_VISIBLE_DEVICES="${GPU}" python -u eval/run.py \
        --run_evaluate \
        --algo padis \
        --model_path "${model}" \
        --val_dir "${VAL_DIR}" \
        --image_size "${IMAGE_SIZE}" \
        --pad "${PAD}" \
        --psize "${PSIZE}" \
        --mask_select "${MASK_SELECT}" \
        --val_count 1 \
        --sample_indices "${SAMPLE_IDX}" \
        --seed "${SEED}" \
        --fixed_seed_per_sample \
        --zeta "${ZETA}" \
        --steps "${STEPS}" \
        --inner_loops "${INNER_LOOPS}" \
        --report_every 1 \
        --save_dir "${sample_dir}" \
        --gpus "${LOGICAL_GPU}" \
        2>&1 | tee "${log_file}"

    [[ -s "${sample_dir}/evaluate/recons/recon_patch_${SAMPLE_IDX}.npy" ]] || {
        echo "[ERROR] Missing reconstruction after ${group}" >&2; return 3;
    }
    [[ -s "${sample_dir}/evaluate/comp_plots/results.json" ]] || {
        echo "[ERROR] Missing normalized metrics after ${group}" >&2; return 3;
    }
    printf '%s\n' "${model}" > "${sample_dir}/model_path.txt"
}

summarize_pair() {
    python - "${RESULT_ROOT}" "${SAMPLE_IDX}" "${KIMG}" "${E2_MODEL}" "${E4_MODEL}" <<'PY'
import csv, json, math, sys
from pathlib import Path
root = Path(sys.argv[1]); idx = int(sys.argv[2]); kimg = sys.argv[3]
models = {'E2_Uniform_Anchor_OC0':sys.argv[4], 'E4_Uniform_Anchor_OC0p5':sys.argv[5]}

def read(group):
    path = root / group / f'sample_{idx}' / 'evaluate' / 'comp_plots' / 'results.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    m = d.get('per_image', {}).get(str(idx))
    if m is None:
        s = d.get('summary', {})
        m = {k:s[f'{k}_mean'] for k in ('psnr','ssim','nrmse')}
    vals = {k.upper():float(m[k]) for k in ('psnr','ssim','nrmse')}
    if not all(math.isfinite(v) for v in vals.values()):
        raise ValueError(f'Nonfinite metrics: {path}: {vals}')
    return vals

u = read('E2_Uniform_Anchor_OC0'); oc = read('E4_Uniform_Anchor_OC0p5')
delta = {k:oc[k]-u[k] for k in u}
rows = [dict(group='E2_Uniform_Anchor_OC0',sample_index=idx,**u),
        dict(group='E4_Uniform_Anchor_OC0p5',sample_index=idx,**oc),
        dict(group='Delta_OC_minus_noOC',sample_index=idx,**delta)]
csv_path = root / f'sample26_{kimg}_E2E4_OC_paired_metrics.csv'
with csv_path.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f, fieldnames=['group','sample_index','PSNR','SSIM','NRMSE'])
    w.writeheader(); w.writerows(rows)
json_path = root / f'sample26_{kimg}_E2E4_OC_paired_metrics.json'
json_path.write_text(json.dumps({
    'sample':idx, 'checkpoint_kimg':kimg,
    'comparison':'E4 OC=0.5 minus E2 OC=0; AGGA disabled in both',
    'settings':dict(mask_select=4,steps=78,inner_loops=10,seed=123,fixed_seed_per_sample=True,LGFC=False),
    'models':models, 'E2_no_OC':u, 'E4_OC_0p5':oc, 'delta_OC_minus_noOC':delta
}, indent=2,ensure_ascii=False), encoding='utf-8')
print(f'\n=========== Pure OC matched E4-E2, sample {idx}, {kimg} kimg ===========')
print(f"{'Metric':<8} {'E2 OC=0':>14} {'E4 OC=0.5':>14} {'E4-E2':>16}")
for k in ('PSNR','SSIM','NRMSE'):
    print(f'{k:<8} {u[k]:14.6f} {oc[k]:14.6f} {delta[k]:+16.6f}')
print('Positive PSNR/SSIM deltas and negative NRMSE delta mean OC improvement.')
print('CSV :',csv_path)
print('JSON:',json_path)
print('=======================================================================')
PY
}

{
    echo "[START] ${EXPERIMENT_ID}"
    echo "[INFO] Serial E2 (no OC) -> E4 (OC=0.5), same model family, sample26, R4, 78x10, seed123"
    run_one 'E2_Uniform_Anchor_OC0' "${E2_MODEL}"
    run_one 'E4_Uniform_Anchor_OC0p5' "${E4_MODEL}"
    summarize_pair
    echo "[SUCCESS] OC-only paired diagnostic completed. ${RESULT_ROOT}"
} 2>&1 | tee "${MASTER_LOG}"
