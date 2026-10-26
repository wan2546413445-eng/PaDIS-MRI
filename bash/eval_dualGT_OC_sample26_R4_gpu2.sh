#!/usr/bin/env bash
set -Eeuo pipefail

# Old Dual-GT PaDIS-MRI, matched OC ablation on the NEW 3T T1-FLAIR sample 26.
# Control: BOTH overlapping patches have GT EDM supervision, lambda_overlap=0.
# OC arm:  SAME Dual-GT EDM supervision, lambda_overlap=0.5.
# Training checkpoint: 005040 (default). No retraining; serial GPU2 inference.
# R4 / 78 outer steps x 10 inner loops / seed123 fixed per sample / no LGFC.
# Based on the user's existing E2/E4 script; separate output location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train/padis-mri:${REPO_ROOT}/eval${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=2  # fixed physical CUDA GPU 2; mapped to cuda:0 inside this process
export CUDA_VISIBLE_DEVICES="${GPU}"
LOGICAL_GPU=0
KIMG="${KIMG:-005040}"
[[ "${KIMG}" =~ ^[0-9]{6}$ ]] || { echo "[ERROR] KIMG must be a 6-digit snapshot identifier" >&2; exit 2; }

TRAIN_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/OC/overlap_independent_training-runs/brain/32dB"
GT0_DIR="${TRAIN_ROOT}/main_overlap_independent_control_lam0p0_active64_bgpu1_s16s32s64_p020305_b2_seed123/00001-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-overlap-independent-center-lam0p0-p64"
GT5_DIR="${TRAIN_ROOT}/main_overlap_independent_center_lam0p5_active64_bgpu1_s16s32s64_p020305_b2_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-overlap-independent-center-lam0p5-p64"
GT0_MODEL="${GT0_DIR}/network-snapshot-${KIMG}.pkl"
GT5_MODEL="${GT5_DIR}/network-snapshot-${KIMG}.pkl"

VAL_DIR="/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB"
RESULT_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/dualGT_old_OC_sample26_new3T_R4_s78x10_${KIMG}"
EXPERIMENT_ID="dualGT_OC_old_ckpt${KIMG}_new3T_S26_R4_78x10"
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
for model in "${GT0_MODEL}" "${GT5_MODEL}"; do
    [[ -s "${model}" ]] || { echo "[ERROR] Missing checkpoint: ${model}" >&2; exit 2; }
done
for cfg in "${GT0_DIR}/training_options.json" "${GT5_DIR}/training_options.json"; do
    [[ -s "${cfg}" ]] || { echo "[ERROR] Missing training config: ${cfg}" >&2; exit 2; }
done

# Verify identical old Dual-GT training recipes; only OC weight differs.
python - "${GT0_DIR}/training_options.json" "${GT5_DIR}/training_options.json" <<'PY'
import json, sys
from pathlib import Path
cfgs = [json.loads(Path(p).read_text(encoding='utf-8')) for p in sys.argv[1:3]]
for cfg, expected in zip(cfgs, (0.0, 0.5)):
    loss = cfg.get('loss_kwargs', {})
    if not loss.get('class_name', '').endswith('IndependentNoiseOverlapPatch_EDMLoss'):
        raise SystemExit('[ERROR] Expected old IndependentNoiseOverlapPatch_EDMLoss (Dual-GT)')
    if float(loss.get('lambda_overlap', -1)) != expected:
        raise SystemExit(f'[ERROR] Expected lambda_overlap={expected}')
    if int(loss.get('active_patch_size', -1)) != 64:
        raise SystemExit('[ERROR] Expected OC active patch size=64')
    if cfg.get('patch_list') != [16, 32, 64] or cfg.get('patch_probs') != [0.2, 0.3, 0.5]:
        raise SystemExit('[ERROR] Unexpected patch sizes/probabilities')
    if int(cfg.get('pad_width', -1)) != 96:
        raise SystemExit('[ERROR] Expected old training pad_width=96')
for cfg in cfgs:
    cfg.pop('run_dir', None)
    cfg['loss_kwargs'].pop('lambda_overlap', None)
if cfgs[0] != cfgs[1]:
    keys = sorted(k for k in set(cfgs[0]) | set(cfgs[1]) if cfgs[0].get(k) != cfgs[1].get(k))
    raise SystemExit(f'[ERROR] Training configuration mismatch beyond OC weight: {keys}')
print('[CHECK] Old Dual-GT training options matched: only lambda_overlap differs (0.0 vs 0.5).')
print('[INFO] Both trained with pad_width=96; both evaluated with pad=64 for NEW-3T protocol.')
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
    python - "${RESULT_ROOT}" "${SAMPLE_IDX}" "${KIMG}" "${GT0_MODEL}" "${GT5_MODEL}" <<'PY'
import csv, json, math, sys
from pathlib import Path
root = Path(sys.argv[1]); idx = int(sys.argv[2]); kimg = sys.argv[3]
models = {'DualGT_OC0':sys.argv[4], 'DualGT_OC0p5':sys.argv[5]}

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

u = read('DualGT_OC0'); oc = read('DualGT_OC0p5')
delta = {k:oc[k]-u[k] for k in u}
rows = [dict(group='DualGT_OC0',sample_index=idx,**u),
        dict(group='DualGT_OC0p5',sample_index=idx,**oc),
        dict(group='Delta_OC_minus_noOC',sample_index=idx,**delta)]
csv_path = root / f'sample26_{kimg}_dualGT_OC_paired_metrics.csv'
with csv_path.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f, fieldnames=['group','sample_index','PSNR','SSIM','NRMSE'])
    w.writeheader(); w.writerows(rows)
json_path = root / f'sample26_{kimg}_dualGT_OC_paired_metrics.json'
json_path.write_text(json.dumps({
    'sample':idx, 'checkpoint_kimg':kimg,
    'comparison':'Old Dual-GT OC=0.5 minus old Dual-GT OC=0; matched old training configurations',
    'settings':dict(validation='new3T_T1FLAIR_32dB',mask_select=4,steps=78,inner_loops=10,seed=123,fixed_seed_per_sample=True,training_pad_width=96,inference_pad=64,LGFC=False,physical_gpu=2),
    'models':models, 'DualGT_no_OC':u, 'DualGT_OC_0p5':oc, 'delta_OC_minus_noOC':delta
}, indent=2,ensure_ascii=False), encoding='utf-8')
print(f'\n=========== Old Dual-GT matched OC ablation, sample {idx}, {kimg} kimg ===========')
print(f"{'Metric':<8} {'DualGT OC=0':>14} {'DualGT OC=0.5':>14} {'OC-noOC':>16}")
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
    echo "[INFO] Old Dual-GT OC=0 -> Old Dual-GT OC=0.5, sample26 NEW3T, R4, 78x10, seed123"
    run_one 'DualGT_OC0' "${GT0_MODEL}"
    run_one 'DualGT_OC0p5' "${GT5_MODEL}"
    summarize_pair
    echo "[SUCCESS] Old Dual-GT OC paired diagnostic completed. ${RESULT_ROOT}"
} 2>&1 | tee "${MASTER_LOG}"
