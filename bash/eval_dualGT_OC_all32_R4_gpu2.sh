#!/usr/bin/env bash
set -Eeuo pipefail

# Matched OLD Dual-GT overlap consistency ablation, NEW 3T T1-FLAIR 32 slices.
# Same dataset, same old Dual-GT loss class, same training setup; only lambda_overlap 0 vs 0.5.
# Checkpoints: 005040; inference: R4, 78 outer x 10 inner, seed 123 fixed per sample.
# Physical GPU 2 only; no AGGA inference, no LGFC. Runs one sample at a time to allow safe resume.
# Put this script in <repo>/bash/ and run: bash bash/eval_dualGT_OC_all32_R4_gpu2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train/padis-mri:${REPO_ROOT}/eval${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=2
export CUDA_VISIBLE_DEVICES="${GPU}"
LOGICAL_GPU=0
KIMG=005040
REUSE_COMPLETE="${REUSE_COMPLETE:-1}"
REUSE_PREVIOUS_S26="${REUSE_PREVIOUS_S26:-1}"

TRAIN_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/OC/overlap_independent_training-runs/brain/32dB"
GT0_DIR="${TRAIN_ROOT}/main_overlap_independent_control_lam0p0_active64_bgpu1_s16s32s64_p020305_b2_seed123/00001-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-overlap-independent-center-lam0p0-p64"
GT5_DIR="${TRAIN_ROOT}/main_overlap_independent_center_lam0p5_active64_bgpu1_s16s32s64_p020305_b2_seed123/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-overlap-independent-center-lam0p5-p64"
GT0_MODEL="${GT0_DIR}/network-snapshot-${KIMG}.pkl"
GT5_MODEL="${GT5_DIR}/network-snapshot-${KIMG}.pkl"

VAL_DIR="/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB"
RESULT_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/dualGT_old_OC_all32_new3T_R4_s78x10_${KIMG}"
PREVIOUS_S26_ROOT="/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/dualGT_old_OC_sample26_new3T_R4_s78x10_${KIMG}"
EXPERIMENT_ID="dualGT_OC_old_ckpt${KIMG}_new3T_ALL32_R4_78x10"

IMAGE_SIZE=384
PAD=64
PSIZE=64
MASK_SELECT=4
SEED=123
ZETA=3.0
STEPS=78
INNER_LOOPS=10
MASTER_LOG="${RESULT_ROOT}/eval_${EXPERIMENT_ID}_gpu${GPU}.log"

# ----------------------- Preflight -----------------------
[[ -s "${REPO_ROOT}/eval/run.py" ]] || { echo '[ERROR] Missing eval/run.py' >&2; exit 2; }
[[ -d "${VAL_DIR}" ]] || { echo "[ERROR] Missing validation directory: ${VAL_DIR}" >&2; exit 2; }
for model in "${GT0_MODEL}" "${GT5_MODEL}"; do
    [[ -s "${model}" ]] || { echo "[ERROR] Missing checkpoint: ${model}" >&2; exit 2; }
done
for cfg in "${GT0_DIR}/training_options.json" "${GT5_DIR}/training_options.json"; do
    [[ -s "${cfg}" ]] || { echo "[ERROR] Missing training options: ${cfg}" >&2; exit 2; }
done

python - "${GT0_DIR}/training_options.json" "${GT5_DIR}/training_options.json" <<'PY'
import copy, json, sys
from pathlib import Path
raw = [json.loads(Path(p).read_text(encoding='utf-8')) for p in sys.argv[1:3]]
for cfg, lam in zip(raw, (0.0, 0.5)):
    loss = cfg.get('loss_kwargs', {})
    if not loss.get('class_name', '').endswith('IndependentNoiseOverlapPatch_EDMLoss'):
        raise SystemExit('[ERROR] Wrong loss class; expected OLD Dual-GT independent-noise OC')
    if float(loss.get('lambda_overlap', -1)) != lam:
        raise SystemExit(f'[ERROR] Expected lambda_overlap={lam}')
    if int(loss.get('active_patch_size', -1)) != 64:
        raise SystemExit('[ERROR] Expected active_patch_size=64')
    if cfg.get('patch_list') != [16, 32, 64] or cfg.get('patch_probs') != [0.2, 0.3, 0.5]:
        raise SystemExit('[ERROR] Different patch sampling schedule')
    if int(cfg.get('pad_width', -1)) != 96:
        raise SystemExit('[ERROR] Expected training pad_width=96')
a, b = copy.deepcopy(raw)
for cfg in (a, b):
    cfg.pop('run_dir', None)
    cfg['loss_kwargs'].pop('lambda_overlap', None)
if a != b:
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    raise SystemExit('[ERROR] Training mismatch beyond lambda_overlap: ' + ', '.join(keys))
print('[CHECK] Training options match, except lambda_overlap: 0 vs 0.5.')
print('[INFO] Training pad=96, both evaluated using NEW-3T inference pad=64.')
PY

mkdir -p "${RESULT_ROOT}"

# A case counts as complete only if its reconstruction, per-image metrics, and model ID exist.
case_complete() {
    local d="$1" model="$2" idx="$3"
    [[ -s "${d}/evaluate/recons/recon_patch_${idx}.npy" ]] &&
    [[ -s "${d}/evaluate/comp_plots/results.json" ]] &&
    [[ -s "${d}/model_path.txt" ]] &&
    [[ "$(< "${d}/model_path.txt")" == "${model}" ]]
}

run_one() {
    local group="$1" model="$2" idx="$3"
    local sample_dir="${RESULT_ROOT}/${group}/sample_${idx}"
    local previous_dir="${PREVIOUS_S26_ROOT}/${group}/sample_${idx}"
    local log_file="${sample_dir}/run.log"
    mkdir -p "${sample_dir}"

    if [[ "${REUSE_COMPLETE}" == "1" ]] && case_complete "${sample_dir}" "${model}" "${idx}"; then
        echo "[REUSE] ${group} sample=${idx} already complete in ALL32 output"
        return 0
    fi

    # The earlier successful sample 26 used exactly the same model and inference protocol.
    # Only reuse it when its model identity and metrics are present; never modify the old run.
    if [[ "${REUSE_PREVIOUS_S26}" == "1" && "${idx}" == "26" ]] && \
       case_complete "${previous_dir}" "${model}" "${idx}"; then
        cp -a "${previous_dir}/." "${sample_dir}/"
        printf '%s\n' "${previous_dir}" > "${sample_dir}/reuse_source.txt"
        echo "[IMPORT] ${group} sample=26 from previously completed matched experiment"
        return 0
    fi

    echo '=================================================================='
    echo "[RUN] ${group} | sample ${idx}/31 | physical GPU ${GPU} (logical ${LOGICAL_GPU})"
    echo "[MODEL] ${model}"
    echo "[PROTOCOL] NEW-3T T1-FLAIR; R4; 78x10; seed=123; no LGFC"
    echo '=================================================================='

    # Each sample is an independent eval, enabling reliable per-case restart and fixed RNG.
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
        --sample_indices "${idx}" \
        --seed "${SEED}" \
        --fixed_seed_per_sample \
        --zeta "${ZETA}" \
        --steps "${STEPS}" \
        --inner_loops "${INNER_LOOPS}" \
        --report_every 1 \
        --save_dir "${sample_dir}" \
        --gpus "${LOGICAL_GPU}" \
        2>&1 | tee "${log_file}"

    [[ -s "${sample_dir}/evaluate/recons/recon_patch_${idx}.npy" ]] || {
        echo "[ERROR] Missing reconstruction: ${group}, sample ${idx}" >&2; return 3;
    }
    [[ -s "${sample_dir}/evaluate/comp_plots/results.json" ]] || {
        echo "[ERROR] Missing post-normalization results: ${group}, sample ${idx}" >&2; return 3;
    }
    printf '%s\n' "${model}" > "${sample_dir}/model_path.txt"
}

summarize_pair() {
    python - "${RESULT_ROOT}" "${KIMG}" "${GT0_MODEL}" "${GT5_MODEL}" <<'PY'
import csv, json, math, statistics, sys
from pathlib import Path
root = Path(sys.argv[1]); kimg = sys.argv[2]
models = {'DualGT_OC0': sys.argv[3], 'DualGT_OC0p5': sys.argv[4]}
fields = ('psnr', 'ssim', 'nrmse')

def get_metrics(group, idx):
    directory = root / group / f'sample_{idx}'
    model_id = (directory / 'model_path.txt').read_text(encoding='utf-8').strip()
    if model_id != models[group]:
        raise ValueError(f'Checkpoint mismatch: {group}, sample {idx}')
    path = directory / 'evaluate/comp_plots/results.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    measures = record.get('per_image', {}).get(str(idx))
    if measures is None:
        raise ValueError(f'Missing per_image[{idx}] in {path}; refusing summary fallback')
    result = {k: float(measures[k]) for k in fields}
    if any(not math.isfinite(v) for v in result.values()):
        raise ValueError(f'Non-finite metric for {group}, sample {idx}: {result}')
    if result['psnr'] <= 0 or result['ssim'] <= 0 or result['nrmse'] <= 0:
        raise ValueError(f'Nonpositive metric for {group}, sample {idx}: {result}')
    return result

rows = []
for i in range(32):
    a = get_metrics('DualGT_OC0', i)
    b = get_metrics('DualGT_OC0p5', i)
    row = {'sample_index': i}
    for k in fields:
        row[f'{k}_OC0'] = a[k]
        row[f'{k}_OC0p5'] = b[k]
        row[f'delta_{k}'] = b[k] - a[k]
    rows.append(row)

csv_out = root / f'all32_{kimg}_dualGT_OC_paired_metrics.csv'
with csv_out.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for k in fields:
    values_a = [r[f'{k}_OC0'] for r in rows]
    values_b = [r[f'{k}_OC0p5'] for r in rows]
    differences = [r[f'delta_{k}'] for r in rows]
    improves = [d > 0 if k != 'nrmse' else d < 0 for d in differences]
    summary[k] = {
        'OC0_mean': statistics.mean(values_a),
        'OC0p5_mean': statistics.mean(values_b),
        'mean_paired_delta_OC_minus_noOC': statistics.mean(differences),
        'median_paired_delta': statistics.median(differences),
        'paired_delta_std_sample': statistics.stdev(differences),
        'improved_count': sum(improves),
        'total_count': len(differences),
    }

json_out = root / f'all32_{kimg}_dualGT_OC_paired_metrics.json'
json_out.write_text(json.dumps({
    'description': 'Old Dual-GT controlled ablation on NEW 3T T1-FLAIR validation, 32 paired slices',
    'checkpoint_kimg': kimg,
    'models': models,
    'settings': {
        'sample_indices': list(range(32)), 'mask_select': 4,
        'steps': 78, 'inner_loops': 10, 'seed': 123,
        'fixed_seed_per_sample': True, 'inference_pad': 64, 'psize': 64,
        'training_pad_width': 96, 'LGFC': False, 'physical_GPU': 2,
        'note': 'Paired comparison within old Dual-GT only; not an isolated SingleGT-vs-DualGT comparison.'
    },
    'summary': summary,
    'per_image': rows,
}, ensure_ascii=False, indent=2), encoding='utf-8')

print('\n============== Dual-GT OC ALL-32 matched evaluation ==============')
print(f"{'Metric':<10}{'OC=0 mean':>15}{'OC=0.5 mean':>17}{'OC - noOC':>16}{'Improved':>12}")
for k in fields:
    s = summary[k]
    print(f"{k.upper():<10}{s['OC0_mean']:15.6f}{s['OC0p5_mean']:17.6f}{s['mean_paired_delta_OC_minus_noOC']:+16.6f}{s['improved_count']:>8}/32")
print('Per-sample CSV:', csv_out)
print('Summary JSON  :', json_out)
print('==================================================================')
PY
}

{
    echo "[START] ${EXPERIMENT_ID}"
    echo "[INFO] Serial 32-sample paired evaluation: 0..31, GPU=2, fixed seed per sample"
    echo "[INFO] One case at a time; safely restart with REUSE_COMPLETE=1 (default)"
    echo "[INFO] Optionally reuse existing sample26 from prior verified run: REUSE_PREVIOUS_S26=${REUSE_PREVIOUS_S26}"
    echo "[INFO] Output: ${RESULT_ROOT}"

    for idx in $(seq 0 31); do
        echo "[PROGRESS] sample ${idx} / 31"
        run_one 'DualGT_OC0' "${GT0_MODEL}" "${idx}"
        run_one 'DualGT_OC0p5' "${GT5_MODEL}" "${idx}"
    done
    summarize_pair
    echo "[SUCCESS] ${EXPERIMENT_ID} completed: ${RESULT_ROOT}"
} 2>&1 | tee "${MASTER_LOG}"
