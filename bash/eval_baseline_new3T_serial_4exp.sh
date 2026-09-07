#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# PaDIS-MRI baseline serial evaluation on NEW 3T validation sets
#
# Stage 1: new 3T T1-FLAIR, 32 samples, 78 x 10
# Stage 2: new 3T T2,       50 samples, 78 x 10
# Stage 3: new 3T T2,       50 samples, 104 x 10
#
# No LGFC. Same baseline checkpoint for all three stages.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train/padis-mri:${REPO_ROOT}/eval${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# First serial script is using physical GPU 1, so this companion script
# defaults to physical GPU 0. Override with GPU=<id> if needed.
GPU="${GPU:-3}"
LOGICAL_GPU=0
export CUDA_VISIBLE_DEVICES="${GPU}"

# IMPORTANT:
# eval/run.py expects the network snapshot .pkl, not training-state-010200.pt.
MODEL_PATH="${MODEL_PATH:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/baseline_seed123_to20k/brain/32dB/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-baseline-pad96-seed123-to20k/network-snapshot-010200.pkl}"

VAL_T1FLAIR="${VAL_T1FLAIR:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB}"
VAL_T2="${VAL_T2:-/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t2/32dB}"

RESULT_ROOT="${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/BASELINE_new3T_serial}"

IMAGE_SIZE="${IMAGE_SIZE:-384}"
PAD="${PAD:-64}"
PSIZE="${PSIZE:-64}"
MASK_SELECT="${MASK_SELECT:-7}"
ZETA="${ZETA:-3.0}"
INNER_LOOPS="${INNER_LOOPS:-10}"
SEED="${SEED:-123}"

REUSE_COMPLETE="${REUSE_COMPLETE:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

mkdir -p "${RESULT_ROOT}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "找不到 baseline network snapshot：${MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -d "${VAL_T1FLAIR}" ]]; then
    echo "找不到 T1-FLAIR 验证集：${VAL_T1FLAIR}" >&2
    exit 2
fi
if [[ ! -d "${VAL_T2}" ]]; then
    echo "找不到 T2 验证集：${VAL_T2}" >&2
    exit 2
fi

# ------------------------------------------------------------
# Experiment definitions
# ------------------------------------------------------------
EXP_LABELS=(
    "new3T_T1FLAIR_all32_s78x10"
    "new3T_T1FLAIR_all32_s104x10"
    "new3T_T2_all50_s78x10"
    "new3T_T2_all50_s104x10"
)

EXP_VAL_DIRS=(
    "${VAL_T1FLAIR}"
    "${VAL_T1FLAIR}"
    "${VAL_T2}"
    "${VAL_T2}"
)

EXP_STEPS=(
    "78"
    "104"
    "78"
    "104"
)

EXP_COUNTS=(
    "32"
    "32"
    "50"
    "50"
)

CURRENT_STAGE="startup"
trap 'status=$?; echo; echo "[ERROR] stage=${CURRENT_STAGE}, line=${LINENO}, exit=${status}" >&2; echo "结果根目录：${RESULT_ROOT}" >&2; exit ${status}' ERR

stage_complete() {
    local sample_dir="$1"
    local sample_idx="$2"

    [[ -f "${sample_dir}/evaluate/recons/recon_patch_${sample_idx}.npy" ]] \
        && [[ -f "${sample_dir}/evaluate/comp_plots/results.json" ]]
}

run_one_sample() {
    local exp_label="$1"
    local val_dir="$2"
    local steps="$3"
    local sample_idx="$4"
    local exp_dir="${RESULT_ROOT}/${exp_label}"
    local sample_dir="${exp_dir}/sample_${sample_idx}"
    local log_file="${sample_dir}/run.log"

    mkdir -p "${sample_dir}"

    if [[ "${REUSE_COMPLETE}" == "1" ]] && stage_complete "${sample_dir}" "${sample_idx}"; then
        echo "[Reuse] ${exp_label} sample ${sample_idx} 已完成，跳过"
        return 0
    fi

    CURRENT_STAGE="${exp_label}_sample_${sample_idx}"

    echo
    echo "============================================================"
    echo "Experiment : ${exp_label}"
    echo "Sample     : ${sample_idx}"
    echo "Steps      : ${steps}"
    echo "Updates    : $((steps * INNER_LOOPS))"
    echo "Output     : ${sample_dir}"
    echo "============================================================"

    if python -u eval/run.py \
        --run_evaluate \
        --algo padis \
        --model_path "${MODEL_PATH}" \
        --val_dir "${val_dir}" \
        --image_size "${IMAGE_SIZE}" \
        --pad "${PAD}" \
        --psize "${PSIZE}" \
        --mask_select "${MASK_SELECT}" \
        --val_count 1 \
        --sample_indices "${sample_idx}" \
        --seed "${SEED}" \
        --fixed_seed_per_sample \
        --zeta "${ZETA}" \
        --steps "${steps}" \
        --inner_loops "${INNER_LOOPS}" \
        --report_every 1 \
        --save_dir "${sample_dir}" \
        --gpus "${LOGICAL_GPU}" \
        2>&1 | tee "${log_file}"
    then
        echo "[Done] ${exp_label} sample ${sample_idx}"
        return 0
    else
        local status="${PIPESTATUS[0]}"
        printf '%s,%s,%s,%s\n' "${exp_label}" "${sample_idx}" "${status}" "${log_file}" >> "${RESULT_ROOT}/failures.csv"
        echo "[Failed] ${exp_label} sample ${sample_idx}, exit=${status}" >&2
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
            return 0
        fi
        return "${status}"
    fi
}

aggregate_experiment() {
    local exp_label="$1"
    local sample_count="$2"
    CURRENT_STAGE="aggregate_${exp_label}"

    python -u - "${RESULT_ROOT}/${exp_label}" "${sample_count}" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
count = int(sys.argv[2])
rows = []

for idx in range(count):
    metrics_path = root / f"sample_{idx}" / "evaluate" / "comp_plots" / "results.json"
    if not metrics_path.exists():
        continue

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_image = data.get("per_image", {})
    metrics = per_image.get(str(idx))

    if not metrics:
        summary = data.get("summary", {})
        if not summary:
            continue
        metrics = {
            "psnr": summary["psnr_mean"],
            "ssim": summary["ssim_mean"],
            "nrmse": summary["nrmse_mean"],
        }

    rows.append({
        "sample_index": idx,
        "PSNR": float(metrics["psnr"]),
        "SSIM": float(metrics["ssim"]),
        "NRMSE": float(metrics["nrmse"]),
    })

root.mkdir(parents=True, exist_ok=True)

if not rows:
    print(json.dumps({"completed_samples": 0}, indent=2))
    raise SystemExit(0)

with (root / "per_sample_results.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def stats(key):
    vals = [r[key] for r in rows]
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals),
        "min": min(vals),
        "max": max(vals),
    }

summary = {
    "completed_samples": len(rows),
    "requested_samples": count,
    "PSNR": stats("PSNR"),
    "SSIM": stats("SSIM"),
    "NRMSE": stats("NRMSE"),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
}

printf 'experiment,sample,exit_code,log\n' > "${RESULT_ROOT}/failures.csv"

cat <<EOF
============================================================
PaDIS-MRI baseline NEW-3T serial evaluation
Physical GPU : ${GPU}
Logical GPU  : ${LOGICAL_GPU}
Checkpoint   : ${MODEL_PATH}

Common settings:
  image_size       = ${IMAGE_SIZE}
  pad / psize      = ${PAD} / ${PSIZE}
  mask_select      = ${MASK_SELECT}
  zeta             = ${ZETA}
  inner_loops      = ${INNER_LOOPS}
  seed             = ${SEED}
  RNG              = fixed_seed_per_sample

Experiments:
  1) new 3T T1-FLAIR: 32 samples, 78 x 10
  2) new 3T T1-FLAIR: 32 samples, 104 x 10
  3) new 3T T2      : 50 samples, 78 x 10
  4) new 3T T2      : 50 samples, 104 x 10

Result root:
  ${RESULT_ROOT}
============================================================
EOF

for exp_idx in "${!EXP_LABELS[@]}"; do
    label="${EXP_LABELS[exp_idx]}"
    val_dir="${EXP_VAL_DIRS[exp_idx]}"
    steps="${EXP_STEPS[exp_idx]}"
    count="${EXP_COUNTS[exp_idx]}"

    mkdir -p "${RESULT_ROOT}/${label}"

    echo
    echo "############################################################"
    echo "Starting: ${label}"
    echo "Validation: ${val_dir}"
    echo "Samples: ${count}"
    echo "Steps: ${steps}"
    echo "############################################################"

    for ((idx=0; idx<count; idx++)); do
        if [[ ! -f "${val_dir}/sample_${idx}.pt" ]]; then
            echo "找不到验证样本：${val_dir}/sample_${idx}.pt" >&2
            exit 2
        fi
        run_one_sample "${label}" "${val_dir}" "${steps}" "${idx}"
    done

    aggregate_experiment "${label}" "${count}" | tee "${RESULT_ROOT}/${label}/summary.log"
done

CURRENT_STAGE="done"
failed_count=$(( $(wc -l < "${RESULT_ROOT}/failures.csv") - 1 ))

echo
echo "============================================================"
echo "Baseline 四组串行实验结束"
echo "结果目录：${RESULT_ROOT}"
echo "失败任务数：${failed_count}"
echo
for label in "${EXP_LABELS[@]}"; do
    echo "  ${RESULT_ROOT}/${label}/summary.json"
    echo "  ${RESULT_ROOT}/${label}/per_sample_results.csv"
done
echo
echo "失败清单：${RESULT_ROOT}/failures.csv"
echo "============================================================"

if (( failed_count > 0 )); then
    exit 1
fi
