#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-}"
SSIM_MODEL_PATH="${SSIM_MODEL_PATH:-}"
VAL_DIR="${VAL_DIR:-}"
SAVE_ROOT="${SAVE_ROOT:-}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"

for variable in BASELINE_MODEL_PATH SSIM_MODEL_PATH VAL_DIR SAVE_ROOT; do
    if [[ -z "${!variable}" ]]; then
        echo "Set ${variable} before running this script." >&2
        exit 2
    fi
done
if [[ ! -f "${BASELINE_MODEL_PATH}" ]]; then
    echo "Baseline checkpoint not found: ${BASELINE_MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -f "${SSIM_MODEL_PATH}" ]]; then
    echo "SSIM checkpoint not found: ${SSIM_MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -d "${VAL_DIR}" ]]; then
    echo "Validation directory not found: ${VAL_DIR}" >&2
    exit 2
fi
if ! [[ "${SAMPLE_INDICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "SAMPLE_INDICES must be a comma-separated list of non-negative integers." >&2
    exit 2
fi
IFS=',' read -r -a SAMPLE_ARRAY <<< "${SAMPLE_INDICES}"
for sample_index in "${SAMPLE_ARRAY[@]}"; do
    if [[ ! -f "${VAL_DIR}/sample_${sample_index}.pt" ]]; then
        echo "Missing validation sample: ${VAL_DIR}/sample_${sample_index}.pt" >&2
        exit 2
    fi
done

mkdir -p "${SAVE_ROOT}"
export PYTHONPATH="${REPO_ROOT}/train/padis-mri:${REPO_ROOT}:${PYTHONPATH:-}"

COMMON_ARGS=(
    --run_evaluate
    --algo padis
    --val_dir "${VAL_DIR}"
    --image_size 384
    --pad 64
    --psize 64
    --mask_select 7
    --val_count 1
    --sample_indices "${SAMPLE_INDICES}"
    --seed 123
    --fixed_seed_per_sample
    --zeta 3
    --steps 78
    --inner_loops 10
    --report_every 1
    --gpus 0
)

run_model() {
    local run_name="$1"
    local model_path="$2"
    local save_dir="${SAVE_ROOT}/${run_name}"
    mkdir -p "${save_dir}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
        "${PYTHON_BIN}" "${REPO_ROOT}/eval/run.py" \
        --model_path "${model_path}" \
        --save_dir "${save_dir}" \
        "${COMMON_ARGS[@]}" \
        >"${save_dir}/stdout.log" \
        2>"${save_dir}/stderr.log"
}

# 串行运行，确保两个模型使用完全相同的逐样本随机数策略。
run_model B0_baseline "${BASELINE_MODEL_PATH}"
run_model A1_ssim "${SSIM_MODEL_PATH}"

"${PYTHON_BIN}" - "${SAVE_ROOT}" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])


def load_metrics(run_name):
    path = root / run_name / "evaluate" / "comp_plots" / "results.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


baseline = load_metrics("B0_baseline")
ssim_model = load_metrics("A1_ssim")
baseline_per_image = baseline.get("per_image", {})
ssim_per_image = ssim_model.get("per_image", {})
baseline_keys = set(str(key) for key in baseline_per_image)
ssim_keys = set(str(key) for key in ssim_per_image)
if baseline_keys != ssim_keys:
    raise SystemExit(
        "B0/A1 sample mismatch: "
        f"baseline_only={sorted(baseline_keys - ssim_keys)}, "
        f"ssim_only={sorted(ssim_keys - baseline_keys)}"
    )

paired = {}
for key in sorted(baseline_keys, key=int):
    base = baseline_per_image[key]
    auxiliary = ssim_per_image[key]
    paired[key] = {
        "baseline_psnr": float(base["psnr"]),
        "ssim_model_psnr": float(auxiliary["psnr"]),
        "delta_psnr": float(auxiliary["psnr"] - base["psnr"]),
        "baseline_ssim": float(base["ssim"]),
        "ssim_model_ssim": float(auxiliary["ssim"]),
        "delta_ssim": float(auxiliary["ssim"] - base["ssim"]),
        "baseline_nrmse": float(base["nrmse"]),
        "ssim_model_nrmse": float(auxiliary["nrmse"]),
        "delta_nrmse": float(auxiliary["nrmse"] - base["nrmse"]),
    }

if not paired:
    raise SystemExit("No paired metrics were found.")
delta_psnr = [row["delta_psnr"] for row in paired.values()]
delta_ssim = [row["delta_ssim"] for row in paired.values()]
delta_nrmse = [row["delta_nrmse"] for row in paired.values()]
output = {
    "protocol": {
        "comparison": "B0 original baseline vs A1 low-noise magnitude SSIM",
        "fixed_seed_per_sample": True,
        "seed": 123,
        "image_size": 384,
        "pad": 64,
        "psize": 64,
        "mask_select": 7,
        "zeta": 3,
        "steps": 78,
        "inner_loops": 10,
        "total_updates": 780,
        "sample_count": len(paired),
    },
    "per_image": paired,
    "paired_delta_summary": {
        "psnr_mean": statistics.fmean(delta_psnr),
        "psnr_median": statistics.median(delta_psnr),
        "ssim_mean": statistics.fmean(delta_ssim),
        "ssim_median": statistics.median(delta_ssim),
        "nrmse_mean": statistics.fmean(delta_nrmse),
        "nrmse_median": statistics.median(delta_nrmse),
        "psnr_improved_count": sum(value > 0 for value in delta_psnr),
        "ssim_improved_count": sum(value > 0 for value in delta_ssim),
        "nrmse_improved_count": sum(value < 0 for value in delta_nrmse),
    },
}
output_path = root / "paired_results.json"
with output_path.open("w", encoding="utf-8") as handle:
    json.dump(output, handle, indent=2)
print(json.dumps(output, indent=2))
PY
