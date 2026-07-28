#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/train/padis-mri:${REPO_ROOT}/eval:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ------------------------------

# ------------------------------
GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/baseline_seed123_to20k/brain/32dB/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32-baseline-pad96-seed123-to20k/network-snapshot-010200.pkl}"
VAL_DIR="${VAL_DIR:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp_sample3_equispaced_strict/32dB}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon}"
SAMPLE_INDICES="${SAMPLE_INDICES:-3}"

IMAGE_SIZE="${IMAGE_SIZE:-384}"
PAD="${PAD:-64}"
PSIZE="${PSIZE:-64}"
MASK_SELECT="${MASK_SELECT:-7}"
ZETA="${ZETA:-3.0}"
STEPS="${STEPS:-78}"
INNER_LOOPS="${INNER_LOOPS:-10}"
SEED="${SEED:-123}"

# 只在结果完整时复用，避免重复花时间。设 REUSE_COMPLETE=0 可强制重跑。
REUSE_COMPLETE="${REUSE_COMPLETE:-1}"

checkpoint_stem="$(basename "${MODEL_PATH%.pkl}")"
model_dir="$(basename "$(dirname "${MODEL_PATH}")")"
safe_model_dir="$(printf '%s' "${model_dir}" | tr -cs 'A-Za-z0-9._-' '_')"
EXP_NAME="${EXP_NAME:-${safe_model_dir}_${checkpoint_stem}_sample${SAMPLE_INDICES}_B0_A3}"
SAVE_ROOT="${SAVE_ROOT:-${RESULT_ROOT}/${EXP_NAME}}"

export CUDA_VISIBLE_DEVICES="${GPU}"
LOGICAL_GPU=0

CURRENT_STAGE="startup"
trap 'status=$?; echo; echo "[ERROR] stage=${CURRENT_STAGE}, line=${LINENO}, exit=${status}" >&2; echo "检查日志目录：${SAVE_ROOT}" >&2; exit ${status}' ERR

for variable in MODEL_PATH VAL_DIR SAVE_ROOT SAMPLE_INDICES; do
    if [[ -z "${!variable}" ]]; then
        echo "${variable} 不能为空。" >&2
        exit 2
    fi
done

if [[ "${SAMPLE_INDICES}" == *,* ]] || ! [[ "${SAMPLE_INDICES}" =~ ^[0-9]+$ ]]; then
    echo "SAMPLE_INDICES 必须是单个非负整数；当前值：${SAMPLE_INDICES}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "找不到 checkpoint：${MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -d "${VAL_DIR}" ]]; then
    echo "找不到验证集目录：${VAL_DIR}" >&2
    exit 2
fi
if [[ ! -f "${VAL_DIR}/sample_${SAMPLE_INDICES}.pt" ]]; then
    echo "找不到验证样本：${VAL_DIR}/sample_${SAMPLE_INDICES}.pt" >&2
    exit 2
fi

mkdir -p "${SAVE_ROOT}"

cat <<EOF
============================================================
PaDIS-MRI LGFC paired evaluation: B0 baseline vs A3 EMA
GPU(physical)    : ${GPU}
GPU(logical)     : ${LOGICAL_GPU}
MODEL_PATH       : ${MODEL_PATH}
VAL_DIR          : ${VAL_DIR}
SAMPLE_INDICES   : ${SAMPLE_INDICES}
SAVE_ROOT        : ${SAVE_ROOT}
STEPS            : ${STEPS}
INNER_LOOPS      : ${INNER_LOOPS}
TOTAL_UPDATES    : $((STEPS * INNER_LOOPS))
LGFC             : EMA beta=0.8, weight=0.40, cap=0.08
LGFC schedule    : outer 40-${STEPS}, every step
REUSE_COMPLETE   : ${REUSE_COMPLETE}
============================================================
EOF

python - "${SAVE_ROOT}" "${MODEL_PATH}" "${VAL_DIR}" "${SAMPLE_INDICES}" "${GPU}" <<'PY'
import json
import pathlib
import sys

save_root, model_path, val_dir, sample_idx, gpu = sys.argv[1:]
manifest = {
    "model_path": model_path,
    "val_dir": val_dir,
    "sample_index": int(sample_idx),
    "physical_gpu": int(gpu),
    "experiments": ["B0_baseline", "A3_lgfc_ema_strong"],
    "lgfc": {
        "reference_mode": "ema",
        "ema_beta": 0.8,
        "global_start_outer": 40,
        "global_every": 1,
        "unsampled_weight": 0.40,
        "max_relative_update": 0.08,
        "dc_growth_limit": 1.02,
        "backtrack_factor": 0.5,
        "max_backtracks": 3,
    },
}
path = pathlib.Path(save_root) / "run_manifest.json"
path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

run_front() {
    local stage="$1"
    shift
    CURRENT_STAGE="${stage}"
    echo
    echo "==================== ${stage} ===================="
    "$@" 2>&1 | tee "${SAVE_ROOT}/${stage}.log"
}

stage_complete() {
    local stage_dir="$1"
    [[ -f "${stage_dir}/evaluate/recons/recon_patch_${SAMPLE_INDICES}.npy" ]] \
        && [[ -f "${stage_dir}/evaluate/comp_plots/results.json" ]] \
        && [[ -f "${stage_dir}/runtime_profile.json" ]]
}

CURRENT_STAGE="preflight"
echo
 echo "[Preflight] Python syntax"
python -m py_compile eval_lgfc/*.py
echo "[Preflight] Shell syntax"
bash -n "${BASH_SOURCE[0]}"
echo "[Preflight] LGFC smoke tests"
python -u eval_lgfc/smoke_test_lgfc.py 2>&1 | tee "${SAVE_ROOT}/preflight_smoke.log"

COMMON_ARGS=(
    --model_path "${MODEL_PATH}"
    --val_dir "${VAL_DIR}"
    --image_size "${IMAGE_SIZE}"
    --pad "${PAD}"
    --psize "${PSIZE}"
    --mask_select "${MASK_SELECT}"
    --val_count 1
    --sample_indices "${SAMPLE_INDICES}"
    --zeta "${ZETA}"
    --steps "${STEPS}"
    --inner_loops "${INNER_LOOPS}"
    --seed "${SEED}"
    --gpus "${LOGICAL_GPU}"
    --run_evaluate
)

B0_DIR="${SAVE_ROOT}/B0_baseline"
A3_DIR="${SAVE_ROOT}/A3_lgfc_ema_strong"

if [[ "${REUSE_COMPLETE}" == "1" ]] && stage_complete "${B0_DIR}"; then
    echo
    echo "[B0] 已存在完整结果，跳过：${B0_DIR}"
else
    start_sec="$(date +%s)"
    run_front B0_baseline \
        python -u eval/run.py \
        --algo padis \
        --save_dir "${B0_DIR}" \
        "${COMMON_ARGS[@]}"
    end_sec="$(date +%s)"

    python - "${B0_DIR}" "$((end_sec - start_sec))" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
profile = {
    "total_wall_time_sec": float(sys.argv[2]),
    "run_peak_memory_allocated_mb": None,
    "run_peak_memory_reserved_mb": None,
    "cuda_available": True,
    "profile_device": "cuda:0",
    "note": "B0 uses the original eval/run.py entry; PyTorch process peak memory is not instrumented.",
    "exit_code": 0,
}
(output_dir / "runtime_profile.json").write_text(
    json.dumps(profile, indent=2), encoding="utf-8"
)
PY
fi

if [[ "${REUSE_COMPLETE}" == "1" ]] && stage_complete "${A3_DIR}" \
    && compgen -G "${A3_DIR}/evaluate/lgfc_diagnostics/*_lgfc.csv" >/dev/null; then
    echo
    echo "[A3] 已存在完整结果，跳过：${A3_DIR}"
else
    run_front A3_lgfc_ema_strong \
        python -u eval_lgfc/run_lgfc.py \
        --save_dir "${A3_DIR}" \
        --enable_lgfc \
        --reference_mode ema \
        --ema_beta 0.8 \
        --global_start_outer 40 \
        --global_every 1 \
        --unsampled_weight 0.40 \
        --max_relative_update 0.08 \
        --dc_growth_limit 1.02 \
        --backtrack_factor 0.5 \
        --max_backtracks 3 \
        --save_lgfc_diagnostics \
        --profile_memory \
        "${COMMON_ARGS[@]}"
fi

CURRENT_STAGE="summary"
python -u - "${SAVE_ROOT}" "${SAMPLE_INDICES}" <<'PY' | tee "${SAVE_ROOT}/B0_vs_A3_summary.log"
import csv
import json
import pathlib
import statistics
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])


def load_metrics(dirname):
    path = root / dirname / "evaluate" / "comp_plots" / "results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    per_image = data.get("per_image", {})
    metrics = per_image.get(str(index), per_image.get(index))
    if not metrics:
        metrics = data["summary"]
        return {
            "psnr": metrics["psnr_mean"],
            "ssim": metrics["ssim_mean"],
            "nrmse": metrics["nrmse_mean"],
        }
    return metrics


def load_runtime(dirname):
    path = root / dirname / "runtime_profile.json"
    return json.loads(path.read_text(encoding="utf-8"))


b0 = load_metrics("B0_baseline")
a3 = load_metrics("A3_lgfc_ema_strong")
b0_runtime = load_runtime("B0_baseline")
a3_runtime = load_runtime("A3_lgfc_ema_strong")

diag_files = list(
    (root / "A3_lgfc_ema_strong" / "evaluate" / "lgfc_diagnostics").glob("*_lgfc.csv")
)
if len(diag_files) != 1:
    raise RuntimeError(f"Expected exactly one A3 diagnostic CSV, got: {diag_files}")
with diag_files[0].open(newline="", encoding="utf-8") as handle:
    diagnostics = list(csv.DictReader(handle))

accepted = [r for r in diagnostics if r["correction_accepted"].lower() == "true"]
clipped = [r for r in diagnostics if r["update_was_clipped"].lower() == "true"]
backtracked = [r for r in diagnostics if int(r["backtrack_count"]) > 0]
dc_changes = [float(r["dc_relative_change"]) for r in diagnostics]
scales = [float(r["effective_scale"]) for r in accepted]
raw_relative = [float(r["raw_relative_update"]) for r in diagnostics]
final_update = [float(r["final_state_update_norm"]) for r in diagnostics]

b0_recon = np.load(
    root / "B0_baseline" / "evaluate" / "recons" / f"recon_patch_{index}.npy"
)
a3_recon = np.load(
    root / "A3_lgfc_ema_strong" / "evaluate" / "recons" / f"recon_patch_{index}.npy"
)
if b0_recon.shape != a3_recon.shape:
    raise RuntimeError(f"B0/A3 shape mismatch: {b0_recon.shape} vs {a3_recon.shape}")
diff = np.abs(a3_recon - b0_recon)

summary = {
    "sample_index": index,
    "B0_baseline": b0,
    "A3_lgfc_ema_strong": a3,
    "delta_A3_minus_B0": {
        "psnr_db": float(a3["psnr"] - b0["psnr"]),
        "ssim": float(a3["ssim"] - b0["ssim"]),
        "nrmse": float(a3["nrmse"] - b0["nrmse"]),
    },
    "reconstruction_difference": {
        "max_absolute_difference": float(diff.max(initial=0.0)),
        "mean_absolute_difference": float(diff.mean()),
        "array_equal": bool(np.array_equal(b0_recon, a3_recon)),
    },
    "A3_diagnostics": {
        "attempted_corrections": len(diagnostics),
        "accepted_corrections": len(accepted),
        "skipped_corrections": len(diagnostics) - len(accepted),
        "clipped_corrections": len(clipped),
        "backtracked_corrections": len(backtracked),
        "mean_effective_scale": statistics.fmean(scales) if scales else None,
        "mean_dc_relative_change": statistics.fmean(dc_changes) if dc_changes else None,
        "max_dc_relative_change": max(dc_changes) if dc_changes else None,
        "max_raw_relative_update": max(raw_relative) if raw_relative else None,
        "max_final_state_update_norm": max(final_update) if final_update else None,
    },
    "runtime": {
        "B0_seconds": b0_runtime.get("total_wall_time_sec"),
        "A3_seconds": a3_runtime.get("total_wall_time_sec"),
        "A3_peak_memory_allocated_mb": a3_runtime.get("run_peak_memory_allocated_mb"),
        "A3_peak_memory_reserved_mb": a3_runtime.get("run_peak_memory_reserved_mb"),
    },
}

(root / "B0_vs_A3_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

csv_row = {
    "sample_index": index,
    "B0_PSNR": b0["psnr"],
    "A3_PSNR": a3["psnr"],
    "delta_PSNR": summary["delta_A3_minus_B0"]["psnr_db"],
    "B0_SSIM": b0["ssim"],
    "A3_SSIM": a3["ssim"],
    "delta_SSIM": summary["delta_A3_minus_B0"]["ssim"],
    "B0_NRMSE": b0["nrmse"],
    "A3_NRMSE": a3["nrmse"],
    "delta_NRMSE": summary["delta_A3_minus_B0"]["nrmse"],
    "accepted_corrections": len(accepted),
    "skipped_corrections": len(diagnostics) - len(accepted),
    "clipped_corrections": len(clipped),
    "backtracked_corrections": len(backtracked),
}
with (root / "B0_vs_A3_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(csv_row))
    writer.writeheader()
    writer.writerow(csv_row)

print(json.dumps(summary, indent=2))
PY

CURRENT_STAGE="done"
echo
echo "完成。汇总文件："
echo "  ${SAVE_ROOT}/B0_vs_A3_summary.json"
echo "  ${SAVE_ROOT}/B0_vs_A3_summary.csv"
