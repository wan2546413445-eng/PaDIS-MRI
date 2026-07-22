#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Checkpoints may reference ``training.*`` from train/padis-mri. Keep the
# original repository packages importable for both B0 and LGFC runs.
export PYTHONPATH="${REPO_ROOT}/train/padis-mri:${REPO_ROOT}/eval:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Optional physical-GPU selector. When GPU is set, expose only that device and
# keep the program-facing logical GPU id at 0.
if [[ -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
fi
LOGICAL_GPU=0

MODEL_PATH="${MODEL_PATH:-}"
VAL_DIR="${VAL_DIR:-}"
SAVE_ROOT="${SAVE_ROOT:-}"
SAMPLE_INDICES="${SAMPLE_INDICES:-}"

for variable in MODEL_PATH VAL_DIR SAVE_ROOT SAMPLE_INDICES; do
    if [[ -z "${!variable}" ]]; then
        echo "Set ${variable} before running this script." >&2
        exit 2
    fi
done
if [[ "${SAMPLE_INDICES}" == *,* ]] || ! [[ "${SAMPLE_INDICES}" =~ ^[0-9]+$ ]]; then
    echo "SAMPLE_INDICES must contain exactly one non-negative integer." >&2
    exit 2
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -d "${VAL_DIR}" ]]; then
    echo "VAL_DIR does not exist: ${VAL_DIR}" >&2
    exit 2
fi
if [[ ! -f "${VAL_DIR}/sample_${SAMPLE_INDICES}.pt" ]]; then
    echo "Validation sample does not exist: ${VAL_DIR}/sample_${SAMPLE_INDICES}.pt" >&2
    exit 2
fi

mkdir -p "${SAVE_ROOT}"

# Fail before expensive reconstructions if the isolated code does not import or
# pass its CPU smoke tests. CUDA-specific smoke checks run when CUDA is visible.
python -m py_compile eval_lgfc/*.py \
    >"${SAVE_ROOT}/preflight_pycompile.stdout.log" \
    2>"${SAVE_ROOT}/preflight_pycompile.stderr.log"
bash -n bash/eval_lgfc_stage1.sh \
    >"${SAVE_ROOT}/preflight_shell.stdout.log" \
    2>"${SAVE_ROOT}/preflight_shell.stderr.log"
python eval_lgfc/smoke_test_lgfc.py \
    >"${SAVE_ROOT}/preflight_smoke.stdout.log" \
    2>"${SAVE_ROOT}/preflight_smoke.stderr.log"

COMMON_ARGS=(
    --model_path "${MODEL_PATH}"
    --val_dir "${VAL_DIR}"
    --image_size 384
    --pad 64
    --psize 64
    --mask_select 7
    --val_count 1
    --sample_indices "${SAMPLE_INDICES}"
    --zeta 3
    --steps 78
    --inner_loops 10
    --seed 123
    --gpus "${LOGICAL_GPU}"
    --run_evaluate
)

run_logged() {
    local name="$1"
    shift
    "$@" >"${SAVE_ROOT}/${name}.stdout.log" 2>"${SAVE_ROOT}/${name}.stderr.log"
}

start_sec="$(date +%s)"
set +e
python eval/run.py \
    --algo padis \
    --save_dir "${SAVE_ROOT}/B0_baseline" \
    "${COMMON_ARGS[@]}" \
    >"${SAVE_ROOT}/B0_baseline.stdout.log" \
    2>"${SAVE_ROOT}/B0_baseline.stderr.log"
b0_status=$?
set -e
end_sec="$(date +%s)"
python - "${SAVE_ROOT}/B0_baseline" "$((end_sec - start_sec))" "${b0_status}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
output_dir.mkdir(parents=True, exist_ok=True)
profile = {
    "total_wall_time_sec": float(sys.argv[2]),
    "run_peak_memory_allocated_mb": None,
    "run_peak_memory_reserved_mb": None,
    "cuda_available": None,
    "profile_device": "cuda:0",
    "note": "B0 uses original eval/run.py; process-level PyTorch peak memory is unavailable.",
    "exit_code": int(sys.argv[3]),
}
with (output_dir / "runtime_profile.json").open("w", encoding="utf-8") as handle:
    json.dump(profile, handle, indent=2)
PY
if [[ "${b0_status}" -ne 0 ]]; then
    echo "B0 failed; see B0_baseline.stderr.log" >&2
    exit "${b0_status}"
fi

run_logged B1_bypass \
    python eval_lgfc/run_lgfc.py \
    --save_dir "${SAVE_ROOT}/B1_bypass" \
    --profile_memory \
    "${COMMON_ARGS[@]}"

python - "${SAVE_ROOT}" "${SAMPLE_INDICES}" \
    >"${SAVE_ROOT}/B0_B1_comparison.stdout.log" \
    2>"${SAVE_ROOT}/B0_B1_comparison.stderr.log" <<'PY'
import json
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])
baseline_path = root / "B0_baseline" / "evaluate" / "recons" / f"recon_patch_{index}.npy"
bypass_path = root / "B1_bypass" / "evaluate" / "recons" / f"recon_patch_{index}.npy"
baseline = np.load(baseline_path)
bypass = np.load(bypass_path)
if baseline.shape != bypass.shape:
    raise SystemExit(f"B0/B1 shape mismatch: {baseline.shape} != {bypass.shape}")

difference = np.abs(baseline - bypass)
summary = {
    "sample_index": index,
    "shape": list(baseline.shape),
    "max_absolute_difference": float(difference.max(initial=0.0)),
    "mean_absolute_difference": float(difference.mean()),
    "array_equal": bool(np.array_equal(baseline, bypass)),
    "allclose_atol_1e-7": bool(np.allclose(baseline, bypass, rtol=0.0, atol=1e-7)),
}
for name in ("B0_baseline", "B1_bypass"):
    for metric_name, relative_path in (
        ("original_evaluator", pathlib.Path("evaluate") / "metrics.json"),
        ("author_normalization", pathlib.Path("evaluate") / "comp_plots" / "results.json"),
    ):
        with (root / name / relative_path).open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        per_image = metrics.get("per_image", {})
        summary.setdefault(name, {})[metric_name] = per_image.get(
            str(index), per_image.get(index, {})
        )

output = root / "baseline_bypass_comparison.json"
with output.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
if not summary["allclose_atol_1e-7"]:
    raise SystemExit("B0 and B1 differ; A1/A2/A3 were not started.")
PY

run_logged A1_lgfc_last \
    python eval_lgfc/run_lgfc.py \
    --save_dir "${SAVE_ROOT}/A1_lgfc_last" \
    --enable_lgfc \
    --reference_mode last \
    --global_start_outer 40 \
    --global_every 1 \
    --unsampled_weight 0.25 \
    --max_relative_update 0.05 \
    --dc_growth_limit 1.02 \
    --backtrack_factor 0.5 \
    --max_backtracks 3 \
    --save_lgfc_diagnostics \
    --profile_memory \
    "${COMMON_ARGS[@]}"

run_logged A2_lgfc_ema \
    python eval_lgfc/run_lgfc.py \
    --save_dir "${SAVE_ROOT}/A2_lgfc_ema" \
    --enable_lgfc \
    --reference_mode ema \
    --ema_beta 0.8 \
    --global_start_outer 40 \
    --global_every 1 \
    --unsampled_weight 0.25 \
    --max_relative_update 0.05 \
    --dc_growth_limit 1.02 \
    --backtrack_factor 0.5 \
    --max_backtracks 3 \
    --save_lgfc_diagnostics \
    --profile_memory \
    "${COMMON_ARGS[@]}"

run_logged A3_lgfc_ema_strong \
    python eval_lgfc/run_lgfc.py \
    --save_dir "${SAVE_ROOT}/A3_lgfc_ema_strong" \
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

python - "${SAVE_ROOT}" >"${SAVE_ROOT}/summary.stdout.log" \
    2>"${SAVE_ROOT}/summary.stderr.log" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
specs = [
    ("B0", "B0_baseline", "none", 0.0, 0.0),
    ("B1", "B1_bypass", "none", 0.0, 0.0),
    ("A1", "A1_lgfc_last", "last", 0.25, 0.05),
    ("A2", "A2_lgfc_ema", "ema", 0.25, 0.05),
    ("A3", "A3_lgfc_ema_strong", "ema", 0.40, 0.08),
]
rows = []
for method, dirname, reference_mode, weight, max_relative in specs:
    directory = root / dirname
    with (
        directory / "evaluate" / "comp_plots" / "results.json"
    ).open(encoding="utf-8") as handle:
        metrics = json.load(handle)["summary"]
    with (directory / "runtime_profile.json").open(encoding="utf-8") as handle:
        runtime = json.load(handle)
    diagnostics = []
    diagnostic_dir = directory / "evaluate" / "lgfc_diagnostics"
    if diagnostic_dir.exists():
        files = list(diagnostic_dir.glob("*_lgfc.csv"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one diagnostic CSV in {diagnostic_dir}, got {files}")
        with files[0].open(newline="", encoding="utf-8") as handle:
            diagnostics = list(csv.DictReader(handle))
    accepted = [row for row in diagnostics if row["correction_accepted"].lower() == "true"]
    dc_changes = [float(row["dc_relative_change"]) for row in diagnostics]
    scales = [float(row["effective_scale"]) for row in accepted]
    rows.append({
        "method": method,
        "reference_mode": reference_mode,
        "unsampled_weight": weight,
        "max_relative_update": max_relative,
        "accepted_corrections": len(accepted),
        "skipped_corrections": len(diagnostics) - len(accepted),
        "mean_effective_scale": statistics.fmean(scales) if scales else None,
        "mean_dc_relative_change": statistics.fmean(dc_changes) if dc_changes else None,
        "max_dc_relative_change": max(dc_changes) if dc_changes else None,
        "PSNR": metrics["psnr_mean"],
        "SSIM": metrics["ssim_mean"],
        "NRMSE": metrics["nrmse_mean"],
        "runtime": runtime["total_wall_time_sec"],
        "peak_memory": runtime["run_peak_memory_allocated_mb"],
    })

with (root / "stage1_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (root / "stage1_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(rows, handle, indent=2)
print(json.dumps(rows, indent=2))
PY
