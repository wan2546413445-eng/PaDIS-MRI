#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# AGG-overlap-simplified ckpt010006 old random-mask T1-FLAIR rerun
#
# Stage 1: old random-mask T1-FLAIR, AGG+OC only, 32 samples
# Stage 2: old random-mask T1-FLAIR, AGG+OC + LGFC, 32 samples
# New 3T experiments are already complete and are not rerun here.
#
# Reconstruction budget for every run: 78 x 10 = 780 updates
# LGFC: EMA beta=0.8, start=60, weight=0.60, cap=0.08
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}/train/padis-mri:${REPO_ROOT}/eval:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Physical GPU. CUDA_VISIBLE_DEVICES exposes it internally as logical cuda:0.
GPU="${GPU:-4}"
LOGICAL_GPU=0
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------------------------------------
# Checkpoint: keep the current agg_overlap_simplified checkpoint
# ------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/agg_overlap_simplified_training-runs/brain/32dB/main_agg_overlap_rho0p65_gamma1p0_oc0p3_pad64_seed123/00001-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-agg-overlap-rho0p65-gamma1p0-oc0p3-p64/network-snapshot-010006.pkl}"

# ------------------------------------------------------------
# Validation sets
# ------------------------------------------------------------
OLD_T1FLAIR_VAL="${OLD_T1FLAIR_VAL:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB}"

# ------------------------------------------------------------
# Reconstruction settings: keep unchanged
# ------------------------------------------------------------
IMAGE_SIZE="${IMAGE_SIZE:-384}"
PAD="${PAD:-64}"
PSIZE="${PSIZE:-64}"
MASK_SELECT="${MASK_SELECT:-7}"
ZETA="${ZETA:-3.0}"
STEPS="${STEPS:-78}"
INNER_LOOPS="${INNER_LOOPS:-10}"
SEED="${SEED:-123}"

# ------------------------------------------------------------
# LGFC settings: start=60, weight=0.60
# ------------------------------------------------------------
EMA_BETA="${EMA_BETA:-0.8}"
GLOBAL_START_OUTER="${GLOBAL_START_OUTER:-60}"
GLOBAL_EVERY="${GLOBAL_EVERY:-1}"
UNSAMPLED_WEIGHT="${UNSAMPLED_WEIGHT:-0.60}"
MAX_RELATIVE_UPDATE="${MAX_RELATIVE_UPDATE:-0.08}"
DC_GROWTH_LIMIT="${DC_GROWTH_LIMIT:-1.02}"
BACKTRACK_FACTOR="${BACKTRACK_FACTOR:-0.5}"
MAX_BACKTRACKS="${MAX_BACKTRACKS:-3}"

# ------------------------------------------------------------
# Runtime behavior
# ------------------------------------------------------------
REUSE_COMPLETE="${REUSE_COMPLETE:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
RESULT_ROOT="${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/agg_overlap_simply_ckpt010006_78x10_old_randommask}"
mkdir -p "${RESULT_ROOT}"

FAILURE_FILE="${RESULT_ROOT}/failures.csv"
printf 'stage,sample,exit_code,log\n' > "${FAILURE_FILE}"

CURRENT_STAGE="startup"
trap 'status=$?; echo; echo "[ERROR] stage=${CURRENT_STAGE}, line=${LINENO}, exit=${status}" >&2; echo "结果目录：${RESULT_ROOT}" >&2; exit ${status}' ERR

# ============================================================
# Stage configuration
# ============================================================
STAGE_LABELS=(
    "old_randommask_t1flair_agg_only"
    "old_randommask_t1flair_agg_lgfc"
)

STAGE_VAL_DIRS=(
    "${OLD_T1FLAIR_VAL}"
    "${OLD_T1FLAIR_VAL}"
)

# Number of samples in each stage.
STAGE_COUNTS=(32 32)

# 0 = AGG+OC only; 1 = AGG+OC + LGFC
STAGE_LGFC=(0 1)

# ============================================================
# Preflight
# ============================================================
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "找不到 checkpoint：${MODEL_PATH}" >&2
    exit 2
fi

for i in "${!STAGE_LABELS[@]}"; do
    label="${STAGE_LABELS[i]}"
    val_dir="${STAGE_VAL_DIRS[i]}"
    count="${STAGE_COUNTS[i]}"

    if [[ ! -d "${val_dir}" ]]; then
        echo "找不到验证集目录 [${label}]：${val_dir}" >&2
        exit 2
    fi

    for ((idx=0; idx<count; idx++)); do
        if [[ ! -f "${val_dir}/sample_${idx}.pt" ]]; then
            echo "找不到验证样本 [${label}]：${val_dir}/sample_${idx}.pt" >&2
            exit 2
        fi
    done
done

cat <<EOF_HEADER
============================================================
PaDIS-MRI AGG-overlap-simplified serial evaluation

Physical GPU      : ${GPU}
Logical GPU       : ${LOGICAL_GPU}
Checkpoint        : ${MODEL_PATH}

Image size        : ${IMAGE_SIZE}
Pad / patch       : ${PAD} / ${PSIZE}
Mask select       : ${MASK_SELECT}
Seed              : ${SEED}
Steps             : ${STEPS}
Inner loops       : ${INNER_LOOPS}
Total updates/run : $((STEPS * INNER_LOOPS))

LGFC settings:
  reference       : EMA
  beta            : ${EMA_BETA}
  start outer     : ${GLOBAL_START_OUTER}
  every           : ${GLOBAL_EVERY}
  weight          : ${UNSAMPLED_WEIGHT}
  update cap      : ${MAX_RELATIVE_UPDATE}
  dc growth limit : ${DC_GROWTH_LIMIT}
  backtrack       : ${BACKTRACK_FACTOR}
  max backtracks  : ${MAX_BACKTRACKS}

Stages:
  1. old random-mask T1-FLAIR : AGG only       : 32
  2. old random-mask T1-FLAIR : AGG + LGFC     : 32

Total reconstructions : 64
Result root           : ${RESULT_ROOT}
Reuse complete        : ${REUSE_COMPLETE}
Continue on error     : ${CONTINUE_ON_ERROR}
============================================================
EOF_HEADER

CURRENT_STAGE="preflight"
echo
echo "[Preflight] Python syntax"
python -m py_compile eval_lgfc/*.py

echo "[Preflight] Shell syntax"
bash -n "${BASH_SOURCE[0]}"

echo "[Preflight] LGFC smoke tests"
python -u eval_lgfc/smoke_test_lgfc.py 2>&1 | tee "${RESULT_ROOT}/preflight_smoke.log"

# ============================================================
# Completion check
# ============================================================
stage_complete() {
    local sample_dir="$1"
    local sample_idx="$2"
    local lgfc_enabled="$3"

    [[ -f "${sample_dir}/evaluate/recons/recon_patch_${sample_idx}.npy" ]] \
        && [[ -f "${sample_dir}/evaluate/comp_plots/results.json" ]] \
        && [[ -f "${sample_dir}/runtime_profile.json" ]] \
        || return 1

    if [[ "${lgfc_enabled}" == "1" ]]; then
        compgen -G "${sample_dir}/evaluate/lgfc_diagnostics/*_lgfc.csv" >/dev/null
    fi
}

# ============================================================
# Run one sample
# ============================================================
run_one_sample() {
    local stage_label="$1"
    local val_dir="$2"
    local lgfc_enabled="$3"
    local sample_idx="$4"

    local stage_dir="${RESULT_ROOT}/${stage_label}"
    local sample_dir="${stage_dir}/sample_${sample_idx}"
    local log_file="${sample_dir}/run.log"
    local status=0

    mkdir -p "${sample_dir}"

    if [[ "${REUSE_COMPLETE}" == "1" ]] \
        && stage_complete "${sample_dir}" "${sample_idx}" "${lgfc_enabled}"; then
        echo
        echo "[Reuse] ${stage_label} sample ${sample_idx} 已完成，跳过"
        return 0
    fi

    CURRENT_STAGE="${stage_label}_sample_${sample_idx}"

    echo
    echo "============================================================"
    echo "Stage  : ${stage_label}"
    echo "LGFC   : ${lgfc_enabled}"
    echo "Sample : ${sample_idx}"
    echo "Val    : ${val_dir}"
    echo "Output : ${sample_dir}"
    echo "============================================================"

    cmd=(
        python -u eval_lgfc/run_lgfc.py
        --model_path "${MODEL_PATH}"
        --val_dir "${val_dir}"
        --image_size "${IMAGE_SIZE}"
        --pad "${PAD}"
        --psize "${PSIZE}"
        --mask_select "${MASK_SELECT}"
        --val_count 1
        --sample_indices "${sample_idx}"
        --zeta "${ZETA}"
        --steps "${STEPS}"
        --inner_loops "${INNER_LOOPS}"
        --seed "${SEED}"
        --gpus "${LOGICAL_GPU}"
        --run_evaluate
        --save_dir "${sample_dir}"
        --profile_memory
    )

    if [[ "${lgfc_enabled}" == "1" ]]; then
        cmd+=(
            --enable_lgfc
            --reference_mode ema
            --ema_beta "${EMA_BETA}"
            --global_start_outer "${GLOBAL_START_OUTER}"
            --global_every "${GLOBAL_EVERY}"
            --unsampled_weight "${UNSAMPLED_WEIGHT}"
            --max_relative_update "${MAX_RELATIVE_UPDATE}"
            --dc_growth_limit "${DC_GROWTH_LIMIT}"
            --backtrack_factor "${BACKTRACK_FACTOR}"
            --max_backtracks "${MAX_BACKTRACKS}"
            --save_lgfc_diagnostics
        )
    fi

    if "${cmd[@]}" 2>&1 | tee "${log_file}"; then
        if stage_complete "${sample_dir}" "${sample_idx}" "${lgfc_enabled}"; then
            echo "[Done] ${stage_label} sample ${sample_idx}"
            return 0
        fi

        status=99
        printf '%s,%s,%s,%s\n' \
            "${stage_label}" \
            "${sample_idx}" \
            "${status}" \
            "${log_file}" \
            >> "${FAILURE_FILE}"

        echo "[Failed] ${stage_label} sample ${sample_idx}: command exited 0 but outputs are incomplete" >&2

        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
            return 0
        fi
        return "${status}"
    else
        status="${PIPESTATUS[0]}"
        printf '%s,%s,%s,%s\n' \
            "${stage_label}" \
            "${sample_idx}" \
            "${status}" \
            "${log_file}" \
            >> "${FAILURE_FILE}"

        echo "[Failed] ${stage_label} sample ${sample_idx}, exit=${status}" >&2

        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
            return 0
        fi
        return "${status}"
    fi
}

# ============================================================
# Aggregate a stage
# ============================================================
aggregate_stage() {
    local stage_label="$1"
    local count="$2"
    local lgfc_enabled="$3"

    CURRENT_STAGE="aggregate_${stage_label}"

    python -u - "${RESULT_ROOT}" "${stage_label}" "${count}" "${lgfc_enabled}" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
label = sys.argv[2]
count = int(sys.argv[3])
lgfc_enabled = bool(int(sys.argv[4]))
rows = []

for idx in range(count):
    sample_dir = root / label / f"sample_{idx}"
    metrics_path = sample_dir / "evaluate" / "comp_plots" / "results.json"
    runtime_path = sample_dir / "runtime_profile.json"
    diag_dir = sample_dir / "evaluate" / "lgfc_diagnostics"

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

    runtime_sec = None
    peak_memory = None
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime_sec = runtime.get("total_wall_time_sec")
        peak_memory = runtime.get("run_peak_memory_allocated_mb")

    diagnostics = []
    if lgfc_enabled and diag_dir.exists():
        diag_files = list(diag_dir.glob("*_lgfc.csv"))
        if len(diag_files) == 1:
            with diag_files[0].open(newline="", encoding="utf-8") as handle:
                diagnostics = list(csv.DictReader(handle))

    accepted = [r for r in diagnostics if r["correction_accepted"].lower() == "true"]
    clipped = [r for r in diagnostics if r["update_was_clipped"].lower() == "true"]
    backtracked = [r for r in diagnostics if int(r["backtrack_count"]) > 0]
    dc_changes = [float(r["dc_relative_change"]) for r in diagnostics]

    rows.append({
        "sample_index": idx,
        "PSNR": float(metrics["psnr"]),
        "SSIM": float(metrics["ssim"]),
        "NRMSE": float(metrics["nrmse"]),
        "runtime_sec": runtime_sec,
        "peak_memory_allocated_mb": peak_memory,
        "attempted_corrections": len(diagnostics),
        "accepted_corrections": len(accepted),
        "skipped_corrections": len(diagnostics) - len(accepted),
        "clipped_corrections": len(clipped),
        "backtracked_corrections": len(backtracked),
        "mean_dc_relative_change": statistics.fmean(dc_changes) if dc_changes else None,
    })

out_dir = root / label
out_dir.mkdir(parents=True, exist_ok=True)

if not rows:
    print(json.dumps({"stage": label, "completed_samples": 0}, indent=2))
    raise SystemExit(0)

csv_path = out_dir / "per_sample_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def stat(key):
    vals = [float(r[key]) for r in rows]
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals),
        "min": min(vals),
        "max": max(vals),
    }

runtime_values = [float(r["runtime_sec"]) for r in rows if r["runtime_sec"] is not None]
memory_values = [float(r["peak_memory_allocated_mb"]) for r in rows if r["peak_memory_allocated_mb"] is not None]

summary = {
    "stage": label,
    "lgfc_enabled": lgfc_enabled,
    "completed_samples": len(rows),
    "requested_samples": count,
    "PSNR": stat("PSNR"),
    "SSIM": stat("SSIM"),
    "NRMSE": stat("NRMSE"),
    "total_runtime_sec": sum(runtime_values) if runtime_values else None,
    "mean_runtime_sec": statistics.fmean(runtime_values) if runtime_values else None,
    "max_peak_memory_allocated_mb": max(memory_values) if memory_values else None,
    "total_attempted_corrections": sum(r["attempted_corrections"] for r in rows),
    "total_accepted_corrections": sum(r["accepted_corrections"] for r in rows),
    "total_skipped_corrections": sum(r["skipped_corrections"] for r in rows),
    "total_clipped_corrections": sum(r["clipped_corrections"] for r in rows),
    "total_backtracked_corrections": sum(r["backtracked_corrections"] for r in rows),
}

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
}

# ============================================================
# Run all stages serially
# ============================================================
for i in "${!STAGE_LABELS[@]}"; do
    label="${STAGE_LABELS[i]}"
    val_dir="${STAGE_VAL_DIRS[i]}"
    count="${STAGE_COUNTS[i]}"
    lgfc_enabled="${STAGE_LGFC[i]}"

    mkdir -p "${RESULT_ROOT}/${label}"

    echo
    echo "############################################################"
    echo "Starting stage : ${label}"
    echo "Validation     : ${val_dir}"
    echo "Samples        : ${count}"
    echo "LGFC           : ${lgfc_enabled}"
    echo "############################################################"

    for ((sample_idx=0; sample_idx<count; sample_idx++)); do
        run_one_sample "${label}" "${val_dir}" "${lgfc_enabled}" "${sample_idx}"
    done

    aggregate_stage "${label}" "${count}" "${lgfc_enabled}" \
        | tee "${RESULT_ROOT}/${label}/summary.log"
done

# ============================================================
# Paired comparison for old random-mask T1-FLAIR
# ============================================================
CURRENT_STAGE="paired_summary"
python -u - "${RESULT_ROOT}" <<'PY' | tee "${RESULT_ROOT}/paired_summary.log"
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])

pairs = {
    "old_randommask_t1flair": (
        "old_randommask_t1flair_agg_only",
        "old_randommask_t1flair_agg_lgfc",
    ),
}

all_summary = {}

for pair_name, (base_label, lgfc_label) in pairs.items():
    def load(label):
        path = root / label / "per_sample_results.csv"
        if not path.exists():
            return {}
        with path.open(newline="", encoding="utf-8") as handle:
            return {int(r["sample_index"]): r for r in csv.DictReader(handle)}

    base = load(base_label)
    lgfc = load(lgfc_label)
    common = sorted(set(base) & set(lgfc))
    rows = []

    for idx in common:
        b = base[idx]
        l = lgfc[idx]
        rows.append({
            "sample_index": idx,
            "agg_PSNR": float(b["PSNR"]),
            "agg_lgfc_PSNR": float(l["PSNR"]),
            "delta_PSNR": float(l["PSNR"]) - float(b["PSNR"]),
            "agg_SSIM": float(b["SSIM"]),
            "agg_lgfc_SSIM": float(l["SSIM"]),
            "delta_SSIM": float(l["SSIM"]) - float(b["SSIM"]),
            "agg_NRMSE": float(b["NRMSE"]),
            "agg_lgfc_NRMSE": float(l["NRMSE"]),
            "delta_NRMSE": float(l["NRMSE"]) - float(b["NRMSE"]),
        })

    out_csv = root / f"{pair_name}_paired_results.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        def stat(key):
            vals = [float(r[key]) for r in rows]
            return {
                "mean": statistics.fmean(vals),
                "std": statistics.pstdev(vals),
                "min": min(vals),
                "max": max(vals),
            }

        all_summary[pair_name] = {
            "paired_samples": len(rows),
            "delta_lgfc_minus_agg": {
                "PSNR": stat("delta_PSNR"),
                "SSIM": stat("delta_SSIM"),
                "NRMSE": stat("delta_NRMSE"),
            },
            "win_counts": {
                "PSNR_higher": sum(r["delta_PSNR"] > 0 for r in rows),
                "SSIM_higher": sum(r["delta_SSIM"] > 0 for r in rows),
                "NRMSE_lower": sum(r["delta_NRMSE"] < 0 for r in rows),
            },
            "paired_csv": str(out_csv),
        }
    else:
        all_summary[pair_name] = {"paired_samples": 0}

(root / "paired_summary.json").write_text(
    json.dumps(all_summary, indent=2), encoding="utf-8"
)
print(json.dumps(all_summary, indent=2))
PY

# ============================================================
# Finished
# ============================================================
CURRENT_STAGE="done"
failed_count=$(( $(wc -l < "${FAILURE_FILE}") - 1 ))

echo
echo "============================================================"
echo "全部串行任务结束"
echo "Checkpoint : ${MODEL_PATH}"
echo "Result root: ${RESULT_ROOT}"
echo "失败任务数：${failed_count}"
echo
echo "Stage summaries:"
for label in "${STAGE_LABELS[@]}"; do
    echo "  ${RESULT_ROOT}/${label}/summary.json"
done
echo
echo "Paired summaries:"
echo "  ${RESULT_ROOT}/paired_summary.json"
echo "  ${RESULT_ROOT}/old_randommask_t1flair_paired_results.csv"
echo
echo "Failures:"
echo "  ${FAILURE_FILE}"
echo "============================================================"

if (( failed_count > 0 )); then
    exit 1
fi
