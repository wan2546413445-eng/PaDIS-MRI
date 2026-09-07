#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODE="${MODE:-smoke}"
GPU="${GPU:-0}"
NPROC="${NPROC:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-}"
OUT_ROOT="${OUT_ROOT:-}"
SSIM_PYCACHE_DIR="${SSIM_PYCACHE_DIR:-${TMPDIR:-/tmp}/padis_ssim_pycache}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT:-${TMPDIR:-/tmp}}/logs}"
TARGET_KIMG_OVERRIDE="${TARGET_KIMG:-}"
SNAP_OVERRIDE="${SNAP:-}"
DUMP_OVERRIDE="${DUMP:-}"
RESUME_STATE="${RESUME_STATE:-}"
if [[ -z "${DATA_DIR}" || ! -f "${DATA_DIR}/noisy.pt" ]]; then
    echo "Set DATA_DIR to the training directory containing noisy.pt." >&2
    exit 2
fi
if [[ -z "${OUT_ROOT}" ]]; then
    echo "Set OUT_ROOT to the SSIM stage-1 training output directory." >&2
    exit 2
fi
if [[ "${NPROC}" != "1" ]]; then
    echo "SSIM stage 1 is fixed to one GPU; set NPROC=1." >&2
    exit 2
fi

if [[ -n "${RESUME_STATE}" && "${MODE}" != "main" ]]; then
    echo "RESUME_STATE is only valid when MODE=main." >&2
    exit 2
fi

case "${MODE}" in
    smoke)
        TARGET_KIMG=1
        RUN_DESC="ssim-stage1-smoke"
        SNAP=0
        DUMP=0
        export WANDB_MODE="${WANDB_MODE:-disabled}"
        ;;
    probe)
        TARGET_KIMG=100
        RUN_DESC="ssim-stage1-probe100kimg"
        SNAP=0
        DUMP=0
        export WANDB_MODE="${WANDB_MODE:-disabled}"
        ;;
    main)
        TARGET_KIMG="${TARGET_KIMG_OVERRIDE:-10000}"
        RUN_DESC="ssim-stage1-main-to${TARGET_KIMG}kimg"

        SNAP="${SNAP_OVERRIDE:-1000}"
        DUMP="${DUMP_OVERRIDE:-2000}"
        ;;
    *)
        echo "MODE must be one of: smoke, probe, main." >&2
        exit 2
        ;;
esac

if ! [[ "${TARGET_KIMG}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TARGET_KIMG must be a positive integer; received: ${TARGET_KIMG}" >&2
    exit 2
fi
if ! [[ "${SNAP}" =~ ^[0-9]+$ ]]; then
    echo "SNAP must be a non-negative integer; received: ${SNAP}" >&2
    exit 2
fi
if ! [[ "${DUMP}" =~ ^[0-9]+$ ]]; then
    echo "DUMP must be a non-negative integer; received: ${DUMP}" >&2
    exit 2
fi
if [[ -n "${RESUME_STATE}" && ! -f "${RESUME_STATE}" ]]; then
    echo "Resume state not found: ${RESUME_STATE}" >&2
    exit 2
fi

DURATION="$(
    awk -v target_kimg="${TARGET_KIMG}" \
        'BEGIN { printf "%.6f", target_kimg / 1000.0 }'
)"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}/train/padis-mri:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="${SSIM_PYCACHE_DIR}"

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
GIT_DIRTY="$(git -C "${REPO_ROOT}" status --short 2>/dev/null || true)"
TIME_TAG="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/train_ssim_${MODE}_${TIME_TAG}.log"

"${PYTHON_BIN}" -m py_compile \
    "${REPO_ROOT}/train/padis-mri/training/patch_ssim_loss.py" \
    "${REPO_ROOT}/train/padis-mri/train_ssim.py" \
    "${REPO_ROOT}/train/padis-mri/ssim_stage1_smoke_test.py"
bash -n "${REPO_ROOT}/bash/train_ssim_stage1.sh"
bash -n "${REPO_ROOT}/bash/eval_ssim_stage1.sh"
"${PYTHON_BIN}" \
    "${REPO_ROOT}/train/padis-mri/ssim_stage1_smoke_test.py" \
    --data "${DATA_DIR}/noisy.pt" \
    --json-out "${OUT_ROOT}/ssim_stage1_smoke_test.json"

{
    echo "=================================================="
    echo "PaDIS-MRI Baseline + Low-noise Magnitude SSIM"
    echo "MODE=${MODE}"
    echo "TARGET_KIMG=${TARGET_KIMG}"
    echo "DURATION_MIMG=${DURATION}"
    echo "GPU=${GPU}"
    echo "DATA_DIR=${DATA_DIR}"
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "GIT_BRANCH=${GIT_BRANCH}"
    echo "GIT_COMMIT=${GIT_COMMIT}"
    echo "GIT_DIRTY=${GIT_DIRTY}"
    echo "P_MEAN=-1.2"
    echo "P_STD=1.2"
    echo "SIGMA_DATA=0.5"
    echo "SSIM_WEIGHT=0.50"
    echo "SSIM_SIGMA_MAX=0.50"
    echo "SNAP=${SNAP}"
    echo "DUMP=${DUMP}"
    echo "RESUME_STATE=${RESUME_STATE:-<none>}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "=================================================="
} | tee "${LOG_FILE}"

RESUME_ARGS=()
if [[ -n "${RESUME_STATE}" ]]; then
    RESUME_ARGS+=(--resume "${RESUME_STATE}")
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" \
torchrun --standalone --nproc_per_node="${NPROC}" \
    train/padis-mri/train_ssim.py \
    --outdir "${OUT_ROOT}" \
    --data "${DATA_DIR}" \
    --duration "${DURATION}" \
    --desc "${RUN_DESC}" \
    --precond pedm \
    --batch 2 \
    --lr 0.0001 \
    --dropout 0.05 \
    --augment 0 \
    --real_p 0.5 \
    --padding True \
    --pad_width 96 \
    --patch-list 16,32,64 \
    --patch-probs 0.2,0.3,0.5 \
    --seed 123 \
    --workers 4 \
    --fp16 False \
    "${RESUME_ARGS[@]}" \
    --ssim-weight 0.50 \
    --ssim-sigma-max 0.50 \
    --ssim-window-size 11 \
    --ssim-window-sigma 1.5 \
    --ssim-k1 0.01 \
    --ssim-k2 0.03 \
    --ssim-data-range 1.0 \
    --magnitude-eps 1e-8 \
    --tick 1 \
    --snap "${SNAP}" \
    --dump "${DUMP}" \
    2>&1 | tee -a "${LOG_FILE}"
