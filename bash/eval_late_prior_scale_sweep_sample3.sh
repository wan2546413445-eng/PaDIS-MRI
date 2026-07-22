#!/bin/bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
SAMPLE_INDICES=${SAMPLE_INDICES:-3}
LATE_STEPS=${LATE_STEPS:-10}

GPUS=(0 1 3)
BETAS=(1.0 1.25 1.5)
SAMPLE_INDICES=${SAMPLE_INDICES:-0,3,7,11,15,19,23,27}
LOG_DIR="$RESULT_ROOT/PaDIS-MRI-recon/late_prior_scale/logs"
mkdir -p "$LOG_DIR"

cd "$CODE_ROOT"

for i in "${!BETAS[@]}"; do
  gpu="${GPUS[$i]}"
  beta="${BETAS[$i]}"
  beta_tag=$(echo "$beta" | sed 's/\./p/g')
  log_path="$LOG_DIR/sample${SAMPLE_INDICES}_last${LATE_STEPS}_beta${beta_tag}.log"
  echo "[launch] GPU=$gpu beta=$beta"
  GPU_ID="$gpu" BETA="$beta" LATE_STEPS="$LATE_STEPS" SAMPLE_INDICES="$SAMPLE_INDICES" \
    bash bash/eval_late_prior_scale_sample3.sh > "$log_path" 2>&1 &
done

wait
echo "[done] All beta runs finished."

for beta in "${BETAS[@]}"; do
  beta_tag=$(echo "$beta" | sed 's/\./p/g')
  result_json="$RESULT_ROOT/PaDIS-MRI-recon/late_prior_scale/sample${SAMPLE_INDICES}_last${LATE_STEPS}_beta${beta_tag}/evaluate/comp_plots/results.json"
  echo
  echo "===== beta=$beta ====="
  python - "$result_json" <<'PY'
import json, os, sys
path = sys.argv[1]
if not os.path.isfile(path):
    print("Missing:", path)
else:
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f).get("summary", {})
    print("PSNR :", s.get("psnr_mean"))
    print("SSIM :", s.get("ssim_mean"))
    print("NRMSE:", s.get("nrmse_mean"))
PY
done
