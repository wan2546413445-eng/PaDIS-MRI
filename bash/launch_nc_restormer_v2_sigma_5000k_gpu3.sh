#!/bin/bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}
LOG_DIR="$RESULT_ROOT/results_record/logs"
mkdir -p "$LOG_DIR"
cd "$CODE_ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)
LAUNCH_LOG="$LOG_DIR/launch_NC_Restormer_V2_Sigma_5000k_gpu3_${STAMP}.log"
PID_FILE="$LOG_DIR/launch_NC_Restormer_V2_Sigma_5000k_gpu3_${STAMP}.pid"

nohup env GPU=3 KIMG=5000 ACTIVATION_CHECKPOINTING=false \
  bash bash/train_opa_nc_restormer_v2_sigma.sh \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
echo "Launched GPU3 Sigma+ 5000kimg | PID=$PID"
echo "Log: $LAUNCH_LOG"
echo "PID: $PID_FILE"
echo "Monitor: tail -f '$LAUNCH_LOG'"
echo "You may disconnect SSH; nohup keeps the process alive."
