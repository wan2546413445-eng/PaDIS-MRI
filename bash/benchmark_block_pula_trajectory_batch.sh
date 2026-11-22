#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${MODE:-quick}
OUT_ROOT=${OUT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/Block_pULA_OPA1p25_10k_S10_batch_benchmark_${MODE}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT_ROOT"
OUT_ROOT=$(cd "$OUT_ROOT" && pwd)
SUMMARY="$OUT_ROOT/benchmark_status.tsv"
if [[ -e "$SUMMARY" ]]; then echo "Existing benchmark at $SUMMARY; use a new OUT_ROOT" >&2; exit 1; fi
printf 'batch_size\tstatus\texit_code\toutput\n' > "$SUMMARY"
for size in 1 2 4 8 10; do
    if OUT_ROOT="$OUT_ROOT/tb$size" TRAJECTORY_BATCH_SIZE="$size" BENCHMARK_MODE="$MODE" \
        bash "$ROOT/bash/eval_block_pula_author_N60K4_S10_T1FLAIR.sh"; then
        printf '%s\tcomplete\t0\t%s\n' "$size" "$OUT_ROOT/tb$size" >> "$SUMMARY"
    else
        code=$?
        printf '%s\tfailed\t%s\t%s\n' "$size" "$code" "$OUT_ROOT/tb$size" >> "$SUMMARY"
        if [[ "$code" == 42 ]]; then
            echo "CUDA OOM at batch=$size. Preserved all outputs; stopping larger batches. Summary: $SUMMARY"
            break
        fi
        echo "Non-OOM failure; see $OUT_ROOT/tb$size/run.log. Summary: $SUMMARY" >&2
        exit "$code"
    fi
done
echo "Benchmark summary: $SUMMARY"
