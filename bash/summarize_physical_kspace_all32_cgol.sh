#!/bin/bash
set -e
set -o pipefail

ROOT=${ROOT:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/03_MODULE_TRIALS/overlap/overlap_independent_center_lam0p3_ckpt015120_s78_all32_seed123/evaluate/physical_kspace_error_analysis_all32}

OUT_DIR=${OUT_DIR:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/03_MODULE_TRIALS/overlap/overlap_independent_center_lam0p3_ckpt015120_s78_all32_seed123/evaluate/physical_kspace_error_summary_all32}

mkdir -p "$OUT_DIR"

python eval/summarize_physical_kspace_all32.py \
    --root "$ROOT" \
    --out_dir "$OUT_DIR"

echo "Done."
echo "OUT_DIR=$OUT_DIR"
echo "SUMMARY=$OUT_DIR/physical_kspace_all32_summary.csv"
echo "BAND_MEAN=$OUT_DIR/physical_kspace_all32_band_mean.csv"