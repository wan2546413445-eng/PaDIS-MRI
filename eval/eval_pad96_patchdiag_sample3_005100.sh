#!/bin/bash
set -e
set -o pipefail

GPU=${GPU:-6}
SAMPLE_INDICES=${SAMPLE_INDICES:-"3"}

CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}
RESULT_ROOT=${RESULT_ROOT:-/mnt/SSD2/wsy/PaDIS-MRI}

VAL_DIR=${VAL_DIR:-/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB}
LOG_DIR=$RESULT_ROOT/results_record/logs
mkdir -p "$LOG_DIR"

CKPT_PAD96=${CKPT_PAD96:-/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/checkpoints/00003-aapm_3-uncond-ddpmpp-pedm-gpus1-batch2-fp32/network-snapshot-005100.pkl}

# Patch diagnostic controls.
# Baseline boundary-stat run:
#   PATCH_AGG=hard bash eval/eval_pad96_patchdiag_sample3_005100.sh
# Edge residual attenuation ablation:
#   PATCH_AGG=edge_atten EDGE_SIGMA=0.6 EDGE_FLOOR=0.5 bash eval/eval_pad96_patchdiag_sample3_005100.sh
PATCH_AGG=${PATCH_AGG:-hard}
EDGE_SIGMA=${EDGE_SIGMA:-0.6}
EDGE_FLOOR=${EDGE_FLOOR:-0.5}

BOUNDARY_EVERY=${BOUNDARY_EVERY:-10}
BOUNDARY_WIDTH=${BOUNDARY_WIDTH:-4}
BOUNDARY_FG_THRESHOLD=${BOUNDARY_FG_THRESHOLD:-0.05}

EXP_NAME="patchdiag_pad96_ckpt005100_sample${SAMPLE_INDICES}_seed123_${PATCH_AGG}"
SAVE_DIR=$RESULT_ROOT/PaDIS-MRI-recon/$EXP_NAME
mkdir -p "$SAVE_DIR"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=$LOG_DIR/eval_${EXP_NAME}_gpu${GPU}_${TIME_TAG}.log

echo "==================================================" | tee "$LOG_FILE"
echo "PaDIS-MRI patch diagnostic eval/run_patchdiag.py" | tee -a "$LOG_FILE"
echo "GPU=$GPU" | tee -a "$LOG_FILE"
echo "SAMPLE_INDICES=$SAMPLE_INDICES" | tee -a "$LOG_FILE"
echo "MODEL_PATH=$CKPT_PAD96" | tee -a "$LOG_FILE"
echo "VAL_DIR=$VAL_DIR" | tee -a "$LOG_FILE"
echo "SAVE_DIR=$SAVE_DIR" | tee -a "$LOG_FILE"
echo "STEPS=78" | tee -a "$LOG_FILE"
echo "INNER_LOOPS=10" | tee -a "$LOG_FILE"
echo "PAD=64" | tee -a "$LOG_FILE"
echo "PSIZE=64" | tee -a "$LOG_FILE"
echo "SEED=123" | tee -a "$LOG_FILE"
echo "PATCH_AGG=$PATCH_AGG" | tee -a "$LOG_FILE"
echo "EDGE_SIGMA=$EDGE_SIGMA" | tee -a "$LOG_FILE"
echo "EDGE_FLOOR=$EDGE_FLOOR" | tee -a "$LOG_FILE"
echo "BOUNDARY_EVERY=$BOUNDARY_EVERY" | tee -a "$LOG_FILE"
echo "BOUNDARY_WIDTH=$BOUNDARY_WIDTH" | tee -a "$LOG_FILE"
echo "BOUNDARY_FG_THRESHOLD=$BOUNDARY_FG_THRESHOLD" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

cd "$CODE_ROOT"

export PYTHONPATH="$CODE_ROOT/train/padis-mri:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$GPU python eval/run_patchdiag.py \
    --run_evaluate \
    --algo padis \
    --model_path "$CKPT_PAD96" \
    --val_dir "$VAL_DIR" \
    --image_size 384 \
    --pad 64 \
    --psize 64 \
    --mask_select 7 \
    --val_count 1 \
    --sample_indices "$SAMPLE_INDICES" \
    --seed 123 \
    --zeta 3.0 \
    --steps 78 \
    --inner_loops 10 \
    --report_every 1 \
    --gpus 0 \
    --save_dir "$SAVE_DIR" \
    --boundary_stat \
    --boundary_every "$BOUNDARY_EVERY" \
    --boundary_width "$BOUNDARY_WIDTH" \
    --boundary_fg_threshold "$BOUNDARY_FG_THRESHOLD" \
    --patch_agg "$PATCH_AGG" \
    --edge_sigma "$EDGE_SIGMA" \
    --edge_floor "$EDGE_FLOOR" \
    2>&1 | tee -a "$LOG_FILE"

echo "Done."
echo "SAVE_DIR=$SAVE_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "Boundary CSV should be under: $SAVE_DIR/evaluate/*_boundary_stat.csv"
