#!/bin/bash
set -e
set -o pipefail

GRAD_GPU=${GRAD_GPU:-5}
WAVELET_GPU=${WAVELET_GPU:-6}
CODE_ROOT=${CODE_ROOT:-/mnt/SSD/wsy/projects/PaDIS-MRI-main}

cd "$CODE_ROOT"

nohup env GPU="$GRAD_GPU" \
    /bin/bash bash/train_overlap_gradient_balanced_15k.sh \
    > gradient_balanced_15k_launcher.log 2>&1 &
GRAD_PID=$!

nohup env GPU="$WAVELET_GPU" \
    /bin/bash bash/train_overlap_wavelet_hf_balanced_15k.sh \
    > wavelet_hf_balanced_15k_launcher.log 2>&1 &
WAVELET_PID=$!

echo "Gradient experiment: GPU=$GRAD_GPU PID=$GRAD_PID"
echo "Wavelet experiment:  GPU=$WAVELET_GPU PID=$WAVELET_PID"
echo
echo "Check:"
echo "  nvidia-smi"
echo "  tail -f gradient_balanced_15k_launcher.log"
echo "  tail -f wavelet_hf_balanced_15k_launcher.log"
