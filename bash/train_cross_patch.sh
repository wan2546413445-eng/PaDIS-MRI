#!/bin/bash
GPU=0
NPROC=1
ROOT_OUTDIR=/mnt/SSD/wsy/projects/PaDIS-MRI/training-runs-cross
ROOT_DATA=/mnt/SSD/wsy/projects/PaDIS-MRI/data/brain_train_d384_s200
SNR=32dB
CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --nproc_per_node=$NPROC train/padis-mri/cross_train.py --outdir=$ROOT_OUTDIR/cp64_k8_l3_g4_cbase96 --data=$ROOT_DATA/$SNR --cond=0 --arch=ddpmpp --precond=pedm --batch=1 --batch-gpu=1 --cbase=96 --lr=1e-4 --dropout=0.05 --augment=0 --padding=1 --pad_width=96 --tick=1 --snap=50 --seed=2025 --cp_k=8 --cp_local_k=3 --cp_global_k=4 --cp_patch_size=64 --cp_depth=2 --cp_num_heads=4 --fp16=0
