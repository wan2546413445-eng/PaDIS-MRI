#!/bin/bash
GPU=0
export PYTHONPATH=$PWD/train/padis-mri:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$GPU python eval/cross_run.py --algo cross_padis --model_path /path/to/network-snapshot.pkl --val_dir /path/to/val_samples --image_size 384 --pad 96 --psize 64 --mask_select 7 --val_count 1 --sample_indices 0 --zeta 3.0 --steps 10 --inner_loops 1 --cp_k 8 --cp_local_k 3 --cp_global_k 4 --cp_eval_batch_size 2 --save_dir results-cross/debug_cp64_k8_l3_g4
