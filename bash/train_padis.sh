# !/bin/bash
# Train PaDIS on FastMRI brain dataset

GPU=0
NPROC=1
ANATOMY=brain                                # brain
SNR=32dB                                     # 32dB SNR (average PSNR in FastMRI brain dataset) – indicates no additional noise added
ROOT_OUTDIR=/home/rohan/padismri-runs        # root directory of where to save model checkpoints
ROOT_DATA=/data/rohan/brain_train_d384_s200  # path to the train dataset
BATCH_SIZE=4                                 # use 4 or 8 depending on GPU memory
LR=1e-4
PADDING=1                                    # use 1 for padding, 0 for no padding    
PAD_WIDTH=96                                 # zero padding width
PATCH_SIZES=16,32,64                         # must be from {512, 256, 128, 96, 64, 56, 48, 32, 24, 16}
PROBS=0.2,0.3,0.5                            # must sum to 1.0 (will be normalized if not)


CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --nproc_per_node=$NPROC train/padis-mri/train.py \
 --outdir=$ROOT_OUTDIR/$ANATOMY/$SNR  --data=$ROOT_DATA/$SNR \
 --cond=0 --arch=ddpmpp --batch=$BATCH_SIZE \
 --lr=$LR --dropout=0.05 \
 --augment=0 --real_p=0.5 --padding=$PADDING \
 --tick=1 --snap=50 --seed=2025 --pad_width=$PAD_WIDTH  \
 --patch-list=$PATCH_SIZES --patch-probs=$PROBS \
