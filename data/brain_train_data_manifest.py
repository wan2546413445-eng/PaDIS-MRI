#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_train_data_manifest.py

用途：
    从 fastMRI multicoil h5 生成 PaDIS-MRI 训练 noisy.pt。
    支持两种模式：
        1. 原始随机抽样模式：--manifest_csv 不传
        2. manifest 固定样本模式：--manifest_csv mixed_center320_qc_selected.csv

推荐实验：
    用同一份 mixed manifest 生成两套训练数据：
        A_new: imsize=384，用训练时 pad_width=96
        B_new: imsize=320，用训练时 pad_width=64

示例 d320：
python data/brain_train_data_manifest.py \
  --h5_folder /mnt/public/成像组/dataset/fast_MRI/multicoil_brain/brain_multicoil_train_batch_0/multicoil_train \
  --output_root /mnt/SSD/wsy/data/fastmri_train_batch0_pilot_mixedQC/ \
  --manifest_csv /mnt/SSD/wsy/data/fastmri_train_batch0_pilot_center320_manifest/mixed_center320_qc_selected.csv \
  --noise_level 32dB \
  --acs_size 24 \
  --imsize 320 \
  --nproc 20

示例 d384：
python data/brain_train_data_manifest.py \
  --h5_folder /mnt/public/成像组/dataset/fast_MRI/multicoil_brain/brain_multicoil_train_batch_0/multicoil_train \
  --output_root /mnt/SSD/wsy/data/fastmri_train_batch0_pilot_mixedQC/ \
  --manifest_csv /mnt/SSD/wsy/data/fastmri_train_batch0_pilot_center320_manifest/mixed_center320_qc_selected.csv \
  --noise_level 32dB \
  --acs_size 24 \
  --imsize 384 \
  --nproc 20
"""

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import shutil
import argparse
from multiprocessing import Pool

import numpy as np
import h5py
import sigpy as sp
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dnnlib.util import configure_bart
configure_bart()

from bart import bart
from data_utils import normalization_const, tqdm


parser = argparse.ArgumentParser(description="Process MRI volumes for PaDIS-MRI training set.")
parser.add_argument('--max_volumes', type=int, default=200, help='Maximum number of volumes to process in random mode')
parser.add_argument('--num_slices', type=int, default=1, help='Number of slices per volume in random mode')
parser.add_argument('--h5_folder', type=str, required=True, help='Path to input folder containing .h5 files')
parser.add_argument('--output_root', type=str, default="/data/datasets/fastmri/", help='Path to output folder')
parser.add_argument('--random_seed', type=int, default=42, help='Seed for random sampling mode')
parser.add_argument('--noise_level', type=str, default="32dB", choices=["32dB", "22dB", "12dB"], help='Noise level')
parser.add_argument('--nproc', type=int, default=30, help='Number of CPU workers')
parser.add_argument('--acs_size', type=int, default=24, help='Number of ACS lines')
parser.add_argument('--imsize', type=int, default=320, help='Output image size, e.g. 320 or 384')
parser.add_argument('--manifest_csv', type=str, default=None, help='CSV with selected,file,slice columns')
parser.add_argument('--center_slice', type=int, default=2, help='Original PaDIS random-mode fixed slice index')
parser.add_argument('--out_name', type=str, default=None, help='Optional output folder name, e.g. brain_train_d320_s200_mixedQC')
args = parser.parse_args()

n_proc = args.nproc
num_slices = args.num_slices
center_slice = args.center_slice
ACS_size = args.acs_size
imsize = args.imsize
db = args.noise_level

if db == "32dB":
    noise_amp = np.sqrt(0)
elif db == "22dB":
    noise_amp = np.sqrt(10)
elif db == "12dB":
    noise_amp = np.sqrt(100)
else:
    raise ValueError(f"Unsupported db level: {db}")

h5_folder = args.h5_folder
ksp_files = sorted([os.path.join(h5_folder, f) for f in os.listdir(h5_folder) if f.endswith(".h5")])
if not ksp_files:
    raise FileNotFoundError(f"No .h5 files found in {h5_folder}")
print(f"Found {len(ksp_files)} .h5 files in {h5_folder}")

manifest_items = None

if args.manifest_csv is not None:
    manifest_items = []
    with open(args.manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "selected" in row and str(row["selected"]) not in ["1", "1.0", "True", "true"]:
                continue
            h5_path = row.get("file", None)
            slice_idx = row.get("slice", None)
            if h5_path is None or slice_idx is None:
                raise KeyError("manifest_csv must contain columns 'file' and 'slice'.")
            manifest_items.append((h5_path, int(float(slice_idx))))

    if len(manifest_items) == 0:
        raise RuntimeError(f"No selected samples found in manifest: {args.manifest_csv}")

    total_iterations = len(manifest_items)
    indexes = list(range(total_iterations))
    print(f"Using manifest: {args.manifest_csv}")
    print(f"Manifest samples: {total_iterations}")

else:
    max_volumes = args.max_volumes if args.max_volumes < len(ksp_files) else len(ksp_files)
    total_iterations = max_volumes * num_slices
    all_possible = list(range(len(ksp_files) * num_slices))
    rng = np.random.default_rng(seed=args.random_seed)
    indexes = rng.choice(all_possible, size=total_iterations, replace=False).tolist()
    print(f"Using random sampling mode: {total_iterations} samples")

x_est_gt = torch.zeros(total_iterations, imsize, imsize, dtype=torch.complex64)
x_est = torch.zeros(total_iterations, imsize, imsize, dtype=torch.complex64)
u_images = torch.zeros(total_iterations, imsize, imsize, dtype=torch.complex64)
norm_consts_99 = torch.zeros(total_iterations, dtype=torch.float32)
noise_var_noisy = torch.zeros(total_iterations, dtype=torch.float32)

if args.out_name is not None:
    dataset_name = args.out_name
else:
    dataset_name = f"brain_train_d{imsize}_s{total_iterations}"

path = os.path.join(args.output_root, dataset_name, db)
os.makedirs(os.path.join(path, "ksp"), exist_ok=True)
print("Output path:", path)


def _process_one(i):
    idx = indexes[i]

    if manifest_items is not None:
        h5_path, slice_idx = manifest_items[idx]
    else:
        sample_idx = idx // num_slices
        slice_idx = center_slice + np.mod(idx, num_slices) - num_slices // 2
        h5_path = ksp_files[sample_idx]

    with h5py.File(h5_path, 'r') as contents:
        nslices = contents['kspace'].shape[0]
        if slice_idx < 0 or slice_idx >= nslices:
            raise IndexError(f"{h5_path}: slice_idx={slice_idx} out of range [0,{nslices})")
        ksp = np.asarray(contents['kspace'][slice_idx]).transpose(1, 2, 0)

    cimg = bart(1, 'fft -iu 3', ksp)
    cimg = sp.resize(cimg, [396, cimg.shape[1], cimg.shape[2]])

    # Use corner region to estimate coil noise covariance.
    noise = cimg[0:30, 0:30]
    noise_flat = np.reshape(noise, (-1, cimg.shape[2]))

    cimg_white = sp.resize(
        bart(1, 'whiten', cimg[:, :, None, :], noise_flat[:, None, None, :]).squeeze(),
        [imsize, imsize, cimg.shape[2]]
    )

    cimg_white_noisy = cimg_white + (noise_amp / np.sqrt(2)) * (
        np.random.normal(size=cimg_white.shape) + 1j * np.random.normal(size=cimg_white.shape)
    )

    ksp_white = bart(1, 'fft -u 3', cimg_white)
    ksp_white_noisy = bart(1, 'fft -u 3', cimg_white_noisy)

    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white[:, :, None, :]).squeeze()
    s_maps_white_noisy = bart(1, 'ecalib -m 1 -c0', ksp_white_noisy[:, :, None, :]).squeeze()

    gt_img_white_cropped = bart(1, 'pics -S -i 30', ksp_white[:, :, None, :], s_maps_white[:, :, None, :])
    gt_img_white_cropped_noisy = bart(1, 'pics -S -i 30', ksp_white_noisy[:, :, None, :], s_maps_white_noisy[:, :, None, :])

    ksp_white = ksp_white.transpose(2, 0, 1)
    ksp_white_noisy = ksp_white_noisy.transpose(2, 0, 1)
    s_maps_white = s_maps_white.transpose(2, 0, 1)
    s_maps_white_noisy = s_maps_white_noisy.transpose(2, 0, 1)
    cimg_white = cimg_white.transpose(2, 0, 1)
    cimg_white_noisy = cimg_white_noisy.transpose(2, 0, 1)

    # Critical: pass imsize explicitly.
    norm_const_99_white = normalization_const(s_maps_white, gt_img_white_cropped, ACS_size=ACS_size, imsize=imsize)
    norm_const_99_white_noisy = normalization_const(s_maps_white_noisy, gt_img_white_cropped_noisy, ACS_size=ACS_size, imsize=imsize)

    ksp_white = ksp_white / norm_const_99_white
    ksp_white_noisy = ksp_white_noisy / norm_const_99_white_noisy

    s_maps_white = bart(
        1, 'ecalib -m 1 -c0',
        ksp_white.transpose(1, 2, 0)[:, :, None, :]
    ).squeeze().transpose(2, 0, 1)

    s_maps_white_noisy = bart(
        1, 'ecalib -m 1 -c0',
        ksp_white_noisy.transpose(1, 2, 0)[:, :, None, :]
    ).squeeze().transpose(2, 0, 1)

    gt_img_white_cropped = bart(
        1, 'pics -S -i 30',
        ksp_white.transpose(1, 2, 0)[:, :, None, :],
        s_maps_white.transpose(1, 2, 0)[:, :, None, :]
    )

    gt_img_white_cropped_noisy = bart(
        1, 'pics -S -i 30',
        ksp_white_noisy.transpose(1, 2, 0)[:, :, None, :],
        s_maps_white_noisy.transpose(1, 2, 0)[:, :, None, :]
    )

    cimg_white = bart(1, 'fft -iu 3', ksp_white.transpose(1, 2, 0)).transpose(2, 0, 1)
    cimg_white_noisy = bart(1, 'fft -iu 3', ksp_white_noisy.transpose(1, 2, 0)).transpose(2, 0, 1)

    var_noisy = np.var(cimg_white_noisy[:, 0:30, 0:30])

    coil_imgs_with_maps_white_noisy = cimg_white_noisy * np.conj(s_maps_white_noisy)
    u_white_noisy = np.sum(coil_imgs_with_maps_white_noisy, axis=-3)
    u_cropped_white_noisy = u_white_noisy

    meta = {
        "source_file": os.path.basename(h5_path),
        "source_path": h5_path,
        "slice_idx": int(slice_idx),
        "imsize": int(imsize),
    }

    return (
        i,
        gt_img_white_cropped,
        gt_img_white_cropped_noisy,
        u_cropped_white_noisy,
        norm_const_99_white_noisy,
        var_noisy,
        ksp_white_noisy,
        s_maps_white_noisy,
        meta,
    )


metas = [None] * total_iterations

with Pool(n_proc) as p:
    iterator = p.imap(_process_one, range(total_iterations))
    for (
        i,
        gt_img_white_cropped,
        gt_img_white_cropped_noisy,
        u_cropped_white_noisy,
        norm_const_99,
        var_noisy,
        ksp_white_noisy,
        s_maps_white_noisy,
        meta,
    ) in tqdm(iterator, total=total_iterations):

        x_est_gt[i] = torch.tensor(gt_img_white_cropped, dtype=torch.complex64)
        x_est[i] = torch.tensor(gt_img_white_cropped_noisy, dtype=torch.complex64)
        u_images[i] = torch.tensor(u_cropped_white_noisy, dtype=torch.complex64)
        norm_consts_99[i] = torch.tensor(norm_const_99, dtype=torch.float32)
        noise_var_noisy[i] = torch.tensor(var_noisy, dtype=torch.float32)
        metas[i] = meta

        ksp_white_noisy_t = torch.tensor(ksp_white_noisy, dtype=torch.complex64)
        s_maps_white_noisy_t = torch.tensor(s_maps_white_noisy, dtype=torch.complex64)

        torch.save(
            {
                "ksp_white_noisy": ksp_white_noisy_t,
                "s_maps_white_noisy": s_maps_white_noisy_t,
                "meta": meta,
            },
            os.path.join(path, "ksp", f"{i}.pt")
        )
        print(f"Step {i} Done: {meta['source_file']} slice {meta['slice_idx']}")

torch.save(
    {
        'x_est_gt': x_est_gt,
        'x_est': x_est,
        'u_images': u_images,
        'norm_consts_99': norm_consts_99,
        'noise_var_noisy': noise_var_noisy,
        'meta': metas,
        'manifest_csv': args.manifest_csv,
        'imsize': imsize,
        'acs_size': ACS_size,
        'noise_level': db,
    },
    os.path.join(path, "noisy.pt")
)

if args.manifest_csv is not None:
    shutil.copy(args.manifest_csv, os.path.join(path, "manifest_used.csv"))

print("Saved:", os.path.join(path, "noisy.pt"))
