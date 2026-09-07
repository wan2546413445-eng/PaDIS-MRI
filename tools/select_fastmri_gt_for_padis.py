#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始 fastMRI 多线圈脑部验证集重新选择适合 PaDIS-MRI 定性展示的样本（V2）。

V2 修复的问题
--------------
旧版存在以下问题：
1. 每张切片独立归一化，几乎空白的噪声切片也会被拉伸到 [0, 1]。
2. 梯度、拉普拉斯和熵会把随机噪声误判为“清晰细节”。
3. 对 16 层体数据排除 12% 边缘，只会排除第 0 和第 15 层。
4. 某个体数据即使没有合格切片，也会退化选择一个无效切片。

V2 的核心修正
--------------
1. 先读取一个体数据的全部切片，再使用该体数据统一的强度尺度归一化。
2. 使用原始强度比例、连通脑区面积、头部包围框面积和背景噪声进行硬筛选。
3. 结构指标在高斯平滑图像上计算，降低白噪声获得高分的可能。
4. 每个体数据没有合格切片时，直接跳过，不再强制选一张。
5. 每个体数据只保留一张最佳切片，避免相邻层面重复。
6. 只依据 GT 选择，不读取任何模型重建结果。

默认扫描
--------
/mnt/public/成像组/dataset/fast_MRI/multicoil_brain/
brain_multicoil_val_batch_0/multicoil_val

仅保留
------
- AXT1
- AXFLAIR
- AXT2

排除
----
- AXT1PRE
- AXT1POST
- 其他序列

重要输出
--------
output_dir/
├── padis_preprocess_manifest.csv
├── review_candidates.csv
├── rejected_volumes.csv
├── contact_sheets/
│   ├── T1_review_top30.png
│   ├── FLAIR_review_top30.png
│   └── T2_review_top30.png
├── selected_gt_png/
├── selected_h5_symlinks/
└── *_all_slice_metrics.csv

运行
----
python select_fastmri_gt_for_padis_v2.py

建议第一次使用：
python select_fastmri_gt_for_padis_v2.py \
    --select-per-contrast 10 \
    --review-per-contrast 30
"""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import (
        binary_closing,
        binary_dilation,
        binary_erosion,
        binary_fill_holes,
        gaussian_filter,
        label,
        laplace,
    )
except ImportError as exc:
    raise ImportError(
        "该脚本需要 scipy。请执行：pip install scipy"
    ) from exc


# ============================================================
# 1. 默认配置
# ============================================================
DEFAULT_SOURCE_DIR = (
    "/mnt/public/成像组/dataset/fast_MRI/multicoil_brain/"
    "brain_multicoil_val_batch_0/multicoil_val"
)

# 使用新目录，避免与旧版错误结果混在一起。
DEFAULT_OUTPUT_DIR = (
    "/mnt/SSD/wsy/data/fastmri_batch0_eval/"
    "selected_from_original_val_v2"
)

DEFAULT_CROP_SIZE = 320
DEFAULT_SELECT_PER_CONTRAST = 10
DEFAULT_REVIEW_PER_CONTRAST = 30

# 16 层数据默认排除 [0, 1] 和 [14, 15]。
DEFAULT_EDGE_EXCLUDE_FRACTION = 0.125

# 以下是硬筛选阈值。它们主要用于去除空白、噪声和极小头部切片。
DEFAULT_MIN_FOREGROUND_RATIO = 0.11
DEFAULT_MIN_BBOX_AREA_RATIO = 0.16
DEFAULT_MIN_BBOX_HEIGHT_RATIO = 0.36
DEFAULT_MIN_BBOX_WIDTH_RATIO = 0.32
DEFAULT_MIN_RELATIVE_SIGNAL = 0.28
DEFAULT_MIN_FOREGROUND_SNR_DB = 8.0
DEFAULT_MAX_CENTER_OFFSET = 0.28

DEFAULT_CREATE_SYMLINKS = True


# ============================================================
# 2. 命令行
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从原始 fastMRI 验证集筛选 PaDIS-MRI 定性样本（V2）。"
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default=DEFAULT_SOURCE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--select-per-contrast",
        type=int,
        default=DEFAULT_SELECT_PER_CONTRAST,
    )
    parser.add_argument(
        "--review-per-contrast",
        type=int,
        default=DEFAULT_REVIEW_PER_CONTRAST,
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_CROP_SIZE,
    )
    parser.add_argument(
        "--edge-exclude-fraction",
        type=float,
        default=DEFAULT_EDGE_EXCLUDE_FRACTION,
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=DEFAULT_MIN_FOREGROUND_RATIO,
    )
    parser.add_argument(
        "--min-relative-signal",
        type=float,
        default=DEFAULT_MIN_RELATIVE_SIGNAL,
    )
    parser.add_argument(
        "--min-foreground-snr-db",
        type=float,
        default=DEFAULT_MIN_FOREGROUND_SNR_DB,
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# 3. 文件分类
# ============================================================
def classify_contrast(filename: str) -> Optional[str]:
    name = filename.upper()

    if "_AXFLAIR_" in name:
        return "FLAIR"

    if "_AXT2_" in name:
        return "T2"

    # 仅匹配普通 AXT1，不会匹配 AXT1PRE 或 AXT1POST。
    if "_AXT1_" in name:
        return "T1"

    return None


def discover_h5_files(source_dir: Path) -> Dict[str, List[Path]]:
    grouped = {
        "T1": [],
        "FLAIR": [],
        "T2": [],
    }

    for path in sorted(source_dir.glob("*.h5")):
        contrast = classify_contrast(path.name)

        if contrast is not None:
            grouped[contrast].append(path)

    return grouped


# ============================================================
# 4. fastMRI GT 读取
# ============================================================
def to_complex(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)

    if np.iscomplexobj(array):
        return array

    if array.ndim >= 1 and array.shape[-1] == 2:
        return array[..., 0] + 1j * array[..., 1]

    return array.astype(np.complex64)


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(
        shifted,
        axes=(-2, -1),
        norm="ortho",
    )
    return np.fft.fftshift(image, axes=(-2, -1))


def root_sum_of_squares(
    image: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(image) ** 2, axis=axis))


def get_reconstruction_dataset(
    h5_file: h5py.File,
) -> Optional[str]:
    for name in (
        "reconstruction_rss",
        "reconstruction_esc",
        "reconstruction",
    ):
        if name in h5_file:
            return name

    return None


def get_num_slices(h5_file: h5py.File) -> int:
    reconstruction_name = get_reconstruction_dataset(h5_file)

    if reconstruction_name is not None:
        return int(h5_file[reconstruction_name].shape[0])

    if "kspace" in h5_file:
        return int(h5_file["kspace"].shape[0])

    raise KeyError(
        "h5 中没有 reconstruction_rss/reconstruction 或 kspace。"
    )


def load_gt_slice(
    h5_file: h5py.File,
    slice_index: int,
) -> np.ndarray:
    reconstruction_name = get_reconstruction_dataset(h5_file)

    if reconstruction_name is not None:
        image = np.asarray(
            h5_file[reconstruction_name][slice_index]
        )
        return np.abs(image).astype(np.float32)

    if "kspace" not in h5_file:
        raise KeyError("h5 中没有可用 GT 或 kspace。")

    kspace = np.asarray(h5_file["kspace"][slice_index])
    kspace = to_complex(kspace)
    coil_images = ifft2c(kspace)

    if coil_images.ndim == 2:
        image = np.abs(coil_images)
    else:
        image = root_sum_of_squares(coil_images, axis=0)

    return np.asarray(image, dtype=np.float32)


def center_crop(
    image: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    if crop_size <= 0:
        return image

    height, width = image.shape[-2:]
    target_h = min(crop_size, height)
    target_w = min(crop_size, width)

    top = max(0, (height - target_h) // 2)
    left = max(0, (width - target_w) // 2)

    return image[
        top:top + target_h,
        left:left + target_w,
    ]


# ============================================================
# 5. 体数据统一归一化
# ============================================================
def robust_top_mean(
    image: np.ndarray,
    top_fraction: float = 0.02,
) -> float:
    values = np.asarray(image, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    count = max(1, int(round(values.size * top_fraction)))
    partition_index = max(0, values.size - count)
    top_values = np.partition(values, partition_index)[partition_index:]

    return float(np.mean(top_values))


def compute_volume_scale(
    raw_slices: Sequence[np.ndarray],
) -> float:
    """
    为一个体数据确定统一显示尺度。

    不能逐切片单独归一化，否则空白噪声切片也会显得很亮。
    """
    if not raw_slices:
        return 1.0

    positive_parts: List[np.ndarray] = []

    for image in raw_slices:
        values = np.asarray(image, dtype=np.float32)
        values = values[
            np.isfinite(values) & (values > 0)
        ]

        if values.size > 0:
            # 每张切片最多抽样约 5 万个值，控制内存。
            if values.size > 50000:
                step = max(1, values.size // 50000)
                values = values[::step]

            positive_parts.append(values)

    if not positive_parts:
        return 1.0

    merged = np.concatenate(positive_parts)
    scale = float(np.percentile(merged, 99.5))

    if not np.isfinite(scale) or scale <= 1e-12:
        return 1.0

    return scale


def normalize_with_volume_scale(
    image: np.ndarray,
    volume_scale: float,
) -> np.ndarray:
    normalized = np.asarray(image, dtype=np.float32)
    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    normalized = np.abs(normalized) / max(volume_scale, 1e-12)

    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


# ============================================================
# 6. 掩膜与背景噪声
# ============================================================
def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = label(mask)

    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    component_id = int(np.argmax(sizes))

    return labeled == component_id


def get_corner_values(
    image: np.ndarray,
    corner_fraction: float = 0.12,
) -> np.ndarray:
    height, width = image.shape
    corner_h = max(4, int(round(height * corner_fraction)))
    corner_w = max(4, int(round(width * corner_fraction)))

    corners = [
        image[:corner_h, :corner_w],
        image[:corner_h, -corner_w:],
        image[-corner_h:, :corner_w],
        image[-corner_h:, -corner_w:],
    ]

    return np.concatenate(
        [corner.ravel() for corner in corners]
    )


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    return 1.4826 * mad


def build_head_mask(
    image: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    使用体数据统一归一化后的图像建立最大连通头部区域。

    返回：
    - mask
    - background_median
    - background_sigma
    """
    corners = get_corner_values(image)
    background_median = float(np.median(corners))
    background_sigma = robust_sigma(corners)

    smoothed = gaussian_filter(image, sigma=1.2)

    threshold = max(
        0.035,
        background_median + 6.0 * background_sigma,
    )

    mask = smoothed > threshold
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    mask = largest_component(mask)

    # 去除非常细小的随机连通区。
    if int(mask.sum()) < 100:
        mask = np.zeros_like(mask, dtype=bool)

    return (
        mask.astype(bool),
        background_median,
        background_sigma,
    )


# ============================================================
# 7. 切片指标
# ============================================================
def safe_entropy(
    values: np.ndarray,
    bins: int = 64,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size < 100:
        return 0.0

    histogram, _ = np.histogram(
        values,
        bins=bins,
        range=(0.0, 1.0),
    )

    total = int(histogram.sum())

    if total == 0:
        return 0.0

    probability = histogram.astype(np.float64) / total
    probability = probability[probability > 0]

    return float(-np.sum(probability * np.log2(probability)))


def compute_slice_metrics(
    image: np.ndarray,
    raw_image: np.ndarray,
    slice_index: int,
    num_slices: int,
    volume_max_top_mean: float,
) -> Dict[str, float]:
    mask, background_median, background_sigma = build_head_mask(
        image
    )

    height, width = image.shape
    image_area = max(1, height * width)

    foreground_count = int(mask.sum())
    foreground_ratio = float(foreground_count / image_area)

    raw_top_mean = robust_top_mean(raw_image)
    relative_signal = float(
        raw_top_mean / max(volume_max_top_mean, 1e-12)
    )

    relative_position = float(
        slice_index / max(1, num_slices - 1)
    )

    if foreground_count < 100:
        return {
            "relative_slice_position": relative_position,
            "foreground_ratio": foreground_ratio,
            "bbox_area_ratio": 0.0,
            "bbox_height_ratio": 0.0,
            "bbox_width_ratio": 0.0,
            "center_offset": 1.0,
            "center_occupancy": 0.0,
            "relative_signal": relative_signal,
            "foreground_snr_db": -100.0,
            "coherent_gradient": 0.0,
            "coherent_laplacian": 0.0,
            "central_structure_std": 0.0,
            "smoothed_entropy": 0.0,
            "background_median": background_median,
            "background_sigma": background_sigma,
            "background_high_frequency": 1.0,
            "rim_dominance": 100.0,
            "border_touch_ratio": 1.0,
        }

    rows, cols = np.where(mask)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1

    bbox_h = max(1, y1 - y0)
    bbox_w = max(1, x1 - x0)
    bbox_area = bbox_h * bbox_w

    bbox_area_ratio = float(bbox_area / image_area)
    bbox_height_ratio = float(bbox_h / height)
    bbox_width_ratio = float(bbox_w / width)

    head_center_y = 0.5 * (y0 + y1 - 1)
    head_center_x = 0.5 * (x0 + x1 - 1)

    normalized_offset_y = (
        (head_center_y - (height - 1) / 2.0)
        / max(1.0, height / 2.0)
    )
    normalized_offset_x = (
        (head_center_x - (width - 1) / 2.0)
        / max(1.0, width / 2.0)
    )
    center_offset = float(
        np.sqrt(
            normalized_offset_x ** 2
            + normalized_offset_y ** 2
        )
    )

    center_region = np.zeros_like(mask, dtype=bool)
    cy0 = int(round(height * 0.15))
    cy1 = int(round(height * 0.85))
    cx0 = int(round(width * 0.15))
    cx1 = int(round(width * 0.85))
    center_region[cy0:cy1, cx0:cx1] = True

    center_occupancy = float(
        np.count_nonzero(mask & center_region)
        / max(1, foreground_count)
    )

    erosion_iterations = max(
        2,
        int(round(min(height, width) * 0.025)),
    )
    interior_mask = binary_erosion(
        mask,
        iterations=erosion_iterations,
    )

    if int(interior_mask.sum()) < 100:
        interior_mask = mask.copy()

    rim_mask = mask & ~binary_erosion(mask, iterations=2)

    # 在平滑图像上计算结构，避免白噪声获得高梯度分数。
    smooth_structure = gaussian_filter(image, sigma=1.4)
    gy, gx = np.gradient(smooth_structure)
    coherent_gradient_map = np.sqrt(gx ** 2 + gy ** 2)
    coherent_laplacian_map = np.abs(
        laplace(smooth_structure)
    )

    coherent_gradient = float(
        np.mean(coherent_gradient_map[interior_mask])
    )
    coherent_laplacian = float(
        np.mean(coherent_laplacian_map[interior_mask])
    )

    central_mask = interior_mask & center_region

    if int(central_mask.sum()) >= 100:
        central_structure_std = float(
            np.std(smooth_structure[central_mask])
        )
    else:
        central_structure_std = 0.0

    smoothed_entropy = safe_entropy(
        smooth_structure[interior_mask]
    )

    foreground_values = image[interior_mask]
    foreground_median = float(
        np.median(foreground_values)
    )

    noise_sigma = max(background_sigma, 1e-8)
    foreground_snr_db = float(
        20.0 * np.log10(
            max(foreground_median - background_median, 1e-8)
            / noise_sigma
        )
    )

    expanded_mask = binary_dilation(mask, iterations=5)
    background_mask = ~expanded_mask

    high_frequency = np.abs(
        image - gaussian_filter(image, sigma=1.0)
    )

    if int(background_mask.sum()) >= 100:
        background_high_frequency = float(
            np.mean(high_frequency[background_mask])
        )
    else:
        background_high_frequency = 1.0

    rim_gradient = (
        float(np.mean(coherent_gradient_map[rim_mask]))
        if int(rim_mask.sum()) >= 50
        else 0.0
    )

    rim_dominance = float(
        rim_gradient / max(coherent_gradient, 1e-8)
    )

    border_width = max(
        2,
        int(round(min(height, width) * 0.025)),
    )
    border = np.zeros_like(mask, dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    border_touch_ratio = float(
        np.count_nonzero(mask & border)
        / max(1, foreground_count)
    )

    return {
        "relative_slice_position": relative_position,
        "foreground_ratio": foreground_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_height_ratio": bbox_height_ratio,
        "bbox_width_ratio": bbox_width_ratio,
        "center_offset": center_offset,
        "center_occupancy": center_occupancy,
        "relative_signal": relative_signal,
        "foreground_snr_db": foreground_snr_db,
        "coherent_gradient": coherent_gradient,
        "coherent_laplacian": coherent_laplacian,
        "central_structure_std": central_structure_std,
        "smoothed_entropy": smoothed_entropy,
        "background_median": background_median,
        "background_sigma": background_sigma,
        "background_high_frequency": background_high_frequency,
        "rim_dominance": rim_dominance,
        "border_touch_ratio": border_touch_ratio,
    }


# ============================================================
# 8. 单体数据扫描
# ============================================================
def scan_volume(
    h5_path: Path,
    contrast: str,
    crop_size: int,
    edge_exclude_fraction: float,
    min_foreground_ratio: float,
    min_relative_signal: float,
    min_foreground_snr_db: float,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    先加载全部切片，建立体数据统一归一化尺度，再计算指标。
    """
    raw_slices: List[np.ndarray] = []

    with h5py.File(h5_path, "r") as h5_file:
        num_slices = get_num_slices(h5_file)

        for slice_index in range(num_slices):
            raw = load_gt_slice(
                h5_file=h5_file,
                slice_index=slice_index,
            )
            raw = center_crop(raw, crop_size)
            raw_slices.append(
                np.asarray(raw, dtype=np.float32)
            )

    volume_scale = compute_volume_scale(raw_slices)

    top_means = [
        robust_top_mean(image)
        for image in raw_slices
    ]
    volume_max_top_mean = max(
        max(top_means, default=0.0),
        1e-12,
    )

    # 使用 ceil，确保 16 层时明确排除 0、1、14、15。
    edge_count = int(
        math.ceil(num_slices * edge_exclude_fraction)
    )
    edge_count = max(1, edge_count)

    first_allowed = edge_count
    last_allowed_exclusive = max(
        first_allowed + 1,
        num_slices - edge_count,
    )

    records: List[Dict[str, Any]] = []
    normalized_slices: List[np.ndarray] = []

    for slice_index, raw_image in enumerate(raw_slices):
        image = normalize_with_volume_scale(
            image=raw_image,
            volume_scale=volume_scale,
        )
        normalized_slices.append(image)

        metrics = compute_slice_metrics(
            image=image,
            raw_image=raw_image,
            slice_index=slice_index,
            num_slices=num_slices,
            volume_max_top_mean=volume_max_top_mean,
        )

        is_edge_excluded = not (
            first_allowed
            <= slice_index
            < last_allowed_exclusive
        )

        rejection_reasons: List[str] = []

        if is_edge_excluded:
            rejection_reasons.append("edge_slice")

        if (
            metrics["foreground_ratio"]
            < min_foreground_ratio
        ):
            rejection_reasons.append("small_foreground")

        if (
            metrics["bbox_area_ratio"]
            < DEFAULT_MIN_BBOX_AREA_RATIO
        ):
            rejection_reasons.append("small_bbox_area")

        if (
            metrics["bbox_height_ratio"]
            < DEFAULT_MIN_BBOX_HEIGHT_RATIO
        ):
            rejection_reasons.append("small_bbox_height")

        if (
            metrics["bbox_width_ratio"]
            < DEFAULT_MIN_BBOX_WIDTH_RATIO
        ):
            rejection_reasons.append("small_bbox_width")

        if (
            metrics["relative_signal"]
            < min_relative_signal
        ):
            rejection_reasons.append("low_relative_signal")

        if (
            metrics["foreground_snr_db"]
            < min_foreground_snr_db
        ):
            rejection_reasons.append("low_foreground_snr")

        if (
            metrics["center_offset"]
            > DEFAULT_MAX_CENTER_OFFSET
        ):
            rejection_reasons.append("off_center")

        # 明显接触裁剪边缘时不进入候选。
        if metrics["border_touch_ratio"] > 0.02:
            rejection_reasons.append("touching_border")

        is_valid = len(rejection_reasons) == 0

        record: Dict[str, Any] = {
            "contrast": contrast,
            "source_h5": str(h5_path),
            "source_filename": h5_path.name,
            "slice_index": int(slice_index),
            "num_slices": int(num_slices),
            "volume_scale": float(volume_scale),
            "is_edge_excluded": int(is_edge_excluded),
            "is_valid_candidate": int(is_valid),
            "rejection_reasons": "|".join(
                rejection_reasons
            ),
        }
        record.update(metrics)
        records.append(record)

    return records, normalized_slices


# ============================================================
# 9. 评分
# ============================================================
def robust_scale_with_reference(
    values: Sequence[float],
    reference_mask: Sequence[bool],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(reference_mask, dtype=bool)

    reference = array[
        mask & np.isfinite(array)
    ]

    if reference.size == 0:
        return np.zeros_like(array)

    low = float(np.percentile(reference, 5))
    high = float(np.percentile(reference, 95))

    if high <= low + 1e-12:
        return np.zeros_like(array)

    scaled = (array - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def assign_scores_within_contrast(
    records: List[Dict[str, Any]],
) -> None:
    if not records:
        return

    valid_mask = [
        bool(record["is_valid_candidate"])
        for record in records
    ]

    positive_weights = {
        "foreground_ratio": 0.16,
        "bbox_area_ratio": 0.10,
        "relative_signal": 0.18,
        "foreground_snr_db": 0.18,
        "coherent_gradient": 0.13,
        "coherent_laplacian": 0.09,
        "central_structure_std": 0.10,
        "smoothed_entropy": 0.04,
        "center_occupancy": 0.02,
    }

    negative_weights = {
        "background_sigma": 0.08,
        "background_high_frequency": 0.08,
        "rim_dominance": 0.05,
        "center_offset": 0.04,
        "border_touch_ratio": 0.04,
    }

    scaled: Dict[str, np.ndarray] = {}

    for metric_name in positive_weights:
        scaled[metric_name] = robust_scale_with_reference(
            [record[metric_name] for record in records],
            valid_mask,
        )

    for metric_name in negative_weights:
        scaled[metric_name] = robust_scale_with_reference(
            [record[metric_name] for record in records],
            valid_mask,
        )

    for index, record in enumerate(records):
        if not bool(record["is_valid_candidate"]):
            record["selection_score"] = -1e6
            continue

        score = 0.0

        for metric_name, weight in positive_weights.items():
            score += weight * scaled[metric_name][index]

        for metric_name, weight in negative_weights.items():
            score -= weight * scaled[metric_name][index]

        record["selection_score"] = float(score)


# ============================================================
# 10. 每个体数据选择最佳合格切片
# ============================================================
def select_best_slice_per_volume(
    records: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for record in records:
        grouped.setdefault(
            record["source_h5"],
            [],
        ).append(record)

    selected: List[Dict[str, Any]] = []
    rejected_volumes: List[Dict[str, Any]] = []

    for source_h5, volume_records in grouped.items():
        valid_records = [
            record
            for record in volume_records
            if bool(record["is_valid_candidate"])
        ]

        if not valid_records:
            rejection_counts: Dict[str, int] = {}

            for record in volume_records:
                reasons = str(
                    record["rejection_reasons"]
                ).split("|")

                for reason in reasons:
                    if reason:
                        rejection_counts[reason] = (
                            rejection_counts.get(reason, 0)
                            + 1
                        )

            rejected_volumes.append(
                {
                    "contrast": volume_records[0]["contrast"],
                    "source_h5": source_h5,
                    "source_filename": volume_records[0][
                        "source_filename"
                    ],
                    "num_slices": volume_records[0][
                        "num_slices"
                    ],
                    "valid_slice_count": 0,
                    "dominant_rejection_reasons": "|".join(
                        f"{key}:{value}"
                        for key, value in sorted(
                            rejection_counts.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ),
                }
            )
            continue

        best = max(
            valid_records,
            key=lambda item: float(
                item["selection_score"]
            ),
        )

        selected.append(dict(best))

    selected.sort(
        key=lambda item: float(
            item["selection_score"]
        ),
        reverse=True,
    )

    return selected, rejected_volumes


# ============================================================
# 11. 输出工具
# ============================================================
SLICE_FIELDS = [
    "contrast",
    "source_h5",
    "source_filename",
    "slice_index",
    "num_slices",
    "is_valid_candidate",
    "is_edge_excluded",
    "rejection_reasons",
    "selection_score",
    "relative_slice_position",
    "foreground_ratio",
    "bbox_area_ratio",
    "bbox_height_ratio",
    "bbox_width_ratio",
    "center_offset",
    "center_occupancy",
    "relative_signal",
    "foreground_snr_db",
    "coherent_gradient",
    "coherent_laplacian",
    "central_structure_std",
    "smoothed_entropy",
    "background_median",
    "background_sigma",
    "background_high_frequency",
    "rim_dominance",
    "border_touch_ratio",
    "volume_scale",
]


def write_csv(
    path: Path,
    records: List[Dict[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fields),
            extrasaction="ignore",
        )
        writer.writeheader()

        for record in records:
            row = dict(record)

            for key, value in row.items():
                if isinstance(value, float):
                    row[key] = f"{value:.10g}"

            writer.writerow(row)


def reload_normalized_candidate(
    record: Dict[str, Any],
    crop_size: int,
) -> np.ndarray:
    source_h5 = Path(record["source_h5"])

    with h5py.File(source_h5, "r") as h5_file:
        raw = load_gt_slice(
            h5_file=h5_file,
            slice_index=int(record["slice_index"]),
        )

    raw = center_crop(raw, crop_size)

    return normalize_with_volume_scale(
        image=raw,
        volume_scale=float(record["volume_scale"]),
    )


def save_single_png(
    image: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    ax.axis("off")
    ax.set_title(title, fontsize=9)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def save_contact_sheet(
    records: List[Dict[str, Any]],
    output_path: Path,
    crop_size: int,
    columns: int,
    title: str,
) -> None:
    if not records:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = max(1, int(columns))
    rows = int(
        math.ceil(len(records) / columns)
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            3.25 * columns,
            3.75 * rows + 0.7,
        ),
    )
    axes = np.asarray(
        axes,
        dtype=object,
    ).reshape(-1)

    for rank, (ax, record) in enumerate(
        zip(axes, records),
        start=1,
    ):
        image = reload_normalized_candidate(
            record=record,
            crop_size=crop_size,
        )

        ax.imshow(
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(
            "\n".join(
                [
                    f"rank={rank} | {record['contrast']}",
                    record["source_filename"],
                    (
                        f"slice={record['slice_index']}/"
                        f"{record['num_slices'] - 1}"
                    ),
                    (
                        f"score={float(record['selection_score']):.3f} | "
                        f"area={float(record['foreground_ratio']):.3f}"
                    ),
                    (
                        f"rel.signal={float(record['relative_signal']):.3f} | "
                        f"fgSNR={float(record['foreground_snr_db']):.1f} dB"
                    ),
                ]
            ),
            fontsize=7.2,
        )
        ax.axis("off")

    for ax in axes[len(records):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 12. PaDIS manifest 和软链接
# ============================================================
PADIS_FIELDS = [
    "sample_id",
    "contrast",
    "source_h5",
    "source_filename",
    "slice_index",
    "num_slices",
    "selection_rank_within_contrast",
    "selection_score",
    "foreground_ratio",
    "relative_signal",
    "foreground_snr_db",
    "nominal_preprocess_snr_db",
    "notes",
]


def build_padis_manifest(
    selected_by_contrast: Dict[
        str,
        List[Dict[str, Any]],
    ],
) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    sample_id = 0

    for contrast in ("T1", "FLAIR", "T2"):
        for rank, record in enumerate(
            selected_by_contrast.get(contrast, []),
            start=1,
        ):
            manifest.append(
                {
                    "sample_id": sample_id,
                    "contrast": contrast,
                    "source_h5": record["source_h5"],
                    "source_filename": record[
                        "source_filename"
                    ],
                    "slice_index": int(
                        record["slice_index"]
                    ),
                    "num_slices": int(
                        record["num_slices"]
                    ),
                    "selection_rank_within_contrast": rank,
                    "selection_score": float(
                        record["selection_score"]
                    ),
                    "foreground_ratio": float(
                        record["foreground_ratio"]
                    ),
                    "relative_signal": float(
                        record["relative_signal"]
                    ),
                    "foreground_snr_db": float(
                        record["foreground_snr_db"]
                    ),
                    "nominal_preprocess_snr_db": 32,
                    "notes": (
                        "v2_gt_only_volume_normalized_"
                        "hard_filtered"
                    ),
                }
            )
            sample_id += 1

    return manifest


def create_symlinks(
    selected_by_contrast: Dict[
        str,
        List[Dict[str, Any]],
    ],
    root: Path,
) -> None:
    for contrast, records in (
        selected_by_contrast.items()
    ):
        target_dir = root / contrast
        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for record in records:
            source = Path(record["source_h5"])
            target = target_dir / source.name

            if target.exists() or target.is_symlink():
                continue

            try:
                os.symlink(source, target)
            except OSError as error:
                print(
                    f"[警告] 创建软链接失败：{target}: {error}",
                    file=sys.stderr,
                )


# ============================================================
# 13. 主程序
# ============================================================
def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"原始目录不存在：{source_dir}"
        )

    if not 0.0 <= args.edge_exclude_fraction < 0.5:
        raise ValueError(
            "--edge-exclude-fraction 必须位于 [0, 0.5)。"
        )

    if args.select_per_contrast <= 0:
        raise ValueError(
            "--select-per-contrast 必须大于 0。"
        )

    for subdir in (
        output_dir,
        output_dir / "contact_sheets",
        output_dir / "selected_gt_png",
        output_dir / "selected_h5_symlinks",
    ):
        subdir.mkdir(
            parents=True,
            exist_ok=True,
        )

    grouped_files = discover_h5_files(
        source_dir
    )

    print("=" * 90)
    print("发现的原始体数据：")

    for contrast in ("T1", "FLAIR", "T2"):
        print(
            f"{contrast:<6}: "
            f"{len(grouped_files[contrast])} 个 h5"
        )

    print("=" * 90)

    all_slice_records: List[Dict[str, Any]] = []
    all_rejected_volumes: List[Dict[str, Any]] = []
    selected_by_contrast: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}
    review_records: List[Dict[str, Any]] = []

    for contrast in ("T1", "FLAIR", "T2"):
        contrast_records: List[Dict[str, Any]] = []
        files = grouped_files[contrast]

        print(
            f"\n扫描 {contrast}，共 {len(files)} 个体数据。"
        )

        for file_order, h5_path in enumerate(
            files,
            start=1,
        ):
            try:
                volume_records, _ = scan_volume(
                    h5_path=h5_path,
                    contrast=contrast,
                    crop_size=int(args.crop_size),
                    edge_exclude_fraction=float(
                        args.edge_exclude_fraction
                    ),
                    min_foreground_ratio=float(
                        args.min_foreground_ratio
                    ),
                    min_relative_signal=float(
                        args.min_relative_signal
                    ),
                    min_foreground_snr_db=float(
                        args.min_foreground_snr_db
                    ),
                )
            except Exception as error:
                print(
                    f"[读取失败] {h5_path.name}: {error}",
                    file=sys.stderr,
                )
                continue

            contrast_records.extend(
                volume_records
            )

            valid_count = sum(
                int(record["is_valid_candidate"])
                for record in volume_records
            )

            print(
                f"[{file_order:>3}/{len(files)}] "
                f"{h5_path.name}，"
                f"{len(volume_records)} 层，"
                f"合格 {valid_count} 层"
            )

        assign_scores_within_contrast(
            contrast_records
        )

        volume_best, rejected_volumes = (
            select_best_slice_per_volume(
                contrast_records
            )
        )

        selected_count = min(
            int(args.select_per_contrast),
            len(volume_best),
        )
        review_count = min(
            int(args.review_per_contrast),
            len(volume_best),
        )

        selected = volume_best[:selected_count]
        review = volume_best[:review_count]

        selected_by_contrast[contrast] = selected
        all_slice_records.extend(contrast_records)
        all_rejected_volumes.extend(
            rejected_volumes
        )

        write_csv(
            output_dir
            / f"{contrast}_all_slice_metrics.csv",
            contrast_records,
            SLICE_FIELDS,
        )

        write_csv(
            output_dir
            / f"{contrast}_volume_best_candidates.csv",
            volume_best,
            SLICE_FIELDS,
        )

        save_contact_sheet(
            records=review,
            output_path=(
                output_dir
                / "contact_sheets"
                / f"{contrast}_review_top{review_count}.png"
            ),
            crop_size=int(args.crop_size),
            columns=5,
            title=(
                f"{contrast}: V2 top {review_count} "
                f"valid candidates from different volumes"
            ),
        )

        save_contact_sheet(
            records=selected,
            output_path=(
                output_dir
                / "contact_sheets"
                / f"{contrast}_auto_selected.png"
            ),
            crop_size=int(args.crop_size),
            columns=5,
            title=(
                f"{contrast}: V2 automatically selected "
                f"{selected_count} valid samples"
            ),
        )

        for rank, record in enumerate(
            selected,
            start=1,
        ):
            image = reload_normalized_candidate(
                record=record,
                crop_size=int(args.crop_size),
            )

            filename = (
                f"{contrast}_rank{rank:02d}_"
                f"{Path(record['source_filename']).stem}_"
                f"slice{int(record['slice_index']):03d}.png"
            )

            save_single_png(
                image=image,
                output_path=(
                    output_dir
                    / "selected_gt_png"
                    / filename
                ),
                title=(
                    f"{contrast} | rank {rank} | "
                    f"slice {record['slice_index']}/"
                    f"{record['num_slices'] - 1} | "
                    f"score={float(record['selection_score']):.3f}"
                ),
            )

        for rank, record in enumerate(
            review,
            start=1,
        ):
            review_item = dict(record)
            review_item[
                "review_rank_within_contrast"
            ] = rank
            review_item["keep"] = ""
            review_item["comment"] = ""
            review_records.append(review_item)

        print(
            f"{contrast}："
            f"{len(contrast_records)} 张切片，"
            f"{len(volume_best)} 个体数据通过硬筛选，"
            f"{len(rejected_volumes)} 个体数据无合格切片，"
            f"自动选择 {selected_count} 张。"
        )

    write_csv(
        output_dir / "all_slice_metrics.csv",
        all_slice_records,
        SLICE_FIELDS,
    )

    rejected_fields = [
        "contrast",
        "source_h5",
        "source_filename",
        "num_slices",
        "valid_slice_count",
        "dominant_rejection_reasons",
    ]

    write_csv(
        output_dir / "rejected_volumes.csv",
        all_rejected_volumes,
        rejected_fields,
    )

    review_fields = list(SLICE_FIELDS) + [
        "review_rank_within_contrast",
        "keep",
        "comment",
    ]

    write_csv(
        output_dir / "review_candidates.csv",
        review_records,
        review_fields,
    )

    manifest = build_padis_manifest(
        selected_by_contrast
    )

    write_csv(
        output_dir
        / "padis_preprocess_manifest.csv",
        manifest,
        PADIS_FIELDS,
    )

    if (
        not args.no_symlink
        and DEFAULT_CREATE_SYMLINKS
    ):
        create_symlinks(
            selected_by_contrast,
            output_dir / "selected_h5_symlinks",
        )

    summary_lines = [
        "fastMRI PaDIS-MRI GT 选择 V2",
        "",
        f"source_dir: {source_dir}",
        f"output_dir: {output_dir}",
        "",
    ]

    for contrast in ("T1", "FLAIR", "T2"):
        summary_lines.append(
            f"{contrast}: selected "
            f"{len(selected_by_contrast.get(contrast, []))}"
        )

    summary_lines.extend(
        [
            "",
            "当前 manifest 为自动初选结果。",
            "应先人工检查 contact_sheets，再固定最终清单。",
            "不要使用旧版 selected_from_original_val 的 manifest。",
        ]
    )

    (output_dir / "selection_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("V2 筛选完成。")
    print(f"输出目录：{output_dir}")
    print("请先检查：")
    print(output_dir / "contact_sheets")
    print("再使用：")
    print(
        output_dir
        / "padis_preprocess_manifest.csv"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
