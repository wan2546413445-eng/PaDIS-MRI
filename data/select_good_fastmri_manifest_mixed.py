#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_good_fastmri_manifest_mixed.py

用途：
    从 fastMRI multicoil_train 的 .h5 文件中，按 contrast 分层筛选质量较好的 slice，
    输出 manifest CSV。后续用同一份 manifest 生成 d384 和 d320 训练集，保证样本一致。

筛选依据：
    1. 多 contrast 混合训练：AXFLAIR / AXT1 / AXT1PRE / AXT1POST / AXT2
    2. 只从每个 volume 的中心区域候选 slice 中挑选
    3. 用 RSS 快速图像做 FOV-QC：center320 不应明显裁掉 head support
    4. 用 corner noise 估计一个 SNR-like 指标，剔除过噪或异常切片
    5. 保存全部候选和 selected manifest，保证可复现

示例：
python data/select_good_fastmri_manifest_mixed.py \
  --h5_folder /mnt/public/成像组/dataset/fast_MRI/multicoil_brain/brain_multicoil_train_batch_0/multicoil_train \
  --quota AXFLAIR:40,AXT1:40,AXT1PRE:40,AXT1POST:40,AXT2:40 \
  --max_per_volume 6 \
  --central_frac 0.60 \
  --crop_ref 384 \
  --crop_target 320 \
  --max_lost_fg_ratio 0.01 \
  --max_border_fg_ratio 0.02 \
  --min_snr_like 8 \
  --out_csv /mnt/SSD/wsy/data/fastmri_train_batch0_pilot_center320_manifest/mixed_center320_qc.csv
"""

import argparse
import csv
import os
from pathlib import Path

import h5py
import numpy as np
import sigpy as sp

from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
from scipy.ndimage import label as ndi_label


def infer_contrast(filename: str) -> str:
    name = filename.upper()
    if "AXFLAIR" in name:
        return "AXFLAIR"
    if "AXT1POST" in name:
        return "AXT1POST"
    if "AXT1PRE" in name:
        return "AXT1PRE"
    if "AXT1" in name:
        return "AXT1"
    if "AXT2" in name:
        return "AXT2"
    return "UNKNOWN"


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """
    kspace: [coil, H, W]
    return: coil images [coil, H, W]
    """
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi_label(mask)
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def make_support(mag: np.ndarray, thr_ratio: float = 0.015) -> np.ndarray:
    """
    低阈值 + 形态学处理得到粗略 head support。
    这里只用于 FOV-QC，不是精确脑分割。
    """
    p99 = np.percentile(mag, 99)
    if p99 <= 0:
        return np.zeros_like(mag, dtype=bool)

    mask = mag > (thr_ratio * p99)
    mask = binary_opening(mask, iterations=1)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    mask = largest_component(mask)
    return mask.astype(bool)


def center_crop_bounds(n: int, crop: int):
    s = (n - crop) // 2
    return s, s + crop


def corner_noise_std(mag: np.ndarray, corner: int = 30) -> float:
    h, w = mag.shape
    c = min(corner, h // 4, w // 4)
    blocks = [
        mag[:c, :c],
        mag[:c, -c:],
        mag[-c:, :c],
        mag[-c:, -c:],
    ]
    vals = np.concatenate([b.reshape(-1) for b in blocks])
    return float(np.std(vals))


def quality_one_slice(
    h5_path: Path,
    slice_idx: int,
    crop_ref: int = 384,
    crop_target: int = 320,
    fg_thr_ratio: float = 0.015,
    border_margin: int = 5,
):
    with h5py.File(h5_path, "r") as f:
        ksp = np.asarray(f["kspace"][slice_idx])  # fastMRI: [coil, H, W]

    coil_imgs = ifft2c(ksp)
    rss = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0))

    # 近似 PaDIS 的中心 resize/crop 尺度，用于 QC
    mag_ref = sp.resize(rss, [crop_ref, crop_ref])
    mag_ref = np.asarray(mag_ref, dtype=np.float32)

    p99 = float(np.percentile(mag_ref, 99))
    p95 = float(np.percentile(mag_ref, 95))
    noise = corner_noise_std(mag_ref)
    snr_like = p95 / (noise + 1e-8)

    support = make_support(mag_ref, thr_ratio=fg_thr_ratio)
    total_fg = int(support.sum())

    y0, y1 = center_crop_bounds(crop_ref, crop_target)
    x0, x1 = center_crop_bounds(crop_ref, crop_target)

    inside = np.zeros_like(support, dtype=bool)
    inside[y0:y1, x0:x1] = True

    lost_fg = int((support & (~inside)).sum())
    lost_ratio = lost_fg / max(total_fg, 1)

    crop_support = support[y0:y1, x0:x1]
    crop_fg = int(crop_support.sum())

    border = np.zeros_like(crop_support, dtype=bool)
    m = border_margin
    border[:m, :] = True
    border[-m:, :] = True
    border[:, :m] = True
    border[:, -m:] = True

    border_fg = int((crop_support & border).sum())
    border_ratio = border_fg / max(crop_fg, 1)

    fg_area_ratio = total_fg / float(crop_ref * crop_ref)

    # 简单排序分数：高 SNR-like、低裁剪损失、低边缘 foreground 更优
    score = snr_like - 200.0 * lost_ratio - 100.0 * border_ratio

    return {
        "file": str(h5_path),
        "filename": os.path.basename(h5_path),
        "slice": int(slice_idx),
        "contrast": infer_contrast(os.path.basename(h5_path)),
        "p99": p99,
        "p95": p95,
        "noise_std": noise,
        "snr_like": float(snr_like),
        "fg_area_ratio": float(fg_area_ratio),
        "lost_fg_ratio": float(lost_ratio),
        "border_fg_ratio": float(border_ratio),
        "score": float(score),
    }


def parse_quota(s: str):
    """
    Example:
        AXFLAIR:40,AXT1:40,AXT1PRE:40,AXT1POST:40,AXT2:40
    """
    out = {}
    for item in s.split(","):
        if not item.strip():
            continue
        k, v = item.split(":")
        out[k.strip().upper()] = int(v)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_folder", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument(
        "--quota",
        type=str,
        default="AXFLAIR:40,AXT1:40,AXT1PRE:40,AXT1POST:40,AXT2:40",
        help="Per-contrast quotas. Use 'AXFLAIR:40,AXT1:40,AXT1PRE:40,AXT1POST:40,AXT2:40'.",
    )
    parser.add_argument("--max_per_volume", type=int, default=6)
    parser.add_argument("--central_frac", type=float, default=0.60)
    parser.add_argument("--crop_ref", type=int, default=384)
    parser.add_argument("--crop_target", type=int, default=320)
    parser.add_argument("--fg_thr_ratio", type=float, default=0.015)
    parser.add_argument("--max_lost_fg_ratio", type=float, default=0.01)
    parser.add_argument("--max_border_fg_ratio", type=float, default=0.02)
    parser.add_argument("--min_snr_like", type=float, default=8.0)
    parser.add_argument("--min_fg_area_ratio", type=float, default=0.05)
    parser.add_argument("--max_fg_area_ratio", type=float, default=0.85)
    args = parser.parse_args()

    quota = parse_quota(args.quota)

    h5_folder = Path(args.h5_folder)
    files = sorted(h5_folder.glob("*.h5"))
    files = [p for p in files if infer_contrast(p.name) in quota]

    print("Candidate files:", len(files))
    if len(files) == 0:
        raise RuntimeError("No candidate .h5 files found. Check --h5_folder and --quota.")

    by_contrast_files = {}
    for p in files:
        by_contrast_files.setdefault(infer_contrast(p.name), []).append(p)

    for c, ps in sorted(by_contrast_files.items()):
        print(c, "volumes:", len(ps), "quota:", quota.get(c, 0))

    all_rows = []
    selected = []

    for contrast, ps in sorted(by_contrast_files.items()):
        contrast_rows_ok = []

        for h5_path in ps:
            try:
                with h5py.File(h5_path, "r") as f:
                    ns = int(f["kspace"].shape[0])
            except Exception as e:
                all_rows.append({
                    "selected": 0,
                    "status": "error",
                    "contrast": contrast,
                    "file": str(h5_path),
                    "filename": h5_path.name,
                    "slice": -1,
                    "error": repr(e),
                    "score": -1e9,
                })
                continue

            lo = int((1.0 - args.central_frac) / 2.0 * ns)
            hi = int((1.0 + args.central_frac) / 2.0 * ns)
            lo = max(0, lo)
            hi = min(ns, hi)
            candidate_slices = list(range(lo, hi))

            rows_this_volume = []

            for sidx in candidate_slices:
                try:
                    row = quality_one_slice(
                        h5_path,
                        sidx,
                        crop_ref=args.crop_ref,
                        crop_target=args.crop_target,
                        fg_thr_ratio=args.fg_thr_ratio,
                    )

                    ok = (
                        row["lost_fg_ratio"] <= args.max_lost_fg_ratio
                        and row["border_fg_ratio"] <= args.max_border_fg_ratio
                        and row["snr_like"] >= args.min_snr_like
                        and args.min_fg_area_ratio <= row["fg_area_ratio"] <= args.max_fg_area_ratio
                    )
                    row["status"] = "ok" if ok else "bad"
                    row["selected"] = 0
                    row["error"] = ""

                    rows_this_volume.append(row)
                    all_rows.append(row)

                except Exception as e:
                    all_rows.append({
                        "selected": 0,
                        "status": "error",
                        "contrast": contrast,
                        "file": str(h5_path),
                        "filename": h5_path.name,
                        "slice": int(sidx),
                        "error": repr(e),
                        "score": -1e9,
                    })

            ok_this_volume = [r for r in rows_this_volume if r.get("status") == "ok"]
            ok_this_volume = sorted(ok_this_volume, key=lambda r: r["score"], reverse=True)
            contrast_rows_ok.extend(ok_this_volume[: args.max_per_volume])

        contrast_rows_ok = sorted(contrast_rows_ok, key=lambda r: r["score"], reverse=True)
        take = quota.get(contrast, 0)
        selected.extend(contrast_rows_ok[:take])

        print(
            contrast,
            "ok candidates after per-volume cap:",
            len(contrast_rows_ok),
            "selected:",
            min(take, len(contrast_rows_ok)),
        )

    selected_keys = {(r["file"], int(r["slice"])) for r in selected}
    for r in all_rows:
        if (r.get("file"), int(r.get("slice", -999))) in selected_keys:
            r["selected"] = 1

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "selected",
        "status",
        "contrast",
        "file",
        "filename",
        "slice",
        "score",
        "snr_like",
        "p99",
        "p95",
        "noise_std",
        "fg_area_ratio",
        "lost_fg_ratio",
        "border_fg_ratio",
        "error",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    selected_csv = out_csv.with_name(out_csv.stem + "_selected.csv")
    selected_sorted = sorted(selected, key=lambda r: (r["contrast"], -r["score"]))

    with open(selected_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in selected_sorted:
            rr = dict(r)
            rr["selected"] = 1
            writer.writerow(rr)

    print("All QC CSV:", out_csv)
    print("Selected CSV:", selected_csv)
    print("Total selected:", len(selected_sorted))

    counts = {}
    for r in selected_sorted:
        counts[r["contrast"]] = counts.get(r["contrast"], 0) + 1
    print("Selected by contrast:", counts)

    missing = {c: quota[c] - counts.get(c, 0) for c in quota if counts.get(c, 0) < quota[c]}
    if missing:
        print("WARNING: Some contrasts did not reach quota:", missing)
        print("You can relax --central_frac, --max_per_volume, --min_snr_like, or FOV thresholds.")


if __name__ == "__main__":
    main()
