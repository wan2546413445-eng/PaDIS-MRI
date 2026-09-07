#!/usr/bin/env python3
import os
import glob
import json
import csv
import argparse
import pandas as pd


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_sample_idx(path):
    # .../sample_3/physical_kspace_error_summary.json
    parent = os.path.basename(os.path.dirname(path))
    return int(parent.replace("sample_", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    json_files = sorted(
        glob.glob(os.path.join(args.root, "sample_*", "physical_kspace_error_summary.json")),
        key=extract_sample_idx,
    )

    if not json_files:
        raise FileNotFoundError(f"No summary json found under {args.root}")

    summary_rows = []
    band_rows = []

    for jf in json_files:
        idx = extract_sample_idx(jf)
        meta = load_json(jf)

        row = {
            "sample_idx": idx,
            "physical_kspace_nmse_full": meta.get("physical_kspace_nmse_full"),
            "measured_nmse": meta.get("measured_nmse"),
            "unmeasured_nmse": meta.get("unmeasured_nmse"),
            "measured_err_percent": meta.get("measured_err_percent"),
            "unmeasured_err_percent": meta.get("unmeasured_err_percent"),
            "measured_gt_energy_percent": meta.get("measured_gt_energy_percent"),
            "unmeasured_gt_energy_percent": meta.get("unmeasured_gt_energy_percent"),
            "mask_measured_pixel_ratio": meta.get("mask_measured_pixel_ratio"),
            "mask_unmeasured_pixel_ratio": meta.get("mask_unmeasured_pixel_ratio"),
        }
        summary_rows.append(row)

        band_csv = os.path.join(os.path.dirname(jf), "physical_kspace_band_error_stats.csv")
        if os.path.exists(band_csv):
            bdf = pd.read_csv(band_csv)
            bdf.insert(0, "sample_idx", idx)
            band_rows.append(bdf)

    summary_df = pd.DataFrame(summary_rows).sort_values("sample_idx")
    band_df = pd.concat(band_rows, ignore_index=True) if band_rows else pd.DataFrame()

    summary_csv = os.path.join(args.out_dir, "physical_kspace_all32_summary.csv")
    band_all_csv = os.path.join(args.out_dir, "physical_kspace_all32_band_all.csv")
    band_mean_csv = os.path.join(args.out_dir, "physical_kspace_all32_band_mean.csv")

    summary_df.to_csv(summary_csv, index=False)

    if not band_df.empty:
        band_df.to_csv(band_all_csv, index=False)

        mean_cols = [
            "err_percent_total",
            "gt_energy_percent_total",
            "band_nmse",
            "measured_nmse",
            "unmeasured_nmse",
            "measured_err_percent_in_band",
            "unmeasured_err_percent_in_band",
        ]

        keep_cols = ["r_lo", "r_hi"] + mean_cols
        band_mean = (
            band_df[keep_cols]
            .groupby(["r_lo", "r_hi"], as_index=False)
            .agg(["mean", "std"])
        )

        # flatten columns
        band_mean.columns = [
            "_".join([str(c) for c in col if c != ""])
            for col in band_mean.columns.values
        ]
        band_mean.to_csv(band_mean_csv, index=False)

    global_mean = summary_df.drop(columns=["sample_idx"]).mean(numeric_only=True)
    global_std = summary_df.drop(columns=["sample_idx"]).std(numeric_only=True)

    print("==================================================")
    print("Physical k-space all32 summary")
    print("--------------------------------------------------")
    print(f"Num samples: {len(summary_df)}")
    print("--------------------------------------------------")
    for k in global_mean.index:
        print(f"{k}: {global_mean[k]:.6e} ± {global_std[k]:.6e}")
    print("--------------------------------------------------")
    print(f"Saved summary CSV: {summary_csv}")
    if not band_df.empty:
        print(f"Saved band all CSV: {band_all_csv}")
        print(f"Saved band mean CSV: {band_mean_csv}")
    print("==================================================")


if __name__ == "__main__":
    main()