#!/usr/bin/env python3
"""
Create a readable mapping table for PaDIS-MRI validation samples.

It traces PaDIS preprocessed files such as:
    sample_0.pt
    sample_7.pt
    sample_t1_10.pt
    sample_flair_0.pt
back to the original fastMRI .h5 filename and the fixed slice index used by
PaDIS-MRI's data/brain_val_data.py.

Typical usage for your batch0 T1/FLAIR subsampled validation set:

python tools/make_val_sample_mapping.py \
  --h5_folder /mnt/public/成像组/dataset/fast_MRI/multicoil_brain/brain_multicoil_val_batch_0/multicoil_val \
  --val_dir /mnt/SSD/wsy/data/fastmri_batch0_eval/val_t1-flair_subsamp/32dB \
  --contrast t1-flair
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def infer_tag_from_h5_name(fname: str, contrast: str) -> str:
    """Match the tag logic in PaDIS-MRI data/brain_val_data.py."""
    if contrast == "t1-flair":
        if "AXT1POST" in fname:
            return "t1post"
        if "AXT1PRE" in fname:
            return "t1pre"
        if "AXT1" in fname:
            return "t1"
        if "FLAIR" in fname:
            return "flair"
        return "other"
    if contrast == "t2":
        return "t2" if "AXT2" in fname else "other"
    return "other"


def collect_source_h5_files(h5_folder: Path, contrast: str) -> List[Path]:
    """Reproduce PaDIS-MRI's sorted glob + contrast filtering."""
    all_h5 = sorted(h5_folder.glob("*.h5"))
    if not all_h5:
        raise FileNotFoundError(f"No .h5 files found in {h5_folder}")

    out: List[Path] = []
    for f in all_h5:
        s = str(f)
        if contrast == "t1-flair":
            if "AXT1" in s or "FLAIR" in s:
                out.append(f)
        elif contrast == "t2":
            if "AXT2" in s:
                out.append(f)
        else:
            raise ValueError(f"Unsupported contrast: {contrast}")

    if not out:
        raise RuntimeError(f"No source files matched contrast={contrast!r} in {h5_folder}")
    return sorted(out)


def parse_generated_sample_name(name: str, contrast: str) -> Tuple[int, str, str]:
    """
    Return (generated_idx, generated_tag, generated_sample_name).

    Supported examples:
      sample_flair_0.pt
      sample_t1_10.pt
      sample_t1post_20.pt
      sample_t1pre_30.pt
      sample_0.pt      # T2 or already-renumbered sample; tag becomes contrast/default.
    """
    base = Path(name).name

    m = re.fullmatch(r"sample_([A-Za-z0-9]+)_(\d+)\.pt", base)
    if m:
        return int(m.group(2)), m.group(1).lower(), base

    m = re.fullmatch(r"sample_(\d+)\.pt", base)
    if m:
        # In raw T2 validation, sample_i.pt maps to ksp_files[i].
        # In subsampled t1-flair, this form only appears as final_sample, not generated source.
        default_tag = "t2" if contrast == "t2" else "unknown"
        return int(m.group(1)), default_tag, base

    raise ValueError(f"Cannot parse generated sample name: {name}")


def parse_mapping_file(mapping_file: Path, contrast: str) -> List[Dict[str, object]]:
    """
    Parse mapping.txt produced by subsampling.

    Handles lines like:
      sample_0.pt <- sample_flair_0.pt
      sample_0.pt <- original sample_7.pt | sample_7.pt <- sample_t1_10.pt
    """
    rows: List[Dict[str, object]] = []
    for raw_line in mapping_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or "<-" not in line:
            continue

        left_match = re.search(r"sample_(\d+)\.pt", line)
        if left_match is None:
            print(f"[WARN] Cannot parse final sample index from line: {line}")
            continue
        final_idx = int(left_match.group(1))
        final_sample = f"sample_{final_idx}.pt"

        # Find the original generated sample with explicit tag if present.
        tagged_matches = re.findall(r"sample_([A-Za-z0-9]+)_(\d+)\.pt", line)
        if tagged_matches:
            tag, idx_str = tagged_matches[-1]
            generated_idx = int(idx_str)
            generated_tag = tag.lower()
            generated_sample = f"sample_{generated_tag}_{generated_idx}.pt"
        else:
            # Fallback: take the last sample_i.pt after '<-'. Mostly for T2.
            sample_matches = re.findall(r"sample_(\d+)\.pt", line)
            if not sample_matches:
                print(f"[WARN] Cannot parse generated sample from line: {line}")
                continue
            generated_idx = int(sample_matches[-1])
            generated_tag = "t2" if contrast == "t2" else "unknown"
            generated_sample = f"sample_{generated_idx}.pt"

        rows.append({
            "final_idx": final_idx,
            "final_sample": final_sample,
            "generated_idx": generated_idx,
            "generated_sample": generated_sample,
            "generated_tag": generated_tag,
            "mapping_line": line,
        })
    return sorted(rows, key=lambda r: int(r["final_idx"]))


def scan_val_dir_without_mapping(val_dir: Path, contrast: str) -> List[Dict[str, object]]:
    """Fallback when mapping.txt does not exist: map files in val_dir directly."""
    pt_files = sorted(p for p in val_dir.glob("sample*.pt") if not p.name.startswith("noise_var"))
    rows: List[Dict[str, object]] = []
    for p in pt_files:
        # Final index: if file is sample_7.pt use 7; if sample_t1_10.pt use an enumerated index.
        m_final = re.fullmatch(r"sample_(\d+)\.pt", p.name)
        if m_final:
            final_idx = int(m_final.group(1))
            final_sample = p.name
        else:
            final_idx = len(rows)
            final_sample = p.name

        generated_idx, generated_tag, generated_sample = parse_generated_sample_name(p.name, contrast)
        rows.append({
            "final_idx": final_idx,
            "final_sample": final_sample,
            "generated_idx": generated_idx,
            "generated_sample": generated_sample,
            "generated_tag": generated_tag,
            "mapping_line": "",
        })
    return sorted(rows, key=lambda r: int(r["final_idx"]))


def build_mapping(
    h5_folder: Path,
    val_dir: Path,
    contrast: str,
    center_slice: int,
) -> List[Dict[str, object]]:
    source_h5 = collect_source_h5_files(h5_folder, contrast)

    mapping_file = val_dir / "mapping.txt"
    if mapping_file.exists():
        rows = parse_mapping_file(mapping_file, contrast)
        source_mode = f"mapping.txt: {mapping_file}"
    else:
        rows = scan_val_dir_without_mapping(val_dir, contrast)
        source_mode = f"direct scan: {val_dir}"

    out: List[Dict[str, object]] = []
    for r in rows:
        idx = int(r["generated_idx"])
        if idx < 0 or idx >= len(source_h5):
            raise IndexError(
                f"Generated index {idx} from {r['generated_sample']} is outside source file list "
                f"length {len(source_h5)}. Check h5_folder and contrast."
            )

        src = source_h5[idx]
        source_tag = infer_tag_from_h5_name(src.name, contrast)
        generated_tag = str(r["generated_tag"])

        out.append({
            "final_idx": int(r["final_idx"]),
            "comparison_png": f"comparison_sample_{int(r['final_idx'])}.png",
            "final_sample": r["final_sample"],
            "recon_npy": f"recon_patch_{int(r['final_idx'])}.npy",
            "generated_sample": r["generated_sample"],
            "generated_idx": idx,
            "generated_tag": generated_tag,
            "source_tag": source_tag,
            "slice_idx_python_0_based": center_slice,
            "slice_number_human_1_based": center_slice + 1,
            "source_h5": src.name,
            "source_h5_path": str(src),
            "mapping_source": source_mode,
            "mapping_line": r.get("mapping_line", ""),
            "tag_match": str(generated_tag == source_tag),
        })
    return out


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_txt(rows: List[Dict[str, object]], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w") as f:
        for r in rows:
            f.write(
                f"{r['comparison_png']} | {r['final_sample']} | {r['source_tag']} | "
                f"slice_idx={r['slice_idx_python_0_based']} (human #{r['slice_number_human_1_based']}) | "
                f"{r['source_h5']}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map PaDIS-MRI validation sample_N files back to original fastMRI h5 names and slice index."
    )
    parser.add_argument("--h5_folder", type=Path, required=True, help="Original fastMRI multicoil_val folder containing .h5 files")
    parser.add_argument("--val_dir", type=Path, required=True, help="PaDIS preprocessed val directory, e.g. val_t1-flair_subsamp/32dB")
    parser.add_argument("--contrast", type=str, default="t1-flair", choices=["t1-flair", "t2"], help="Contrast filter used during preprocessing")
    parser.add_argument("--center_slice", type=int, default=2, help="0-based slice index used by PaDIS-MRI brain_val_data.py")
    parser.add_argument("--out_csv", type=Path, default=None, help="Output CSV path. Default: <val_dir>/sample_source_mapping.csv")
    parser.add_argument("--out_txt", type=Path, default=None, help="Output TXT path. Default: <val_dir>/sample_source_mapping.txt")
    args = parser.parse_args()

    out_csv = args.out_csv or (args.val_dir / "sample_source_mapping.csv")
    out_txt = args.out_txt or (args.val_dir / "sample_source_mapping.txt")

    rows = build_mapping(
        h5_folder=args.h5_folder,
        val_dir=args.val_dir,
        contrast=args.contrast,
        center_slice=args.center_slice,
    )

    if not rows:
        raise RuntimeError("No samples were mapped. Check val_dir and mapping.txt.")

    write_csv(rows, out_csv)
    write_txt(rows, out_txt)

    counts = Counter(str(r["source_tag"]) for r in rows)
    print("Mapped samples:", len(rows))
    print("Counts by source_tag:", dict(counts))
    print("CSV saved to:", out_csv)
    print("TXT saved to:", out_txt)
    print("\nPreview:")
    for r in rows[:10]:
        print(
            f"  {r['comparison_png']} | {r['final_sample']} <- {r['generated_sample']} | "
            f"{r['source_tag']} | slice_idx={r['slice_idx_python_0_based']} | {r['source_h5']}"
        )

    mismatches = [r for r in rows if r["tag_match"] == "False" and r["generated_tag"] != "unknown"]
    if mismatches:
        print("\n[WARN] Tag mismatches found. Please check h5_folder / contrast:")
        for r in mismatches[:10]:
            print(f"  {r['final_sample']}: generated_tag={r['generated_tag']} source_tag={r['source_tag']} source={r['source_h5']}")


if __name__ == "__main__":
    main()
