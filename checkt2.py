from pathlib import Path
import pandas as pd

t2_dir = Path("/mnt/SSD/wsy/data/fastmri_batch0_eval/val_t2/32dB")

csv_path = t2_dir / "sample_source_mapping.csv"
txt_path = t2_dir / "sample_source_mapping.txt"
mapping_path = t2_dir / "mapping.txt"

print(f"[目录] {t2_dir}\n")

if csv_path.exists():
    print(f"[读取] {csv_path}")
    df = pd.read_csv(csv_path)
    print(df)

    print("\n[T2 原始文件名列表]")
    for _, row in df.iterrows():
        print(row.to_dict())

elif txt_path.exists():
    print(f"[读取] {txt_path}")
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    print(text)

elif mapping_path.exists():
    print(f"[读取] {mapping_path}")
    text = mapping_path.read_text(encoding="utf-8", errors="ignore")
    print(text)

else:
    print("[错误] 没找到 mapping.txt / sample_source_mapping.csv / sample_source_mapping.txt")