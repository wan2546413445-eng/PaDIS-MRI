import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASELINE_JSON = Path(
    "/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/"
    "baseline_ckpt010200_trainseed123_valseed123_all32/"
    "evaluate/comp_plots/results.json"
)

AGG_OVERLAP_JSON = Path(
    "/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/"
    "AGG_mild_overlap_ckpt010002_trainseed123_valseed123_all32/"
    "evaluate/comp_plots/results.json"
)

OUTPUT_DIR = Path(
    "/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/"
    "paired_comparison_baseline_vs_agg_overlap"
)


def load_per_image(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "per_image" not in data:
        raise KeyError(f"{path} 中不存在 per_image 字段")

    return {
        int(idx): metrics
        for idx, metrics in data["per_image"].items()
    }


def mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    """返回均值及95%置信区间下界、上界。"""
    values = np.asarray(values, dtype=float)
    n = len(values)

    mean = float(np.mean(values))

    if n < 2:
        return mean, mean, mean

    std = float(np.std(values, ddof=1))
    margin = stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)

    return mean, mean - margin, mean + margin


def safe_wilcoxon(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)

    if np.allclose(values, 0):
        return 0.0, 1.0

    result = stats.wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
    )
    return float(result.statistic), float(result.pvalue)


def summarize_gain(name: str, gain: np.ndarray) -> dict:
    gain = np.asarray(gain, dtype=float)

    mean, ci_low, ci_high = mean_ci95(gain)

    # 检验平均改进量是否显著偏离0
    t_result = stats.ttest_1samp(gain, popmean=0.0)
    w_stat, w_p = safe_wilcoxon(gain)

    return {
        "metric": name,
        "n": len(gain),
        "mean_gain": mean,
        "std_gain": float(np.std(gain, ddof=1)),
        "median_gain": float(np.median(gain)),
        "min_gain": float(np.min(gain)),
        "max_gain": float(np.max(gain)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "improved_samples": int(np.sum(gain > 0)),
        "unchanged_samples": int(np.sum(np.isclose(gain, 0))),
        "worse_samples": int(np.sum(gain < 0)),
        "paired_t_pvalue": float(t_result.pvalue),
        "wilcoxon_pvalue": w_p,
        "wilcoxon_statistic": w_stat,
    }


def main() -> None:
    baseline = load_per_image(BASELINE_JSON)
    agg_overlap = load_per_image(AGG_OVERLAP_JSON)

    baseline_indices = set(baseline.keys())
    agg_indices = set(agg_overlap.keys())
    common_indices = sorted(baseline_indices & agg_indices)

    if not common_indices:
        raise RuntimeError("两个结果文件中没有共同样本")

    if baseline_indices != agg_indices:
        print("警告：两个方法的样本集合不完全一致")
        print("仅baseline存在：", sorted(baseline_indices - agg_indices))
        print("仅AGG+overlap存在：", sorted(agg_indices - baseline_indices))

    rows = []

    for idx in common_indices:
        base = baseline[idx]
        agg = agg_overlap[idx]

        rows.append({
            "sample": idx,

            "baseline_psnr": base["psnr"],
            "agg_overlap_psnr": agg["psnr"],
            # 正数表示AGG+overlap更好
            "psnr_gain": agg["psnr"] - base["psnr"],

            "baseline_ssim": base["ssim"],
            "agg_overlap_ssim": agg["ssim"],
            # 正数表示AGG+overlap更好
            "ssim_gain": agg["ssim"] - base["ssim"],

            "baseline_nrmse": base["nrmse"],
            "agg_overlap_nrmse": agg["nrmse"],
            # NRMSE越低越好，因此使用baseline-新方法
            "nrmse_reduction": base["nrmse"] - agg["nrmse"],
        })

    df = pd.DataFrame(rows).sort_values("sample")

    summaries = [
        summarize_gain("PSNR gain", df["psnr_gain"].to_numpy()),
        summarize_gain("SSIM gain", df["ssim_gain"].to_numpy()),
        summarize_gain(
            "NRMSE reduction",
            df["nrmse_reduction"].to_numpy(),
        ),
    ]

    summary_df = pd.DataFrame(summaries)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_sample_path = OUTPUT_DIR / "paired_per_sample.csv"
    summary_path = OUTPUT_DIR / "paired_summary.csv"

    df.to_csv(per_sample_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n================ 逐样本结果 ================")
    print(
        df[
            [
                "sample",
                "psnr_gain",
                "ssim_gain",
                "nrmse_reduction",
            ]
        ].to_string(index=False)
    )

    print("\n================ 配对统计 ================")

    for result in summaries:
        print(f"\n{result['metric']}")
        print(f"样本数：{result['n']}")
        print(
            f"平均改进：{result['mean_gain']:.6f} "
            f"± {result['std_gain']:.6f}"
        )
        print(f"中位数改进：{result['median_gain']:.6f}")
        print(
            "95% CI："
            f"[{result['ci95_low']:.6f}, "
            f"{result['ci95_high']:.6f}]"
        )
        print(
            f"改善/不变/变差："
            f"{result['improved_samples']}/"
            f"{result['unchanged_samples']}/"
            f"{result['worse_samples']}"
        )
        print(
            f"配对t检验 p={result['paired_t_pvalue']:.6g}"
        )
        print(
            f"Wilcoxon检验 p={result['wilcoxon_pvalue']:.6g}"
        )

    print("\n结果已保存：")
    print(per_sample_path)
    print(summary_path)


if __name__ == "__main__":
    main()