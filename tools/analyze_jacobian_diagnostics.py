#!/usr/bin/env python3
"""Summarize Phase A PaDIS-DPS Jacobian observations (no reconstruction)."""
import argparse
import json
from pathlib import Path


def summarize(source, figure=None):
    source = Path(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = sorted(data["records"], key=lambda r: (r["outer_level"], r["inner_step"]))
    lines = ["PaDIS-DPS Jacobian Phase A | raw MRI operator | fixed partition per observation",
             "Interpretation: these are cross-noise-stage snapshots; state, sigma and partition all vary.",
             "This is NOT a same-state partition intervention or evidence of a partition-caused change.",
             "Seam band is only a two-pixel proximity statistic, not the entire Jacobian influence area.",
             "outer inner sigma shift cos(full) cos(FOV) cos(seam) cos(interior) padding_E fraction ratio status"]
    for r in rows:
        fmt = lambda v: "n/a" if v is None else f"{v:.4g}"
        lines.append(
            f"{r['outer_level']:>5} {r['inner_step']:>5} {r['sigma']:.5g} "
            f"({r['shift_row']},{r['shift_col']}) "
            f"{fmt(r['jacobian_cosine'])} {fmt(r['fov_cosine'])} "
            f"{fmt(r['boundary']['cosine'])} {fmt(r['interior']['cosine'])} "
            f"{fmt(r['padding_g_energy_fraction'])} "
            f"{fmt(r['jacobian_norm_ratio'])} {r['status']}")
    lines.extend(["", "Patch seam and interior: per-complex-pixel RMS of gJ and v",
                  "outer region pixels gJ_RMS v_RMS cosine"])
    for r in rows:
        for region in ("boundary", "interior", "padding"):
            s = r[region]
            cosine = "n/a" if s["cosine"] is None else f"{s['cosine']:.5f}"
            grms = "n/a" if s["g_rms"] is None else f"{s['g_rms']:.6g}"
            vrms = "n/a" if s["v_rms"] is None else f"{s['v_rms']:.6g}"
            lines.append(f"{r['outer_level']} {region} {s['pixels']} "
                         f"{grms} {vrms} {cosine}")
    summary = "\n".join(lines) + "\n"
    target = source.parent / "jacobian_summary.txt"
    target.write_text(summary, encoding="utf-8")
    print(summary)

    if figure is not None and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        xs = [r["outer_level"] for r in rows]
        for name, selector in (
            ("All padded", lambda r: r["jacobian_cosine"]),
            ("FOV", lambda r: r["fov_cosine"]),
            ("Seam band", lambda r: r["boundary"]["cosine"]),
            ("Patch interior", lambda r: r["interior"]["cosine"]),
        ):
            ax.plot(xs, [selector(r) for r in rows], marker="o", label=name)
        ax.set_xlabel("DPS outer level (zero-based)")
        ax.set_title("Cross-stage snapshots (not matched-partition effects)", fontsize=9)
        ax.set_ylabel("cos($J_D^H v$, $v$)")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure, dpi=180)
        plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()
    summarize(args.json_file, args.figure)
