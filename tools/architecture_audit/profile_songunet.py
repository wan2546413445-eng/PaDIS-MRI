#!/usr/bin/env python3
"""Static parameter/MAC audit for the configured PaDIS SongUNet.

Mirrors train/padis-mri/training/networks.py without importing PyTorch.
Counts learned convolution/linear MACs and QK/AV attention MACs. Elementwise
ops, normalization, SiLU, softmax, and fixed [1,1] resampling are excluded from
the conventional MAC total and documented in the report.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def conv_params(cin, cout, k, bias=True):
    return cin * cout * k * k + (cout if bias else 0)


def conv_macs(cin, cout, k, h, w):
    return h * w * cin * cout * k * k


def linear_params(cin, cout, bias=True):
    return cin * cout + (cout if bias else 0)


def block(name, source_key, cin, cout, emb, h, w, *, up=False, down=False, attention=False):
    oh, ow = (h * 2, w * 2) if up else (h // 2, w // 2) if down else (h, w)
    parts = {
        "conv0": conv_macs(cin, cout, 3, oh, ow),
        "conv1": conv_macs(cout, cout, 3, oh, ow),
        "affine": emb * cout,
    }
    params = 2 * cin
    params += conv_params(cin, cout, 3)
    params += linear_params(emb, cout)
    params += 2 * cout
    params += conv_params(cout, cout, 3)
    if cin != cout or up or down:
        parts["skip_projection"] = conv_macs(cin, cout, 1, oh, ow)
        params += conv_params(cin, cout, 1)
    if attention:
        n = oh * ow
        parts["attention_qkv"] = n * cout * 3 * cout
        parts["attention_qk_av"] = 2 * n * n * cout
        parts["attention_projection"] = n * cout * cout
        params += 2 * cout
        params += conv_params(cout, 3 * cout, 1)
        params += conv_params(cout, cout, 1)
    stage = (
        "Downsampling" if name.startswith("enc_down") else
        "Encoder" if name.startswith("enc_") else
        "Bottleneck" if name.startswith("bottleneck") else
        "Upsampling" if name.startswith("dec_up") else
        "Decoder"
    )
    return {
        "name": name, "source_key": source_key, "stage": stage,
        "resolution": f"{oh}x{ow}", "channels": f"{cin}->{cout}",
        "params": params, "macs": sum(parts.values()), "parts": parts,
        "attention": attention,
    }, oh, ow


def audit(c, patch):
    emb = 4 * c
    width = 2 * c
    rows = []
    mapping_params = linear_params(c, emb) + linear_params(emb, emb)
    mapping_macs = c * emb + emb * emb
    rows.append({"name": "mapping", "stage": "Noise embedding", "resolution": "1x1",
                 "channels": f"{c}->{emb}", "params": mapping_params,
                 "macs": mapping_macs, "parts": {"linear": mapping_macs}, "attention": False})

    h = w = patch
    stem_params = conv_params(4, c, 3)
    stem_macs = conv_macs(4, c, 3, h, w)
    rows.append({"name": "stem", "source_key": "512x512_conv", "stage": "Input stem",
                 "resolution": f"{h}x{w}", "channels": f"4->{c}", "params": stem_params,
                 "macs": stem_macs, "parts": {"conv": stem_macs}, "attention": False})

    cin = c
    source_res = [512, 256, 128]
    skip_channels = [c]
    for level in range(3):
        if level > 0:
            row, h, w = block(f"enc_down_L{level}", f"{source_res[level]}x{source_res[level]}_down",
                              width, width, emb, h, w, down=True)
            rows.append(row)
            skip_channels.append(width)
        for idx in range(4):
            bcin = cin if level == 0 and idx == 0 else width
            row, h, w = block(f"enc_L{level}_B{idx}", f"{source_res[level]}x{source_res[level]}_block{idx}",
                              bcin, width, emb, h, w)
            rows.append(row)
            skip_channels.append(width)
            cin = width

    row, h, w = block("bottleneck_B0", "128x128_in0", width, width, emb, h, w, attention=True)
    rows.append(row)
    row, h, w = block("bottleneck_B1", "128x128_in1", width, width, emb, h, w)
    rows.append(row)

    for level in (2, 1, 0):
        if level < 2:
            row, h, w = block(f"dec_up_to_L{level}", f"{source_res[level]}x{source_res[level]}_up",
                              width, width, emb, h, w, up=True)
            rows.append(row)
        for idx in range(5):
            skip_c = skip_channels.pop()
            row, h, w = block(f"dec_L{level}_B{idx}", f"{source_res[level]}x{source_res[level]}_block{idx}",
                              width + skip_c, width, emb, h, w)
            rows.append(row)

    head_params = 2 * width + conv_params(width, 2, 3)
    head_macs = conv_macs(width, 2, 3, h, w)
    rows.append({"name": "head", "source_key": "512x512_aux_norm/aux_conv", "stage": "Output head",
                 "resolution": f"{h}x{w}", "channels": f"{width}->2", "params": head_params,
                 "macs": head_macs, "parts": {"conv": head_macs}, "attention": False})
    assert not skip_channels

    stages = defaultdict(lambda: {"params": 0, "macs": 0})
    for row in rows:
        stages[row["stage"]]["params"] += row["params"]
        stages[row["stage"]]["macs"] += row["macs"]
    total_macs = sum(row["macs"] for row in rows)
    attn_macs = sum(sum(v for k, v in row["parts"].items() if k.startswith("attention")) for row in rows)
    return {"cbase": c, "patch": patch, "total_params": sum(row["params"] for row in rows),
            "total_macs": total_macs, "attention_macs": attn_macs,
            "stages": dict(stages), "blocks": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [audit(c, p) for c in (96, 128) for p in (16, 32, 64)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for x in results:
        print(f"C{x['cbase']} P{x['patch']}: {x['total_params']/1e6:.6f} M params, "
              f"{x['total_macs']/1e9:.6f} GMACs, {2*x['total_macs']/1e9:.6f} GFLOPs")


if __name__ == "__main__":
    main()
