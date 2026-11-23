#!/usr/bin/env python3
"""Reproducible parameter/MAC audit for SongUNet C96 and A0.

Conventions match ``profile_songunet.py``: one learned multiply-accumulate is
one MAC and two FLOPs.  Normalization, activations, softmax, concatenation,
elementwise operations, and fixed [1,1] resampling are excluded.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'train' / 'padis-mri'))

from profile_songunet import audit as audit_songunet  # noqa: E402
from training.efficient_unet import EfficientSongUNet  # noqa: E402
from training.networks import SongUNet  # noqa: E402


def conv_params(cin, cout, k, bias=True):
    return cin * cout * k * k + (cout if bias else 0)


def linear_params(cin, cout, bias=True):
    return cin * cout + (cout if bias else 0)


def original_block(name, source_key, cin, cout, emb, h, w, *, up=False, down=False, attention=False):
    oh, ow = (h * 2, w * 2) if up else (h // 2, w // 2) if down else (h, w)
    parts = {
        'conv0': oh * ow * cin * cout * 9,
        'conv1': oh * ow * cout * cout * 9,
        'sigma_projection': emb * cout,
    }
    params = 2 * cin + conv_params(cin, cout, 3)
    params += linear_params(emb, cout) + 2 * cout + conv_params(cout, cout, 3)
    if cin != cout or up or down:
        parts['skip_projection'] = oh * ow * cin * cout
        params += conv_params(cin, cout, 1)
    if attention:
        n = oh * ow
        parts['attention_qkv'] = n * cout * 3 * cout
        parts['attention_qk_av'] = 2 * n * n * cout
        parts['attention_projection'] = n * cout * cout
        params += 2 * cout + conv_params(cout, 3 * cout, 1) + conv_params(cout, cout, 1)
    stage = (
        'Downsampling' if name.startswith('enc_down') else
        'Bottleneck' if name.startswith('bottleneck') else
        'Upsampling' if name.startswith('dec_up') else 'Encoder'
    )
    return {
        'name': name,
        'source_key': source_key,
        'block_type': 'UNetBlock',
        'stage': stage,
        'resolution': f'{oh}x{ow}',
        'channels': f'{cin}->{cout}',
        'params': params,
        'macs': sum(parts.values()),
        'parts': parts,
        'attention': attention,
    }, oh, ow


def efficient_block(name, source_key, cin, cout, emb, h, w, expansion=2, kernel=5):
    hidden = expansion * cout
    parts = {
        'pointwise_expansion': h * w * cin * hidden,
        'depthwise_spatial': h * w * hidden * kernel * kernel,
        'sigma_projection': emb * hidden,
        'pointwise_projection': h * w * hidden * cout,
    }
    params = 2 * cin
    params += conv_params(cin, hidden, 1)
    params += hidden * kernel * kernel + hidden
    params += linear_params(emb, hidden)
    params += 2 * hidden
    params += conv_params(hidden, cout, 1)
    if cin != cout:
        parts['skip_projection'] = h * w * cin * cout
        params += conv_params(cin, cout, 1)
    stage = 'Bottleneck' if name.startswith('bottleneck') else 'Decoder' if name.startswith('dec_') else 'Encoder'
    return {
        'name': name,
        'source_key': source_key,
        'block_type': 'EfficientDiffusionBlock',
        'stage': stage,
        'resolution': f'{h}x{w}',
        'channels': f'{cin}->{cout} (hidden={hidden})',
        'params': params,
        'macs': sum(parts.values()),
        'parts': parts,
        'attention': False,
    }


def audit_a0(c, patch):
    emb = 4 * c
    width = 2 * c
    rows = []

    mapping_params = linear_params(c, emb) + linear_params(emb, emb)
    mapping_macs = c * emb + emb * emb
    rows.append({
        'name': 'mapping', 'stage': 'Noise embedding', 'resolution': '1x1',
        'channels': f'{c}->{emb}', 'params': mapping_params, 'macs': mapping_macs,
        'parts': {'linear': mapping_macs}, 'attention': False,
    })

    h = w = patch
    stem_params = conv_params(4, c, 3)
    stem_macs = h * w * 4 * c * 9
    rows.append({
        'name': 'stem', 'source_key': '512x512_conv', 'stage': 'Input stem',
        'resolution': f'{h}x{w}', 'channels': f'4->{c}', 'params': stem_params,
        'macs': stem_macs, 'parts': {'conv': stem_macs}, 'attention': False,
    })

    source_res = [512, 256, 128]
    skip_channels = [c]
    cin = c
    for level in range(3):
        if level > 0:
            row, h, w = original_block(
                f'enc_down_L{level}', f'{source_res[level]}x{source_res[level]}_down',
                width, width, emb, h, w, down=True,
            )
            rows.append(row)
            skip_channels.append(width)
        for idx in range(4):
            bcin = cin if level == 0 and idx == 0 else width
            row = efficient_block(
                f'enc_L{level}_B{idx}', f'{source_res[level]}x{source_res[level]}_block{idx}',
                bcin, width, emb, h, w,
            )
            rows.append(row)
            skip_channels.append(width)
            cin = width

    row, h, w = original_block(
        'bottleneck_B0', '128x128_in0', width, width, emb, h, w, attention=True,
    )
    rows.append(row)
    rows.append(efficient_block('bottleneck_B1', '128x128_in1', width, width, emb, h, w))

    for level in (2, 1, 0):
        if level < 2:
            row, h, w = original_block(
                f'dec_up_to_L{level}', f'{source_res[level]}x{source_res[level]}_up',
                width, width, emb, h, w, up=True,
            )
            rows.append(row)
        for idx in range(5):
            skip_c = skip_channels.pop()
            rows.append(efficient_block(
                f'dec_L{level}_B{idx}', f'{source_res[level]}x{source_res[level]}_block{idx}',
                width + skip_c, width, emb, h, w,
            ))

    head_params = 2 * width + conv_params(width, 2, 3)
    head_macs = h * w * width * 2 * 9
    rows.append({
        'name': 'head', 'source_key': '512x512_aux_norm/aux_conv', 'stage': 'Output head',
        'resolution': f'{h}x{w}', 'channels': f'{width}->2', 'params': head_params,
        'macs': head_macs, 'parts': {'conv': head_macs}, 'attention': False,
    })
    assert not skip_channels

    stages = defaultdict(lambda: {'params': 0, 'macs': 0})
    for row in rows:
        stages[row['stage']]['params'] += row['params']
        stages[row['stage']]['macs'] += row['macs']
    total_macs = sum(row['macs'] for row in rows)
    attention_macs = sum(
        sum(value for key, value in row['parts'].items() if key.startswith('attention'))
        for row in rows
    )
    return {
        'network': 'A0 EfficientSongUNet',
        'cbase': c,
        'patch': patch,
        'total_params': sum(row['params'] for row in rows),
        'total_macs': total_macs,
        'total_flops': 2 * total_macs,
        'attention_macs': attention_macs,
        'stages': dict(stages),
        'blocks': rows,
    }


def instantiate_params(model_class):
    model = model_class(
        img_resolution=512,
        in_channels=4,
        out_channels=2,
        model_channels=96,
        channel_mult=[2, 2, 2],
        num_blocks=4,
        dropout=0.05,
        attn_resolutions=[16],
        channel_mult_noise=1,
        embedding_type='positional',
        encoder_type='standard',
        decoder_type='standard',
        resample_filter=[1, 1],
    )
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=THIS_DIR / 'efficient_unet_A0_audit.json')
    args = parser.parse_args()

    song_actual = instantiate_params(SongUNet)
    a0_actual = instantiate_params(EfficientSongUNet)
    results = []
    for patch in (16, 32, 64):
        song = audit_songunet(96, patch)
        song['network'] = 'SongUNet C96'
        song['total_flops'] = 2 * song['total_macs']
        a0 = audit_a0(96, patch)
        assert song['total_params'] == song_actual
        assert a0['total_params'] == a0_actual
        results.extend([song, a0])

    output = {
        'convention': '1 MAC = 2 FLOPs; learned convolution/linear and QK/AV attention only',
        'instantiated_params': {'SongUNet C96': song_actual, 'A0 EfficientSongUNet': a0_actual},
        'results': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')

    for patch in (16, 32, 64):
        for result in [item for item in results if item['patch'] == patch]:
            print(
                f"{result['network']:22s} P{patch}: "
                f"{result['total_params'] / 1e6:9.6f} M params, "
                f"{result['total_macs'] / 1e9:9.6f} GMACs, "
                f"{result['total_flops'] / 1e9:9.6f} GFLOPs"
            )


if __name__ == '__main__':
    main()

