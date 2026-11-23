#!/usr/bin/env python3
"""Analytical upper-bound estimates for the two architecture sketches."""


def conv_params(cin, cout, k):
    return cin * cout * k * k + cout


def conv_macs(cin, cout, k, h):
    return h * h * cin * cout * k * k


def efficient_block_params(c, ratio, spatial_coeff, emb=512):
    # GN + spatial depthwise + FiLM(E->2C) + PW expansion/projection, incl. biases.
    return (2 * c + (spatial_coeff * c + c) + (2 * emb * c + 2 * c)
            + (ratio * c * c + ratio * c) + (ratio * c * c + c))


def efficient_block_macs(c, ratio, spatial_coeff, h, emb=512):
    return h * h * (spatial_coeff * c + 2 * ratio * c * c) + 2 * emb * c


def estimate(dims, enc_blocks, bottleneck_blocks, dec_blocks, ratios, spatial_coeffs, patch):
    emb = 512
    h = [patch, patch // 2, patch // 4]
    params = 128 * emb + emb + emb * emb + emb
    macs = 128 * emb + emb * emb
    params += conv_params(4, dims[0], 3)
    macs += conv_macs(4, dims[0], 3, h[0])
    for level, c in enumerate(dims):
        params += enc_blocks[level] * efficient_block_params(c, ratios[level], spatial_coeffs[level])
        macs += enc_blocks[level] * efficient_block_macs(c, ratios[level], spatial_coeffs[level], h[level])
        if level < 2:
            params += conv_params(c, dims[level + 1], 3)
            macs += conv_macs(c, dims[level + 1], 3, h[level + 1])
    c = dims[2]
    params += bottleneck_blocks * efficient_block_params(c, ratios[2], spatial_coeffs[2])
    macs += bottleneck_blocks * efficient_block_macs(c, ratios[2], spatial_coeffs[2], h[2])
    n = h[2] ** 2
    params += 4 * c * c + 6 * c
    macs += 4 * n * c * c + 2 * n * n * c
    for level in (2, 1, 0):
        if level < 2:
            params += conv_params(dims[level + 1], dims[level], 3)
            macs += conv_macs(dims[level + 1], dims[level], 3, h[level])
        params += conv_params(dims[level], dims[level], 1)
        macs += conv_macs(dims[level], dims[level], 1, h[level])
        params += dec_blocks[level] * efficient_block_params(
            dims[level], ratios[level], spatial_coeffs[level])
        macs += dec_blocks[level] * efficient_block_macs(
            dims[level], ratios[level], spatial_coeffs[level], h[level])
    params += 2 * dims[0] + conv_params(dims[0], 2, 3)
    macs += conv_macs(dims[0], 2, 3, h[0])
    return params, 2 * macs


CONFIGS = {
    "A": dict(dims=[160, 256, 384], enc_blocks=[3, 4, 5], bottleneck_blocks=4,
              dec_blocks=[3, 4, 4], ratios=[2, 3, 4], spatial_coeffs=[25, 25, 9]),
    # 3x3 + 1x7 + 7x1 on three 1/8-channel branches: (9+7+7)/8 = 23/8.
    "B-full-route-upper-bound": dict(
        dims=[160, 288, 416], enc_blocks=[3, 4, 5], bottleneck_blocks=5,
        dec_blocks=[3, 4, 4], ratios=[2, 3, 4], spatial_coeffs=[23 / 8] * 3),
}


for name, config in CONFIGS.items():
    results = [estimate(patch=p, **config) for p in (16, 32, 64)]
    expected = sum(w * result[1] for w, result in zip((0.2, 0.3, 0.5), results))
    print(name)
    print(f"  params: {results[0][0] / 1e6:.6f} M")
    for patch, (_, flops) in zip((16, 32, 64), results):
        print(f"  P{patch}: {flops / 1e9:.6f} GFLOPs")
    print(f"  expected: {expected / 1e9:.6f} GFLOPs")
