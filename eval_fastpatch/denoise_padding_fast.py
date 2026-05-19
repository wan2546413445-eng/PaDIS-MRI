import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import scipy.io
from diffusers import AutoencoderKL
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import sys
import time
import atexit
from collections import defaultdict

torch.manual_seed(2)

# ============================================================
# FastPatch profiling utilities
# ------------------------------------------------------------
# 仅用于统计 denoise_padding_fast.py 内部各阶段耗时。
# 默认关闭，不影响正常重建。
#
# 开启方式：
#   export PADIS_FASTPATCH_PROFILE=1
#
# 关闭方式：
#   unset PADIS_FASTPATCH_PROFILE
# ============================================================

FASTPATCH_PROFILE = os.environ.get("PADIS_FASTPATCH_PROFILE", "0") == "1"

_FASTPATCH_PROFILE_STATS = defaultdict(float)
_FASTPATCH_PROFILE_COUNTS = defaultdict(int)


def _cuda_sync_if_needed():
    """
    CUDA 异步执行会导致直接 time.time() 计时不准。
    profiling 打开时才同步，正常重建路径不增加开销。
    """
    if FASTPATCH_PROFILE and torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_add(name: str, elapsed_ms: float):
    """
    累计某一阶段耗时。
    """
    if not FASTPATCH_PROFILE:
        return
    _FASTPATCH_PROFILE_STATS[name] += float(elapsed_ms)
    _FASTPATCH_PROFILE_COUNTS[name] += 1


def _profile_start():
    """
    返回当前计时起点。
    """
    if not FASTPATCH_PROFILE:
        return None
    _cuda_sync_if_needed()
    return time.perf_counter()


def _profile_end(name: str, start_time):
    """
    结束某一阶段计时并写入统计。
    """
    if not FASTPATCH_PROFILE or start_time is None:
        return
    _cuda_sync_if_needed()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    _profile_add(name, elapsed_ms)


def print_fastpatch_profile():
    """
    程序退出时打印累计 profiling 结果。
    """
    if not FASTPATCH_PROFILE:
        return

    print("\n" + "=" * 72)
    print("[FastPatch Profiling Summary]")
    print("=" * 72)

    ordered_keys = [
        "get_indices_ms",
        "prepare_meta_clone_ms",
        "allocate_buffers_ms",
        "gather_patch_inputs_ms",
        "network_forward_ms",
        "scatter_patch_outputs_ms",
        "finalize_center_copy_ms",
        "denoised_from_patches_total_ms",
    ]

    total_denoise_calls = _FASTPATCH_PROFILE_COUNTS.get(
        "denoised_from_patches_total_ms", 0
    )

    print(f"denoisedFromPatches calls: {total_denoise_calls}")

    for key in ordered_keys:
        count = _FASTPATCH_PROFILE_COUNTS.get(key, 0)
        total = _FASTPATCH_PROFILE_STATS.get(key, 0.0)
        avg = total / count if count > 0 else 0.0
        print(f"{key:36s} | total = {total:10.3f} ms | avg = {avg:8.3f} ms | n = {count}")

    print("=" * 72 + "\n")


if FASTPATCH_PROFILE:
    atexit.register(print_fastpatch_profile)

def getIndices(spaced, patches, pad, psize, freezeindex=False):
    """
    与原始 getIndices 完全一致；
    仅额外统计随机偏移 patch 坐标生成耗时。
    """
    _t_profile = _profile_start()

    a, b = 0, 0  # Default values when pad = 0
    if pad > 0:
        a = random.randint(0, pad - 1)
        b = random.randint(0, pad - 1)

    if freezeindex:
        a = 0
        b = 0

    indices = []
    for p in range(patches):
        for q in range(patches):
            indices.append([
                spaced[p] + a,
                spaced[p] + a + psize,
                spaced[q] + b,
                spaced[q] + b + psize
            ])

    _profile_end("get_indices_ms", _t_profile)
    return indices
def denoisedFromPatches(
    net,
    x,
    t_hat,
    latents_pos,
    class_labels,
    indices,
    pad=96,
    t_goal=-1,
    avg=1,
    spaced=[],
    wrong=False
):
    """
    与原始 denoisedFromPatches 完全一致；
    仅增加各阶段 profiling，不改变任何计算逻辑。
    """

    _t_total = _profile_start()

    # ------------------------------------------------------------
    # 0. 原始特殊路径：若传入 spaced，则重新生成 indices
    # ------------------------------------------------------------
    if len(spaced) > 1:
        indices = getIndices(spaced, 5, 24, 56)

    # ------------------------------------------------------------
    # 1. clone / meta 准备
    # ------------------------------------------------------------
    _t_prepare = _profile_start()

    if wrong:
        x_hat = x
    else:
        x_hat = torch.clone(x)

    channels = len(x_hat[0, :, 0, 0])
    N = len(x_hat[0, 0, 0, :])
    psize = indices[0][1] - indices[0][0]
    patches = len(indices)
    pad = int((N - np.sqrt(patches) * psize))

    _profile_end("prepare_meta_clone_ms", _t_prepare)

    # ------------------------------------------------------------
    # 2. 分配 output / x_input / pos_input
    # ------------------------------------------------------------
    _t_alloc = _profile_start()

    output = torch.zeros_like(x_hat)
    x_input = torch.zeros(
        patches,
        channels,
        psize,
        psize
    ).to(torch.device("cuda"))

    pos_input = torch.zeros(
        patches,
        2,
        psize,
        psize
    ).to(torch.device("cuda"))

    _profile_end("allocate_buffers_ms", _t_alloc)

    # ------------------------------------------------------------
    # 3. 逐 patch 收集网络输入
    #    保持作者原始代码完全不变，只统计耗时
    # ------------------------------------------------------------
    _t_gather = _profile_start()

    for i in range(patches):
        z = indices[i]
        x_input[i, :, :, :] = torch.squeeze(
            x_hat[0, :, z[0]:z[1], z[2]:z[3]]
        )
        pos_input[i, :, :, :] = torch.squeeze(
            latents_pos[:, :, z[0]:z[1], z[2]:z[3]]
        )

    _profile_end("gather_patch_inputs_ms", _t_gather)

    # ------------------------------------------------------------
    # 4. 网络前向
    # ------------------------------------------------------------
    _t_net = _profile_start()

    bigout = net(
        x_input,
        t_hat,
        pos_input,
        class_labels
    ).to(torch.float64)

    _profile_end("network_forward_ms", _t_net)

    # ------------------------------------------------------------
    # 5. 逐 patch 写回整图
    #    保持作者原始代码完全不变，只统计耗时
    # ------------------------------------------------------------
    _t_scatter = _profile_start()

    for i in range(patches):
        z = indices[i]
        x_patch = x_hat[0, :, z[0]:z[1], z[2]:z[3]]
        output[0, :, z[0]:z[1], z[2]:z[3]] += bigout[i, :, :, :]
        output[0, :, z[0]:z[1], z[2]:z[3]] -= x_patch

    x_hat = x_hat + output

    _profile_end("scatter_patch_outputs_ms", _t_scatter)

    # ------------------------------------------------------------
    # 6. 原始中心区域回填
    # ------------------------------------------------------------
    _t_finalize = _profile_start()

    temp = t_goal + torch.randn_like(x_hat) * t_goal
    temp[:, :, pad:N - pad, pad:N - pad] = x_hat[:, :, pad:N - pad, pad:N - pad]

    _profile_end("finalize_center_copy_ms", _t_finalize)

    # ------------------------------------------------------------
    # 7. 整个 denoisedFromPatches 总耗时
    # ------------------------------------------------------------
    _profile_end("denoised_from_patches_total_ms", _t_total)

    return temp