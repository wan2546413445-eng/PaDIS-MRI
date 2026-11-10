# PaDIS-MRI 快速重建对照

## CT-CG-DDIM 首轮实验

在 PaDIS-MRI `main-WSY` 仓库根目录解压 `eval_fast_ct_cg_ddim.zip`。此包基于提交 `72b18b48cbfca1d689b768de2d726e16321fd962`，包含运行入口、采样器、FH 精度与诊断模块，以及 PaDIS-MRI 和 Free Hunch 的许可证。

目标仓库提供 `eval/evaluator_fixed_seed123.py`、`eval/recon.py`、`eval/denoise_padding.py`、`eval/inverse_operators.py`、`eval/utils.py`、`dnnlib/`、`torch_utils/` 和 `train/padis-mri/`。沿用服务器上可运行现有 PaDIS 推理的 Python 环境、PyTorch/CUDA 与 BART 配置。评估依赖包括 NumPy、SigPy、scikit-image、Matplotlib、Pillow、SciPy、Diffusers、Click 和 tqdm。运行实验需要现有 OPA EMA checkpoint 和带 `mask_7` 的 T1-FLAIR 验证样本。

采样流程参考 [Yang 等人的 Algorithm 1 与表 5](https://arxiv.org/html/2512.18161v1#S3.SS2)。在 PaDIS 的 VE 调度上，同层完成两轮分块去噪，融合后在真实 FOV 做一次各向同性 CG，再从 CG 修正后的干净估计进入下一层。默认 `steps=200`、`inner_loops=2`、`ddim_eta=0.8`。首轮 MRI 参数为 `cg_lambda=0.1`、`cg_iters=5`、`cg_gain=1.0`。checkpoint 目录中的 `lam2p0` 对应报告中的 OPA 权重 λ=1.0。

```bash
cd /mnt/SSD/wsy/projects/PaDIS-MRI-main
unzip -o /path/to/eval_fast_ct_cg_ddim.zip
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
export MPLBACKEND=Agg
export PYTHONPATH="$PWD:$PWD/train/padis-mri:$PWD/eval${PYTHONPATH:+:$PYTHONPATH}"
export PADIS_PATCH_BATCH="${PADIS_PATCH_BATCH:-49}" PADIS_PATCH_CHECKPOINT=0

RUNS=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-runs/overlap_independent_training-runs/brain/32dB
RUN=overnight_overlap_independent_center_symmetric_lam2p0_active64_bgpu1_s16s32s64_p020305_b4_seed123_geomidentical_spatialuniform
MODEL="$RUNS/$RUN/00000-aapm_3-uncond-ddpmpp-pedm-gpus1-batch4-fp32-overlap-independent-center-lam2p0-p64-align-symmetric-geom-identical-spatial-uniform/network-snapshot-005000.pkl"
VAL=/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB
OUT=/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/eval_fast_OPA_lam2p0_T1FLAIR_R7_ct_cg_ddim_s200x2_seed123_19_26

python -u eval_fast/run.py --mode ct_cg_ddim \
  --model_path "$MODEL" --val_dir "$VAL" --save_dir "$OUT" \
  --sample_indices 19,26 --image_size 384 --mask_select 7 --seed 123 --pad 64 --psize 64 \
  --steps 200 --inner_loops 2 --ddim_eta 0.8 --cg_lambda 0.1 --cg_iters 5 --cg_gain 1.0
```

两张图分别重置 seed 123。重复实验时为 `OUT` 指定新目录。可用 `python eval_fast/smoke_ct_cg_ddim.py` 在 CPU 上检查两层两轮的调用顺序、跨层状态和末层返回。

运行结束后核对实际操作计数和全图结果。

```bash
python - "$OUT/evaluate" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
operations = json.loads((root / "config.json").read_text())["operations_per_sample"]
timing = json.loads((root / "timing.json").read_text())["recon_seconds_per_sample"]
metrics = json.loads((root / "comp_plots/results.json").read_text())["per_image"]
for sample in ("19", "26"):
    counts = operations[sample]
    assert (counts["main_denoise"], counts["cg_solves"], counts["dps_backprops"]) == (400, 200, 0), counts
    assert counts["cg_iterations"] == 1000, counts
    assert (root / "recons" / f"recon_patch_{sample}.npy").is_file()
    print(f"sample {sample} | 400/200/0 | {timing[sample]:.2f} s | {metrics[sample]}")
PY
```

全图 PSNR、SSIM 和 NRMSE 沿用 `post_eval_normalize` 的 ACS-99 归一化口径。19 号将 `recon_patch_19.npy` 送入既有 MATLAB ROI 流程，使用相同 GT、归一化和 ROI mask 做前景对照。26 号用 `timing.json` 的单张实测时间判断是否接近 100 秒。总时间从重建调用前到 CUDA 同步后，权重加载、样本 I/O、结果写盘和指标计算在计时范围外。

比较纯 DPS、旧 Hybrid 和新模式时固定 GPU、权重、输入、patch batching、checkpointing 与计时范围，按整套采样配置报告质量和速度。首轮比较保持 profiling 关闭。如需拆分耗时，另跑一次并加 `--profile_ct`，`timing.json` 的 `gpu_seconds_per_sample` 记录分块去噪（含切拼）和 CG 求解的 CUDA event 时间。每张图同步后统一汇总。

## FH 混合采样

FH 对照沿用原 v3 的命令和缓存。若需从头生成 FH 训练缓存，从原 `eval_fast_fh_v3.zip` 取 `tools/build_fh_covariance.py`。配对启动脚本 `bash/eval_fast_fh_T1FLAIR_R7_paired.sh` 由目标仓库提供。

| R7 组别 | 第 1–39 层 | 第 40–78 层 |
| --- | --- | --- |
| `isotropic` | 原 CG，`cg_lambda=0.1` | 原 DPS |
| `fh` | 数据协方差初始化、跨层传递、在线 BFGS、协方差 CG | 原 DPS |

配对脚本固定 `lam2p0` checkpoint、T1-FLAIR、mask 7、seed 123、FOV 384、pad 64、patch 64、78×10、CG 5 次迭代、gain 0.1、DPS zeta 3.0。checkpoint 路径和验证目录取自原四组 `eval_fast` 续跑配置，可用 `MODEL` 和 `VAL` 覆盖。直接调用入口时，`--cg_covariance` 默认 `isotropic`，普通 `hybrid` 的默认 DPS 尾段仍是 20 层。

## 1. 生成训练缓存

```bash
TRAIN_PT=/mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB/noisy.pt
export FH_COV_PATH="$PWD/fh_cache/brain_train_d384_acs24_v3.npz"
python tools/build_fh_covariance.py --train_pt "$TRAIN_PT" --output "$FH_COV_PATH"
```

训练路径取自 `bash/train_overlap_independent.sh` 的默认目录。若训练时覆盖了目录，按实际训练日志填写。传入的 `noisy.pt` 和同目录下的 `ksp/<训练样本编号>.pt` 必须来自与评估患者互斥的训练集，并和本次重建使用相同的预处理强度单位。脚本从 `x_est_gt` 逐张计算实部、虚部的二维正交 DCT-II，用 Welford 累积总体方差。图像使用训练预处理保存的原生 384×384 方向和 ACS99 强度。这份全图 DCT 协方差是 FH 对照的初始化，其中每个基函数覆盖全图，无法为脑内和背景独立分配不确定性。旧版缓存需要重新生成。

## 2. 设置测量噪声方差

对已白化、归一化的训练多线圈 k-space 做中心化正交逆 FFT。根据非舍入量级的行列范围识别有效 FOV，排除预处理补零条带。在有效 FOV 内向内退 4 像素，取四个 24×24 角块，选择线圈功率中位数最低的两个。每个角块先按功率剔除亮离群点，再单独按线圈去均值，合并估计复数线圈协方差。每张图取平均线圈方差的一半，并乘有效 FOV 的面积占比，近似全网格正交 FFT 后每个实分量的边际方差。FOV 内背景仍可能有组织信号或伪影，因此这是近似校准。该标量模型假设实虚等方差，忽略零填充引入的 k-space 相关性。

配对脚本从 v3 缓存读取上述训练方差的中位数，用于 smoke 和全组。缓存同时记录估噪分位数、FOV 计数及线圈协方差偏离单位阵的程度，供检查白化假设。构建过程只读训练 k-space 和训练图像。`32dB` 是数据设置标签。单独调用 `eval_fast/run.py` 时，`--fh_noise_var` 可用于训练集或外部独立校准所得的同单位标量。正式评估前固定该值。

评估集的 `noise_var_*.pt` 源自掩膜前的完整测量，不用于 FH 校准。FH 采样使用 evaluator 返回的欠采样测量和既有灵敏度算子，评估 GT 用于重建后的指标和图。两组继续共用原有验证文件的预处理，其中 BART 白化、强度归一化和灵敏度图在施加 R7 掩膜前准备。本次配对衡量该共同预处理下的 FH 增量。严格只使用 R7 实采样数据的评估还需要用实采样 ACS 或独立扫描校准为两组重新准备输入。

## 3. 跑单张和 32 张配对

```bash
# sample 19，两组分别启动
GPU=2 GROUP=isotropic SAMPLE=19 bash bash/eval_fast_fh_T1FLAIR_R7_paired.sh
GPU=2 GROUP=fh        SAMPLE=19 bash bash/eval_fast_fh_T1FLAIR_R7_paired.sh

# 顺序跑 sample 0–31，每张先普通组再 FH，重用已完成的 sample 19
GPU=2 bash bash/eval_fast_fh_T1FLAIR_R7_paired.sh
```

`OUT_ROOT/isotropic/sample_ID/` 与 `OUT_ROOT/fh/sample_ID/` 各自保存 `run.log`、`evaluate/recons/`、`evaluate/comp_plots/`、`evaluate/config.json` 和 `evaluate/timing.json`。默认输出根目录后缀为 `_fh_v3`。重启跳过完整样本，归档中断目录后重跑该样本。FH 的 `.done` 记录 v3 缓存指纹，旧版完成标记或其他缓存生成的结果会归档重跑。改换权重、数据、秩上限或 profiling 设置时请用独立 `OUT_ROOT`。

`config.json` 记录 FH 训练文件、使用的噪声标量及其来源，以及每张图的实际采样次数和 BFGS 跳过原因。若整张图没有接受任何 BFGS 更新，`run.log` 会提示。默认各组每张有 780 主 sweep、390 次 DPS 反传、390 次 CG solve、1950 次 CG 迭代。FH 另有 38 次跨层传递和 1950 次 CG 内精度调用。标准 VE 调度在第 12–32 层满足探测范围，最多额外 21 次 sweep。单个 sweep 可能含多个 patch 前向批次。

`timing.json` 沿用 CUDA 同步后的单张重建墙钟时间，包含 FH 初始化、传递、探测、BFGS 和 CG。离线统计、加载缓存、样本 I/O 和后评估分开。需要拆分时设 `PROFILE_FH=1`，CUDA events 汇总到 `evaluate/fh_profile.json`，包含探测前向、协方差更新、CG 总时间及 CG 内精度时间。精度时间是 CG 子项。两组耗时比较时使用同样的 profiling 和 patch batch 设置。

## 实现与核对

- `x` 是含 padding 的复数 VE 状态。网络 patch 取自 `z = x + sigma * randn_like(x)` 的两个实通道，拼接预测裁到中央 FOV 后用于 DC。复数标准正态的每个实分量方差为 1/2，因此 FH 用 `v = sigma²/2`。BFGS 在合格层的首次 sweep 使用相邻采样状态的 FOV 位移，扰动按实通道 RMS 缩放至 `0.01*sigma`，在同一 `z`、patch 索引和位置编码上前向。额外前向保存并恢复 Torch RNG。
- 初始精度在 DCT 方差下限 `1e-8` 上构造，跨层更新只加对角。逆 BFGS 以 `q = v*(m'−m)` 更新精度，保留正曲率割线。默认总秩 16，可通过 `FH_MAX_RANK` 调整。对相对对角精度的低秩修正做谱截断，保留模式的相对精度特征值下限为 `1e-4`。精度在 GPU 上通过 DCT 和低秩乘法应用。
- FH 求解 `(A*A + fh_noise_var*P) delta = A*(y - A m)`。增量按原 gain 加回 `x`，后 39 层接原 DPS。此实验采用图像域 CG 修正，协方差公式参考 Free Hunch §3.1–3.5、附录 A、J 及官方实现。见 [论文](https://arxiv.org/html/2410.11149v2)、[在线更新源码](https://github.com/AaltoML/free-hunch/blob/main/conditioning_utils/online_update_bfgs.py)、[离线 DCT 源码](https://github.com/AaltoML/free-hunch/blob/main/do_frequency_analysis.py) 和 `LICENSE.free-hunch`。

本地 CPU 核对使用真实多线圈算子、真实 VE 调度和仅将分块临时张量改在 CPU 分配的测试副本，搭配小尺寸合成测量和替身去噪器。固定 seed 的 78×10 普通 hybrid 与原 ZIP 逐元素一致，FH 探测前后随机数流一致。稠密算例通过了初始化、跨层传递、BFGS 割线、截断正定性、5 步 CG 和退化为普通 CG 的核对。有效 FOV 估噪使用含中心信号及单轴或双轴补零的合成多线圈数据核对。真实训练集校准、sample 19 的质量和耗时需在服务器实测。

## 单样本 FH 诊断

在已装 v3 的项目根目录解压诊断补丁，然后运行：

```bash
unzip -o eval_fast_fh_v3_diag.zip
export FH_COV_PATH="$PWD/fh_cache/brain_train_d384_acs24_v3.npz"
GPU=3 SAMPLE=19 FH_DIAGNOSTICS=1 bash bash/eval_fast_fh_T1FLAIR_R7_paired.sh
```

两组沿用原 v3 设置。在 FH 轨迹的第 20、32、39 步的第一个内循环，用同一去噪预测和同一残差求出普通 CG、DCT 加跨步传递、完整 FH 三种修正量。额外求出的两种修正量只做记录，采样仍用完整 FH。两组都保存第 39 步结束后的采样状态和当步最后一次去噪预测，并与最终重建对照。前景统计使用 GT 幅度超过其第 99 百分位的 10% 的像素作为近似前景，GT 只在采样完成后参与诊断。

诊断输出根目录自动加 `_diag`，配对报告是其中的 `fh_diagnostic_pair_19.json`。重跑同一命令可跳过已完成组。诊断包含额外 CG 和数据传输，耗时不用于性能对比。
