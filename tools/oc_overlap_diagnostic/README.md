# E0 / AGGA 多尺度空间重叠诊断

只模拟训练样本访问、Patch 尺度选择和坐标抽样。没有训练、去噪、重建、模型构建或 checkpoint 加载，也不调用 OC pair sampler。

## 当前交付状态

源码依据：`wan2546413445-eng/PaDIS-MRI`，`main-WSY`，提交 `4f58b8569a75e7e9c114b9944533e2dddebcc2cc`。

当前编写环境没有服务器上的 `noisy.pt`。真实 MRI 的 500、1000、5000、10000 kimg 实验**尚未执行**。自动化测试全部采用临时合成张量，不能将测试数据当成论文实验数据。实际运行前，应确认服务器 checkout 与本次核验版本一致。代码会记录实际 commit、分支和相关源码 SHA256。

## 环境与安装

建议在服务器现有 `padis_mri` 环境中运行。需要 Python 3.9+、NumPy、PyTorch、Matplotlib。测试使用标准库 `unittest`，不需要 pytest、CUDA、wandb、diffusers 或网络权重。CPU 即可运行。

把整个 `oc_overlap_diagnostic` 目录放入原仓库的 `tools/`，不会覆盖现有训练实现。所有命令均在原仓库根目录执行。确认以下输出正确：

```bash
git remote -v
git branch --show-current
git status --short
```

先测试，再执行真实实验，测试有失败或 skip 时不应将实验标记为正式结果：

```bash
python tools/oc_overlap_diagnostic/test_overlap.py
```

## 一次运行两组

```bash
python tools/oc_overlap_diagnostic/test_overlap.py && \
python tools/oc_overlap_diagnostic/analyze_overlap.py \
  --data /mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB/noisy.pt \
  --sample_index 0 \
  --modes uniform_e0 agga \
  --patch_sizes 16 32 64 \
  --patch_probs 0.2 0.3 0.5 \
  --batch_size 4 --batch_gpu 1 \
  --pad_width 64 \
  --kimg_checkpoints 500 1000 5000 10000 \
  --overlap_threshold 0.5 --seed 123 \
  --output_dir results/oc_overlap_diagnostic
```

只运行 E0：将 `--modes uniform_e0 agga` 改为 `--modes uniform_e0`。

只运行 AGGA：改为 `--modes agga`。

同一配置与种子下，单独运行与合并运行会产生相同的各模式事件序列。输出目录的时间戳不同，JSON 记录的运行路径、耗时和 Git 状态也可能不同。

修改检查点：例如 `--kimg_checkpoints 500 1000 5000`。修改种子：例如 `--seed 124`。支持任意正数检查点，但正式论文表格应保留指定四个。默认支持的三个尺度可以调整为源码 `batch_mul_dict` 中的尺度；R64 在没有 64 尺度或 64 分母为零时为空。

## 核验后的训练机制

`train_agg_simplified.py` 实际导入 `training.training_loop`，不是 `training_loop_agg.py`。

单 GPU、batch_size=4、batch_gpu=1 时，每次优化迭代有四个累积轮次。每个轮次各自选择一次尺度，之后连续获取 `batch_mul` 个 DataLoader batch，并拼接送入 loss。每个源图事件生成一个 Patch。倍率为 16→64、32→32、64→16。

显式尺度概率为 [0.2,0.3,0.5] 时，平均倍率是 30.4。源码每次完整优化迭代更新 `cur_nimg += int(4*30.4)`，即增加 121。实际 Patch 事件数另行累计，不用 121 替代。

从 0 kimg 开始连续运行到最大检查点。首次在完整优化迭代结束时跨越目标进度才记录，不能截断迭代。两个模式共享一次调度生成器、完整样本索引流和每个 sample0 事件的共同均匀随机数。空间抽样不会推进尺度或 Dataset 随机流。

Dataset 顺序按 `misc.InfiniteSampler` 的 shuffle、window_size=0.5 和逐事件换位机制复现，测试直接从原始源码 AST 提取 `__iter__` 对照。运行时不创建原训练 DataLoader，也不执行不必要的数据读取或全部 MRI 的内容分析。

这是独立随机流下的受控模拟，并不复现历史网络训练完整 RNG 轨迹。配置来源是用户提供的 E0 10000→13300 kimg 恢复训练记录，模拟不采用恢复进度或恢复状态。

## 输入与概率缓存

当前 `ImageFolderDatasetX` 要求 `x_est_gt` 是复数 [N,H,W]。脚本采用同样的 `view_as_real → NumPy float32 → CHW → constant zero padding`。不自行增加归一化，不硬编码样本数与图像大小。sample0 的输入字节 SHA256、真实形状、dtype 和 Padding 后尺寸写入配置。

原始 `Patch_EDMLoss.__call__` 先执行 `pachify`，再进行增强和加噪。因此增强和 EDM 噪声不会改变本轮 AGGA 位置分布。Dataset 当前实例没有随机 xflip 或随机内容变换。固定 sample0 可以缓存。

AGGA 使用原文件中的 `_structure_maps`、`_region_score`、`_sampling_probabilities`，分别计算每个尺度的概率。5% 支持阈值、正梯度候选的 95% 分位归一化、尺度相关平均池化和 rho 混合均未改写。

另外临时拦截原始 `sample_single_locations` 送入 `torch.multinomial` 的概率张量，与缓存逐元素比较，随后立即恢复函数。不会修改任何原始源文件。缓存转为 float64 后再次归一化并建立 CDF，记录这一归一化带来的微小差异。行列映射为 `top=flat//candidate_width, left=flat%candidate_width`。

每个尺度附带 32 个聚合位置区间的经验分布检验，避免对大量稀疏候选做不可靠的逐位置频率比较。自动化测试还在小画布上直接调用原始抽样器进行经验检验，验证零图像/非有限峰值时的均匀退化行为。

E0 设置 rho=0、gamma=0。所有合法左上角位置等概率，包括 Padding 内的位置。与原始 `Patch_EDMLoss.pachify` 的等价范围是**同一画布上的空间坐标概率分布**，不要求 Torch、NumPy 随机序列或历史训练逐事件一致，也不表示其他训练配置均相同。

## 重叠定义与三个指标

同一 MRI、相同尺度，当前 Patch 与任一历史 Patch 的交集面积达到单 Patch 面积的至少 50%，当前事件计数一次。50% 边界和完全重复坐标计入，跨尺度不比较，多个历史交集不能相加。同一累积轮次内按样本顺序定义历史，不表示网络前向内部有先后因果。

- R64 = 有效重叠的 64 Patch 数 / 全部 64 Patch 数 ×100%。
- Rall = 三个尺度有效重叠数之和 / sample0 全部 Patch 数 ×100%。
- R64_total = 有效重叠的 64 Patch 数 / sample0 全部 Patch 数 ×100%。

所有百分比均采用 0–100 单位，报告显示四位小数。零分母在 JSON 中为 null，CSV 中为空。R64_total=R64×N64/Nall，不应再额外除以100。

状态索引为每个合法左上角保存一个满足条件的历史见证事件。新增 Patch 将其单独满足阈值的合法位置集合写入索引。当前 Patch 先查询再加入，查询为常数时间，更新只涉及一个 Patch 附近的行区间。没有 N×N 矩阵或逐历史扫描。整数面积阈值避免 50% 边界误判。

## 输出说明

每次新建独立 UTC 时间戳目录，不覆盖已有结果。

```text
run_.../
  config.json
  probability_checks.json
  execution.json
  REPORT.md
  uniform_e0/
    checkpoints.csv
    curve.csv
    statistics.json
    sample0_events.csv
    probabilities.npz
    example_pair.json
    cumulative_overlap.png / .svg
    actual_overlap_pair.png / .svg
    coverage64.npy / .png / .svg
  agga/                         # 同上
  comparison/
    comparison.csv / .json
    R64_comparison.png / .svg
```

单模式只生成所选模式文件夹。若没有出现合格的64 pair，则 example_pair 为 null，不生成 pair 图，不构造示例。若事件集为空，对应事件 CSV 不创建。`Total sample patches` 明确指 sample0 的全部尺度数量，训练所有样本的事件总数另列为 `total_patch_events`。

正式四个检查点以外，记录 1、5、10、20、50、100、200 kimg 的早期快照，之后默认每100 kimg记录一次。曲线折线连接真实快照，不生成插值统计值。

示例选择实际事件流中首个达到阈值的64 pair，坐标、见证事件、交集面积和事件编号均可追溯。不根据图像是否好看筛选，也不保证首个 pair 位于解剖前景。覆盖热图记录每像素被64 Patch覆盖的次数，所有模式共用颜色范围，不能用其替代 R64。

## 解释范围

结果仅描述 sample0 在给定采样机制下的累计空间重复覆盖。长期累计值可能饱和，因此保留早期曲线。AGGA−E0 的差异单位是百分点，不是相对提升百分比。AGGA 低于 E0 时照实报告。

单样本、单种子的结果不代表整个训练集的统计规律。本轮不增加多图或多种子实验。该实验不测量随机两 Patch 的成对重叠概率、同一前向中的即时重叠、去噪预测不一致、OC loss 或 OC 有效性。

数据缺失、依赖不可用、概率核验失败时停止，不能生成正式实验数值。输入数据检查发生在创建输出目录之前。
