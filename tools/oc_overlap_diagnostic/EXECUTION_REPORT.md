# 本次执行报告

## 状态

代码与自动化测试已完成。真实 MRI 的 E0 / AGGA 四检查点实验未执行，原因为当前执行环境没有指定服务器数据文件。未生成论文实验数值、MRI 示例图或伪造结果。

## 已完成

- 拉取真实仓库 `wan2546413445-eng/PaDIS-MRI` 的 `main-WSY` 分支。
- 核验提交 `4f58b8569a75e7e9c114b9944533e2dddebcc2cc` 的训练入口、训练循环、Dataset、原始 loss、简化 AGGA loss、AGGA sampler、InfiniteSampler 和已有可视化脚本。
- 确认简化 AGGA 入口使用 `training_loop.py`，每累积轮次选择尺度，默认配置每迭代计数增加121。
- 确认 Dataset 为复数到实虚通道、float32和零填充，Patch 裁取发生在增强之前。
- 新增一个连续模拟生成器，两组共享尺度调度、Dataset访问顺序及配对空间随机数。
- 直接使用原始 AGGA 概率函数缓存三个尺度的概率，并核验原始单 Patch 采样调用收到的实际概率。
- 实现精确的同尺度历史重叠索引、见证事件记录、三项指标、检查点及早期快照。
- 实现 CSV、JSON、PNG、可编辑 SVG、采样覆盖图和真实采样 pair 的导出。
- 完成32项测试，全部通过，0失败、0跳过。最后一次完整测试耗时5.226秒。
- 合成坐标性能测试：512×512画布、64×64 Patch、50000次查询与加入，耗时2.066秒。该时间只衡量空间索引，不含真实数据加载、AGGA概率构建、完整训练调度或图像输出。

## 测试证据

`TEST_LOG.txt` 是实际测试输出。日志中的计数和表格来自明确标记的合成数据测试，**不是 MRI 实验结果**。临时合成数据和测试图已随临时目录清理。

测试覆盖：50%边界与低于50%、重复坐标、跨尺度隔离、多个见证只计一次、不合并历史交集、对角位移、边界、随机序列与朴素几何对照、逐位置几何核验、零分母、Dataset顺序与原始InfiniteSampler对照、实际事件计数、同轮多次访问、共享调度、单模式与合并运行一致、连续前缀、检查点跨越、计数单调性、指标恒等式、重复运行复现、CDF行列映射、经验概率、原始AGGA概率一致、E0均匀、异常输入均匀退化、原始采样器经验分布、预处理一致和完整CLI导出。

## 正式运行阻塞

实际尝试了默认命令，得到：

```text
FileNotFoundError: Training data unavailable: /mnt/SSD/wsy/data/fastmri_train_batch0_pilot/brain_train_d384_s200/32dB/noisy.pt. No experiment results generated.
```

没有创建正式结果目录。因此，本次无法报告真实数据样本数、实际512×512尺寸核验、四检查点累计次数、三项重叠率或AGGA−E0差异。

## 交付文件

- `analyze_overlap.py`：正式数据入口、概率核验、连续模拟、统计与绘图。
- `test_overlap.py`：32项独立测试，使用标准库unittest。
- `README.md`：完整运行命令、环境、数学含义和限制。
- `EXECUTION_REPORT.md`：本报告。
- `TEST_LOG.txt`：实际自动化测试记录。

## Git边界

当前本地副本的分支仍为 `main-WSY`，HEAD未变。`git diff --exit-code`通过，没有修改任何受版本控制的原始文件。新增项仅为：

```text
?? tools/oc_overlap_diagnostic/
```

没有创建分支、提交、推送或修改服务器目录。交付包仅包含新增目录，可放入服务器原仓库的 `tools/` 下。完整运行命令见README。
