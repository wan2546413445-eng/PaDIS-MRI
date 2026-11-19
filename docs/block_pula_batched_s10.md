# Block pULA：单进程轨迹批处理与复数 S10 均值

基底：`wan2546413445-eng/PaDIS-MRI`，`main-WSY`，读取提交
`0e5b8cc4dc216895c451dbffc079b6a0fe56d698`。

## 当前代码缺失的实验信息

该提交中的 Python、Bash、Markdown 未找到 OPA1.25 / 10000kimg checkpoint 的真实路径，也未找到纯 pULA47×5 的运行脚本。
`bash/eval_pula_convention_sample0.sh` 实际使用 E0 10k、cycle_k、N48K4 和 DPS tail，并非所要求的 OPA1.25 纯 pULA 实验。
因此新 Bash 的 MODEL 默认空，必须显式填写服务器上真正的 OPA1.25 10000kimg checkpoint。没有替换为 E0、OPA2、OPA4 或 5k 权重。
T1-FLAIR 验证集路径已从现有 Bash 复用。

新 Bash 默认 cycle_k，与当前可核实的 pULA Bash 相同。由于缺少纯47×5脚本，不能声称已核实该实验的 partition 策略。若服务器的纯实验使用 random_step，可设置 `GRID_MODE=random_step`，其逐次随机语义原样保留。

## 阅读范围与修改文件

主要阅读：`eval_pula/run.py`、`recon_pula.py`、`score_opa.py`、`cg.py`，`eval/denoise_padding.py`、`inverse_operators.py`、`utils.py`，并核查 `eval/evaluator.py` 的数据加载/保存链、`eval/run.py` 的入口和 `data/brain_val_data.py` 的白化数据生成约定。检索了仓库现有评估 Bash，包括 `bash/eval_pula_convention_sample0.sh`、E0 与 OPA 评估脚本中的模型、验证集与输出路径。

| 文件 | 修改 |
| --- | --- |
| eval_pula/run.py | 新增显式启用的轨迹参数，复用原指数 schedule，允许固定样本缓存；未启用时仍走旧 runner |
| eval_pula/score_batch.py | 轨迹 patch 打包、共享网络 micro-batch、按 offset 独立拼接；保留原 residual 写回舍入顺序 |
| eval_pula/trajectories.py | 私有 RNG、原 PULASystem 的批处理适配器、串行 CG、保存/复数均值/现有指标、计时/显存/manifest |
| eval/utils.py | post_eval_normalize 增加可选 sample_cache；复用原指标，缓存 k-space 在 fftmod 前 clone |
| tools/test_block_pula_trajectory_batch.py | CPU 独立性测试，以及实际旧 runner 与新版 S1 的 GPU 等价性检查 |
| bash/benchmark_block_pula_trajectory_batch.sh | 顺序测试 tb1/2/4/8/10，独立目录，OOM 记录后停止更大配置 |
| bash/eval_block_pula_author_N60K4_S10_T1FLAIR.sh | sample 0 / R4 默认正式命令与所需环境变量，快测 N2K2 模式 |
| docs/block_pula_batched_s10.md | 本说明 |

`eval_pula/recon_pula.py`、`cg.py`、`score_opa.py`、`eval/denoise_padding.py` 没有修改。

## 每次逻辑更新的真实调用链

旧链：run → PatchScore → 原 denoisedFromPatches → reconstruct 的统一 prior_score → PULASystem.update_rhs → 原 CG → state + delta。

新链：run → run_trajectories → sample_batch → **同一个 reconstruct** → TrajectoryPatchScore → 原 PULASystem.update_rhs/solve（逐轨迹）→ 同一 state + delta。

默认 FOV 为384×384，每侧补64变为512×512。每个 partition 是7×7个互不重叠64×64 patch，共49个。位置编码仍从 padded 全图的[-1,1]坐标图中裁出。
cycle_k 为每个轨迹按私有 partition seed 产生 K 个 shift，第一个(0,0)，其余不重复，随后各噪声层复用该 K 元序列；random_step 则调用原 random_shifts，逐更新抽样。fixed 保持全零 shift。

每个 inner update 都取当前轨迹状态，按该轨迹本次 shift 抽取 patch，合并所有轨迹 patch 为网络输入。记录 `(trajectory, start, stop, indices)` 后按 offset 拆分网络输出，分别写回各自 residual，并裁回384×384。没有跨轨迹平均 score、state、noise 或 CG。
原 helper 的 `float32 residual += float64 prediction; residual -= query; query + residual` 顺序被保留。原 helper 在废弃 padding 上生成的 `randn_like * 0` 没有数值作用，批处理路径省去此全局随机数消耗；pULA 私有 Generator 的抽样顺序保持一致。

## 轨迹、RNG 和网络/CG batch

状态形状为 `[B,1,H,W]` 复数。S=10、tb=4 时依次处理4+4+2，每个 batch 都调用现有 reconstruct。
checkpoint/EMA 网络只加载一次。每个 sample 的文件、raw/whitened SENSE 算子及固定数据各准备一次，复用于所有轨迹和聚合指标。

每条轨迹拥有 `torch.Generator(device).manual_seed(SEED_START+s)`。
partition 由原函数内部的 `random.Random(partition_seed)` 独立产生，无全局 Python/NumPy 随机流。partition seed 为 `grid_seed+s`，正式 Bash 的 grid_seed 与 SEED_START 相同。每条轨迹完整 shift 序列、SHA256、base patch indices、策略、seed 和每次 CG 记录均写入 manifest，可供后续 DPS 配对读取。

网络一次接收当前轨迹 batch 的全部 patch，默认输入为 `[49*B,2,64,64]`。设置 `PADIS_PATCH_BATCH` 可沿此拼接维 micro-batch，允许切口跨越轨迹边界，拆分始终依赖保存的 offset。推理使用 no_grad，模型 eval。
CG **未向量化**，逐条调用原 PULASystem 与原 solve_cg，互不共享 RHS、初值或 warm start。

初始化、精确似然、gamma、Langevin 方差、padis_real/paper_complex、预条件器、CG 右端项与停止条件、最终 raw state 均未改变。无 DPS tail、terminal denoise、多网格 score 平均或 LGFC。

## 计数和计时解释

正式设置：N60、K4，sigma_max10 初始化，随后60个指数下降噪声层，最后 sigma_min0.01。每轨迹240个逻辑更新，10轨迹共2400。CG 总调用2410，包含10次初始化。
49 patches/partition 时实际 patch 数为117600。网络调用数由轨迹 batch 与 PADIS_PATCH_BATCH 决定，独立记录为 network_batch_calls；不能称为2400次网络调用。

模型加载后用零状态 warm-up，fork_rng 保护全局 RNG，并在创建任何轨迹 Generator 前完成。采样前 empty_cache、reset_peak_memory_stats、synchronize，采样后 synchronize。
sampling_wall_seconds 是所有轨迹 batch 的总采样墙钟，包括 patch 准备/网络/拼接/CG/CPU 输出拷贝与同步开销，排除 checkpoint、warm-up、指标与磁盘保存。已完成轨迹按 batch 及时保存，trajectory_save_seconds 单列。
峰值 allocated/reserved 取全部采样 batch 的最大值，排除 warm-up 与指标阶段。
checkpoint_load_seconds、warmup_seconds、aggregation_and_evaluation_seconds 与 end_to_end_seconds 分别记录。end-to-end 从准备 CUDA 后、加载模型前开始，包含数据准备、所有保存和后处理，不含解释器启动/模块导入。多 sample 情况下 sample timing 的 end_to_end 为从本次进程计时起点累计的完成时间；根 timing 是整个运行总时间。
effective_sampling_seconds_per_trajectory 仅为总采样时间除轨迹数，是吞吐量指标，不能拿它冒充 S10 结果延迟。

quality_time_curve.json 包含 S1/2/3/5/8/10 的真实指标、prefix_batch_completion_seconds 和 actual_result_ready_end_to_end_seconds。
prefix completion 表示包含第 S 条轨迹的 batch 完成时刻，例如 tb4 下 S1/S2/S3 都对应第一批4条的完成时刻。聚合在全部采样结束后执行，所以 actual_result_ready_end_to_end_seconds 诚实包含这一等待。未伪造独立运行 S1/S2 的耗时。

## 应用与运行

在项目根目录解压 ZIP，按仓库相对路径覆盖。本包仅含上表8个文件，不复制整个 eval_pula、不含模型/数据/结果/缓存。
以下路径中的 MODEL 必须替换为正确的服务器 OPA1.25 10000kimg checkpoint。

```bash
export MODEL='/填写实际的OPA1.25_10000kimg_checkpoint.pkl'
export VAL='/mnt/SSD2/wsy/brain_multicoil/fastmri_3T_eval/val_t1-flair_subsamp/32dB'
export RESULTS='/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data'
```

单轨迹旧 runner vs 新 S1（GPU，固定 R4/sample0/seed123/N60K4）：

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python tools/test_block_pula_trajectory_batch.py \
  --model_path "$MODEL" --val_dir "$VAL" \
  --out_root "$RESULTS/Block_pULA_OPA1p25_10k_N60K4_S1_equivalence_$(date +%Y%m%d_%H%M%S)"
```

工具用 git show 读取上述原始提交的 run.py，在原项目导入环境中执行，比较实际旧入口与新版入口，而非只比较一个转换 helper。无基底 git 对象时，可提前保存该提交的原 run.py，使用 `--baseline_run /绝对路径/old_run.py`。
检查 sigma/partition 完全一致、初始化 CG 记录和后续迭代序列一致、CG solve 数、最终复数张量（atol1e-6、rtol1e-5）和三项指标（atol1e-5、rtol1e-6）；失败断言分别指向 schedule、partition RNG、CG、patch/batch 或指标层。保存 baseline 源码、两份运行日志及 equivalence.json。

batch1/2/4/8/10 的显存快测（默认 N2K2，只筛查容量，不作为正式质量结果）：

```bash
GPU=0 MASK=4 SAMPLES=0 MODE=quick \
  bash bash/benchmark_block_pula_trajectory_batch.sh
```

全部 batch 配置的正式预算测试（每配置独立 S10，不会启动10个轨迹子进程）：

```bash
GPU=0 MASK=4 SAMPLES=0 MODE=formal \
  bash bash/benchmark_block_pula_trajectory_batch.sh
```

选定安全 batch 后正式 R4/sample0：

```bash
GPU=0 MASK=4 SAMPLES=0 TRAJECTORY_BATCH_SIZE=4 \
  bash bash/eval_block_pula_author_N60K4_S10_T1FLAIR.sh
```

支持 GPU、MASK、SAMPLES（逗号或范围）、TRAJECTORY_BATCH_SIZE、NUM_TRAJECTORIES、SEED_START、MODEL、OUT_ROOT。默认 S10，seed123..132，只运行 sample0/R4。允许 MASK=7，但本次未运行 R7 或全32张。
GRID_MODE、VAL 可按服务器真实旧实验显式覆盖。所有比较需使用同一策略、partition seed 和 micro-batch 设置。
CPU fixture：

```bash
python tools/test_block_pula_trajectory_batch.py --cpu_fixture
```

静态检查：

```bash
python -m py_compile eval_pula/run.py eval_pula/score_batch.py eval_pula/trajectories.py eval/utils.py tools/test_block_pula_trajectory_batch.py
bash -n bash/benchmark_block_pula_trajectory_batch.sh
bash -n bash/eval_block_pula_author_N60K4_S10_T1FLAIR.sh
```

## 输出绝对目录

默认正式根目录：
`/mnt/SSD2/wsy/PaDIS-MRI/PaDIS-MRI-recon/new_data/Block_pULA_OPA1p25_10k_N60K4_S10_sigma10_0p01_T1FLAIR_R4/tb4_<时间戳>/`

| 相对该根目录 | 内容 |
| --- | --- |
| run.log | 全部 Python 标准输出和错误；启动时打印绝对路径 |
| run_manifest.json | 配置、seeds、sigma、预算、是否完成 |
| timing.json | 总加载、warm-up、采样、保存、后处理、端到端、显存与独立计数 |
| sample_0/manifest.json | 数据噪声标定、每轨迹 partition 全序列及 CG 记录 |
| sample_0/timing.json | 单样本总时间、显存、阶段计时和计数 |
| sample_0/quality_time_curve.json | 各 S 的指标与真实完成时刻 |
| sample_0/trajectory_000_seed123/evaluate/recons/recon_patch_0.npy | 第一条 raw 复数重建，shape[1,H,W] |
| sample_0/trajectory_009_seed132/evaluate/recons/recon_patch_0.npy | 第十条 raw 复数重建 |
| sample_0/S01_mean/evaluate/ 至 S10_mean/evaluate/ | S=1,2,3,5,8,10 的复数均值 recons/、results.json、comp_plots/ |
| failure.json | CUDA OOM 状态（退出码42），已保存的批次结果保留 |

benchmark 默认位于同一 new_data 下的 `Block_pULA_OPA1p25_10k_S10_batch_benchmark_<模式>_<时间戳>/`，子目录 tb1/tb2/tb4/tb8/tb10 各有 run.log 与相同结果结构，根目录有 benchmark_status.tsv。OOM 后停止更大 batch。

聚合先对前 S 条**最终复数图像**逐像素均值，再调用原 post_eval_normalize；不平均指标，不先取幅度。复用原 ACS-99 归一化、幅度指标与图像输出。
当前提交的 post_eval_normalize 没有 ESPIRiT foreground 接口，因此未虚构/替换 mask 或加入另一套前景指标。

## 实际验证与待验证

已完成 Python 编译、两 Bash 语法检查、旧参数默认值逐项比较、新参数解析、N2/N47/N60 sigma 精确一致检查，以及原采样/CG/旧 oracle/helper 无 diff 检查。
CPU 合成测试覆盖 fixed/cycle_k/random_step × 两种复数约定 × tb1/2/4/8/10，共30组；10个私有 seed 的最终复数图像符合容差，初始化和全部更新 CG 记录与原链一致。另验证跨轨迹 micro-batch 切口。
另在 CPU 合成问题执行 N60K4/S10/tb4，核实2400更新、2410次 CG；验证缓存/非缓存原指标完全一致、固定数据未被 fftmod 修改；完整 CPU 合成 runner 验证一次数据加载、复数前缀均值、真实指标函数和模拟 OOM 保留已完成 batch。

**未实际执行服务器 checkpoint、MRI sample0、任何 GPU 重建、GPU 旧新 S1 等价性、batch 显存测速或正式质量曲线。** CPU 检查不替代这些项目，也不能据此声称 pULA 提速或质量提升。需先补齐 MODEL，再在服务器依次完成等价性、快测、选定 batch 的正式实验。
