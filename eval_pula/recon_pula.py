# ============================================================================
# 文件作用：把论文的 pULA（预条件无校正 Langevin 算法）接到 PaDIS-MRI 的 patch EDM 上，
#          并可选择在 pULA 后继续执行原始 PaDIS-DPS 的最后若干噪声层。
#
# 约定：
#   x/state : 图像域复数 Tensor，通常为 [1, 1, H, W]。
#   y       : 多线圈欠采样 k-space 复数 Tensor，通常为 [1, Nc, H, W]。
#   A       : 含 sensitivity maps、FFT、mask 的 MRI 前向算子。
#   A^H     : A 的共轭转置（伴随算子），不是 A 的逆。
#   sigma_net : PaDIS 每个实通道的噪声标准差。
#   sigma_system : padis_real 取 sigma_net，paper_complex 取 sqrt(2) * sigma_net。
#   B_sigma = A^H A + sigma^(-2) I，M_sigma = B_sigma^(-1)。
#
# 核心理论：pULA 按照 x <- x + M_sigma * (梯度项 + Langevin 随机项) 更新；
#           M_sigma 不显式构造，而是每一步通过 CG 近似求解线性方程。
# ============================================================================
"""pULA with real-channel or unit complex variance conventions."""

# defaultdict(list) 累计不同阶段的耗时；contextmanager 用于对代码段计时。
from collections import defaultdict
from contextlib import contextmanager
from math import sqrt
from time import perf_counter

import torch
import torch.nn.functional as F

# CG 是求解 B_sigma * delta = rhs 的数值方法；real_channels 把复数图转成两实数通道。
from eval_pula.cg import solve_cg
from eval_pula.score_opa import compare_grids, real_channels
from eval_pula.pa_balance import VARIANTS, PABalanceStats, balanced_coefficients, edm_sigma_data


# [pULA 随机项] 构造每个实数通道方差为 1 的独立高斯噪声。
# PyTorch complex64 的 torch.randn 的实、虚部方差各为 1/2；乘 sqrt(2) 后各为 1。
# 这与下方 whiten_measurement 设定的“每个实数通道测量噪声方差为 1”相对应。
def real_unit_noise(like, generator):
    return sqrt(2.0) * native_complex_noise(like, generator)


# paper_complex 和原始 PaDIS-DPS 都使用 PyTorch 的单位复方差噪声。
def native_complex_noise(like, generator):
    """Match torch.randn_like(complex): variance 1/2 in each real channel."""
    return torch.randn(like.shape, dtype=like.dtype, device=like.device, generator=generator)


# ========================= 0. 数据噪声白化 =========================
# 观测模型 y = A x + n。
# 白化后记 A_w = W A，y_w = W y。白化目标由 convention 决定。
# 注意：改变的是 MRI 测量算子与数据的单位，图像变量 x 不额外缩放。
def whiten_measurement(operator, measurement, complex_covariance, convention="padis_real"):
    """Whiten both SENSE maps and y in the selected variance convention."""
    paper_complex = convention == "paper_complex"
    # 仅有一个“复数方差 E|n|^2”标量时：每实通道方差 = complex_var / 2。
    # 因此 W = sqrt(2 / complex_var)，同时缩放所有 coil maps 与观测 k-space。
    if isinstance(complex_covariance, float):
        weight = 1.0 / sqrt(complex_covariance) if paper_complex else sqrt(2.0 / complex_covariance)
        maps, y = weight * operator.maps, weight * measurement
    # 若噪声在 coil 之间相关，则使用完整的复 Hermitian 协方差矩阵。
    # Cholesky factor 分解 covariance / 2，对应每个实通道单位方差的约定。
    else:
        covariance = complex_covariance.to(device=measurement.device, dtype=torch.complex128)
        coils = measurement.shape[1]
        if covariance.shape != (coils, coils) or not torch.allclose(covariance, covariance.mH):
            raise ValueError("Expected a Hermitian complex coil covariance with shape [coils, coils].")
        factor = torch.linalg.cholesky(covariance if paper_complex else covariance / 2).to(measurement.dtype)

        # 对 coil 维应用 W；先 reshape 成 [coils, pixels] 再解三角系统。
        # 只作用于测量空间/coil 维，不改变图像域坐标。
        def apply(value):
            flat = value.reshape(coils, -1)
            return torch.linalg.solve_triangular(factor, flat, upper=False).reshape_as(value)

        maps, y = apply(operator.maps), apply(measurement)
    # 因采样 mask 为所有 coil 共有的二值掩膜，coil 白化可以与掩膜交换。
    # 返回“新 MRI 算子 + 新观测”，后面的 pULA 始终使用白化后的算子。
    # A common binary mask commutes with the constant coil whitening matrix.
    return type(operator)(mask=operator.mask, maps=maps), y


# ========================= 1. pULA 数学算子 =========================
# PULASystem 封装：A_w、A_w^H、CG、随机扰动和 pULA 更新右端项。
# 注意：后面 DPS tail 会显式使用“未再白化”的 raw_operator/raw_measurement。
class PULASystem:
    """Matrix-free pULA system in the selected variance convention."""

    # 这里的 operator、measurement 是白化后的 A_w 和 y_w。
    # generator 为该样本的独立 torch.Generator，避免依赖全局随机数状态。
    def __init__(self, operator, measurement, generator, cg_iters, cg_rtol, convention="padis_real"):
        self.operator = operator
        self.y = measurement
        self.generator = generator
        self.cg_iters = cg_iters
        self.cg_rtol = cg_rtol
        self.convention = convention
        self.draw_noise = native_complex_noise if convention == "paper_complex" else real_unit_noise
        self.forward_calls = 0
        self.adjoint_calls = 0
        self.reset_update()

    def reset_update(self):
        # Initialization always uses the original H and original noise.
        self.pula_variant = "baseline"
        self.normal_scale = 1.0
        self.normal_diagonal = None

    def configure_update(self, variant, sigma_net, gamma, sigma_data=None, variance=None):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown pula_variant: {variant}")
        self.reset_update()
        self.pula_variant = variant
        if variant in ("balanced_k0", "balanced_full"):
            template = self.adjoint_template
            self.normal_scale, self.normal_diagonal = balanced_coefficients(
                template, sigma_net, sigma_data, gamma, self.convention,
                variance if variant == "balanced_full" else None,
            )

    def normal(self, v, sigma_net):
        if self.normal_diagonal is None:
            # Keep the legacy arithmetic (including division) unchanged.
            return self.adjoint(self.forward(v)) + v / self.system_sigma(sigma_net) ** 2
        return self.normal_scale * self.adjoint(self.forward(v)) + self.normal_diagonal * v

    def system_sigma(self, sigma_net):
        return sqrt(2.0) * sigma_net if self.convention == "paper_complex" else sigma_net

    # A_w x。real_channels 将 [B,1,H,W] complex 转为 [B,2,H,W] real。
    # forward_calls 仅用于诊断 MRI 算子被调用多少次。
    def forward(self, x):
        self.forward_calls += 1
        return self.operator.forward(real_channels(x))

    # A_w^H y。此函数通过实际 MRI_utils.adjoint 实现共轭转置。
    def adjoint(self, y):
        self.adjoint_calls += 1
        return self.operator.adjoint(y)

    # [CG 的线性系统]
    #         (A_w^H A_w + sigma^(-2) I) * z = rhs

    def solve(self, rhs, initial, sigma_net):
        # 无需构造大型矩阵 B_sigma：以函数方式给出 B_sigma 对向量 v 的乘法。
        def normal(v):
            return self.normal(v, sigma_net)

        return solve_cg(normal, rhs, initial, self.cg_iters, self.cg_rtol)

    # [论文随机项：式(11)/(12)附近]
    # 依次抽取 n1（观测/k-space 空间）和 n2（图像空间），两种约定复用相同的底层随机数。
    # 构造 eta = A_w^H n1 + sigma^(-1) n2。
    # 因 Cov(eta) = A_w^H A_w + sigma^(-2) I = B_sigma，
    # 故 Cov(M_sigma eta) = M_sigma（假设精确 CG、白化准确）。
    def noise_rhs(self, template, sigma_net):
        sigma_system = self.system_sigma(sigma_net)
        measurement_noise = self.draw_noise(self.y, self.generator)
        image_noise = self.draw_noise(template, self.generator)
        rhs = self.adjoint(measurement_noise) + image_noise / sigma_system
        return rhs, image_noise

    # ========================= 2. pULA 高斯初始化 =========================
    # 论文的高噪声高斯近似：
    #   x_init 的均值为 M_sigma * A_w^H y_w，协方差按当前单位表示为 M_sigma。
    # 通过“perturb-and-solve”（加扰动再解线性方程）采样：
    #   B_sigma x_init = A_w^H y_w + A_w^H n1 + sigma^(-1) n2。
    def initialization_rhs(self, sigma_net):
        # 确定性的均值项 A_w^H y_w；它本身不是最终初始化图像。
        mean_rhs = self.adjoint(self.y)
        # Shape/device/dtype only; no additional MRI operator application.
        self.adjoint_template = mean_rhs
        # 得到 eta = A_w^H n1 + sigma^(-1)n2；同时保留 n2 供 CG warm start。
        noise, image_noise = self.noise_rhs(mean_rhs, sigma_net)
        # 返回值 1：CG 右端 b = A_w^H y_w + eta。
        # 返回值 2：CG 初值 z0 = sigma*n2，只为加速线性系统求解。
        # 注意它不是额外增加一次噪声；若 CG 完全收敛，最终结果不依赖 z0。
        return mean_rhs + noise, self.system_sigma(sigma_net) * image_noise

    # ========================= 3. 每次 pULA Langevin 更新 =========================
    # 目标：x_new = x + delta，其中
    #   delta = gamma*M_sigma*(A_w^H(y_w-A_w x) + score)
    #           + sqrt(2*gamma)*M_sigma*(A_w^H n1 + sigma^(-1)n2)。
    # score 是 PaDIS patch EDM 给出的近似全图 score，并非训练好的全图 score net。
    def update_rhs(self, x, prior_score, sigma_net, gamma):
        sigma_system = self.system_sigma(sigma_net)
        # 数据一致性梯度（白化后的原始状态 x）：A_w^H(y_w - A_w x)。
        # 和后面 DPS 的 A^H(y - A*denoised) / 对 denoiser 反传不同。
        likelihood = self.adjoint(self.y - self.forward(x))
        # 每次 Langevin 更新重新生成独立随机扰动（不与初始化噪声复用）。
        noise, image_noise = self.noise_rhs(x, sigma_net)
        diffusion = sqrt(2.0 * gamma)
        if self.pula_variant == "noise08":
            noise = 0.8 * noise
            image_noise = 0.8 * image_noise
        # 右端项：B_sigma * delta = gamma*(likelihood + prior_score) + sqrt(2gamma)*eta。
        rhs = gamma * (likelihood + prior_score) + diffusion * noise
        # warm start 用于解 delta 的 CG；它不是公式中单独加到状态上的项。
        # [待核验] 当 CG 迭代有限时，warm start 会影响数值误差与采样统计。
        # BART's state warm start expressed as an increment.
        if self.normal_diagonal is not None:
            initial = (gamma * prior_score + diffusion * image_noise / sigma_system) / self.normal_diagonal
        else:
            initial = gamma * sigma_system ** 2 * prior_score + diffusion * sigma_system * image_noise
        return rhs, initial


# ========================= 4. 主采样过程 =========================
# @torch.no_grad(): pULA 不对 denoiser 求 Jacobian，因此不存计算图；
# 在 DPS tail 中会用 torch.enable_grad() 局部恢复梯度用于反向传播 DC。
@torch.no_grad()
def reconstruct(system, oracle, sigmas, shifts, gamma, *, diagnose_levels=(), diagnostic_shifts=(),
                terminal=False, report=print, pula_levels=None, pula_shifts_by_level=None,
                dps_operator=None,
                dps_measurement=None, dps_sigmas=(), dps_shifts=(), dps_inner_loops=10,
                dps_zeta=3.0, save_handoff=None, pula_variant="baseline",
                pa_block_size=8, pa_epsilon=1e-12):
    # sigmas[0] 是高斯初始化 sigma_max；sigmas[1:] 才是各个 Langevin 更新层。
    # 因而日志显示的实际 pULA 层数可能与建立 sigma 网格时的 num_noise_levels 不同。
    """sigmas[0] initializes the chain, sigmas[1:] drive the Langevin levels."""
    # events：分别记录初始化 CG、Langevin CG、网络 forward、DPS backward 等耗时。
    events = defaultdict(list)
    device = system.y.device
    if pula_variant not in VARIANTS:
        raise ValueError(f"Unknown pula_variant: {pula_variant}")
    balanced = pula_variant in ("balanced_k0", "balanced_full")
    sigma_data = edm_sigma_data(oracle.net) if balanced else getattr(oracle.net, "sigma_data", None)
    pa_records = []

    # 计时上下文：GPU 使用 CUDA Event，更接近实际 kernel 执行时间。
    @contextmanager
    def section(name):
        if device.type == "cuda":
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            yield
            end.record()
            events[name].append((start, end))
        else:
            start = perf_counter()
            yield
            events[name].append(perf_counter() - start)

    # 采样前同步 CUDA、清空峰值显存统计；不影响采样算法。
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = perf_counter()
    network_calls = 0
    forward_timer = None

    # 注册在网络 forward 前的钩子，自动计时一次 patch EDM 的网络调用。
    def start_forward(*_):
        nonlocal forward_timer
        forward_timer = section("network_forward_seconds")
        forward_timer.__enter__()

    # 网络 forward 后关闭计时并累加计数。
    def finish_forward(*_):
        nonlocal network_calls, forward_timer
        forward_timer.__exit__(None, None, None)
        forward_timer = None
        network_calls += 1

    # [程序第一次真正生成 x]：只有初始化的 CG 求解，没有运行 patch EDM。
    system.reset_update()
    init_rhs, initial = system.initialization_rhs(sigmas[0])
    with section("initialization_cg_seconds"):
        # state 是复数图像空间的 pULA 初始随机样本。
        # init_info 含初始化 CG 迭代次数和相对残差，可检查是否收敛。
        state, init_info = system.solve(init_rhs, initial, sigmas[0])
    # 日志记录容器：CG 诊断、可选 patch 网格诊断、DPS 每步残差。
    cg_records, diagnostics, dps_records = [], [], []
    dps_tail_seconds = 0.0
    diagnostic_calls = 0
    terminal_image = None
    # 仅保留 DPS 开始之前的 pULA 噪声层。
    # 比如用户当前日志中实际是 23 层 pULA，而非必须执行完整的 48 层调度。
    selected_sigmas = sigmas[1:] if pula_levels is None else sigmas[1:pula_levels + 1]
    trackers = ([PABalanceStats(sigma_data, oracle.pad, getattr(oracle, "psize", oracle.pad),
                                pa_block_size, pa_epsilon) for _ in range(state.shape[0])]
                if pula_variant == "balanced_full" else [])
    # 用前后向钩子统计同一个 denoiser 在 pULA 和 DPS 阶段的调用次数。
    with oracle.net.register_forward_pre_hook(start_forward), oracle.net.register_forward_hook(finish_forward):
        # 对每一个 pULA 噪声标准差 sigma，从高到低运行 K 次 Langevin 更新。
        for level, sigma_net in enumerate(selected_sigmas):
            sigma_system = system.system_sigma(sigma_net)
            if trackers:
                with section("pa_statistics_seconds"):
                    variances = [tracker.begin_level(sigma_net, state[j:j+1])
                                 for j, tracker in enumerate(trackers)]
                variance = torch.cat(variances)
            else:
                variance = None
            if pula_variant != "baseline":
                system.configure_update(pula_variant, sigma_net, gamma, sigma_data, variance)
            # Use the offsets for this level when per-update random grids are configured.
            level_shifts = shifts if pula_shifts_by_level is None else pula_shifts_by_level[level]
            for step, shift in enumerate(level_shifts):
                # 当前复数状态 state -> patch EDM 去噪预测 D_patch(state,sigma)。
                # 这里只用当前 state，不像 DPS 额外先做 x + sigma*noise。
                denoised = oracle.denoise(state, sigma_net, shift)
                # 可选诊断：比较同一 state 在不同 patch 网格下的去噪预测差异。
                # 不是算法必需步骤；只在 diagnose_levels 指定的层运行。
                if step == 0 and level in diagnose_levels:
                    before = network_calls
                    predictions = [denoised if other == shift else oracle.denoise(state, sigma_net, other)
                                   for other in diagnostic_shifts]
                    diagnostic_calls += network_calls - before
                    diagnostics.append({"level": level,
                                        **compare_grids(state, sigma_net, diagnostic_shifts, predictions)})
                # Tweedie 换算：s_patch ≈ (D_patch(state,sigma)-state)/sigma^2。
                # [理论风险] 拼接的 patch score 不一定是严格的全图概率密度梯度。
                prior_score = (denoised - state) / sigma_system ** 2
                if trackers:
                    with section("pa_statistics_seconds"):
                        step_shifts = [shift] if isinstance(shift[0], int) else shift
                        for j, tracker in enumerate(trackers):
                            tracker.observe(state[j:j+1], denoised[j:j+1], step_shifts[j])
                # 计算 likelihood 梯度 + prior score + Langevin 随机项，构造 CG rhs。
                rhs, initial = system.update_rhs(state, prior_score, sigma_net, gamma)
                with section("update_cg_seconds"):
                    # CG 近似求 delta = M_sigma*rhs（不是直接输出新的 x）。
                    delta, info = system.solve(rhs, initial, sigma_net)
                # 完成当前 pULA 一步：x <- x + delta。
                state = state + delta
                cg_records.append(info)
            if trackers:
                with section("pa_statistics_seconds"):
                    rows = [tracker.end_level(gamma) for tracker in trackers]
            else:
                rows = [dict(sigma=sigma_net, gamma=gamma, v_mean=0., v_max=0.,
                             relative_noise_ratio_mean=0., relative_noise_ratio_max=0., state_slope=0.,
                             next_state_slope=0., innovation_mean=0., innovation_max=0.,
                             innovation_pairs=0) for _ in range(state.shape[0])]
            pa_records.append({"level": level, "per_trajectory": rows})
            if balanced:
                report(f"PA {pula_variant} sigma={sigma_net:.5g} sigma_data={sigma_data} "
                       f"v_mean={rows[0]['v_mean']:.5g} v_max={rows[0]['v_max']:.5g} "
                       f"ratio_mean={rows[0]['relative_noise_ratio_mean']:.5g} "
                       f"ratio_max={rows[0]['relative_noise_ratio_max']:.5g} "
                       f"slope={rows[0]['state_slope']:.5g} innovation={rows[0]['innovation_mean']:.5g}")
            # 只输出该噪声层最后一次 CG 的相对“线性方程”残差。
            # 这里的 CG relative residual ≠ MRI k-space data-consistency residual。
            report(f"pULA level {level + 1}/{len(selected_sigmas)} sigma={sigma_net:.5g} "
                   f"CG relative residual={info['relative_residual']}")

        if save_handoff is not None:
            save_handoff(state.squeeze(1))

        # ========================= 5. 切换到 DPS tail =========================
        # 当前 state 就是 pULA 的终态；没有再抽取新的初始图像。
        # DPS 阶段改用传入的 raw MRI 算子/测量（与 pULA 白化算子不同）。
        if dps_sigmas:
            if dps_operator is None or dps_measurement is None:
                raise ValueError("DPS tail requires the original unwhitened MRI operator and measurement.")
            # 每个 DPS 噪声层有 dps_inner_loops 个 patch 网格偏移。
            if len(dps_shifts) != len(dps_sigmas) * dps_inner_loops:
                raise ValueError("DPS tail shift count does not match tail levels and inner loops.")
            # GPU 同步后开始单独计量 DPS tail 的墙钟时间。
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dps_tail_start = perf_counter()
            # pULA 只维护 FOV 内 [H,W]；原始 PaDIS-DPS 在图像四周补 pad 像素。
            # 在这里把 pULA 终态填到中心，外部 padding 初值为零。
            padded = F.pad(state, (oracle.pad,) * 4)
            # 切片用于从 padded 复数图像中取出真正的 384x384 FOV。
            fov = slice(oracle.pad, oracle.pad + oracle.image_size)
            shift_index = 0
            # 只在 DPS tail 的反传阶段开启自动微分；pULA 本身仍无梯度。
            with torch.enable_grad():
                # 按原始 PaDIS 的 Karras/VE 调度（末 N 层）进行 DPS。
                for outer, sigma in enumerate(dps_sigmas):
                    sigma = float(sigma)
                    # 沿用原始 PaDIS 经验更新幅度 alpha = 0.5*sigma^2。
                    alpha = 0.5 * sigma ** 2
                    for inner in range(dps_inner_loops):
                        shift = dps_shifts[shift_index]
                        shift_index += 1
                        # 每个 inner-loop 截断上一次计算图，只对当前 state 建立图。
                        current = padded.detach().requires_grad_(True)
                        # 原始 PaDIS 的再加噪步骤：x_noisy = x + sigma*n，
                        # 这里 n 为 PyTorch 复标准正态，每实通道方差 1/2。
                        noisy = current + sigma * native_complex_noise(current, system.generator)
                        # 复用同一个 PatchScore 去噪网络，但本次保留 autograd 图。
                        denoised = oracle.denoise_padded(noisy, sigma, shift)
                        denoised_fov = denoised[:, :, fov, fov]
                        # MRI 数据残差 r = y_raw - A_raw*D_patch(x_noisy,sigma)。
                        # 与 pULA 的 likelihood A_w^H(y_w-A_w*x) 不是同一个对象。
                        residual = dps_measurement - dps_operator.forward(real_channels(denoised_fov))
                        # r_norm 是多线圈/全采样 k-space 的整体 L2 范数（非均方误差）。
                        residual_norm = torch.linalg.vector_norm(residual.reshape(residual.shape[0], -1), dim=-1)
                        # 数据一致性损失 L_DC = ||r||_2^2。
                        sse = residual_norm.square()
                        with section("dps_backward_seconds"):
                            # 反传至 current：包含 denoiser 的 Jacobian，即 DPS 的重要开销。
                            likelihood_grad = torch.autograd.grad(sse.sum(), current)[0]
                        # score 用于 PaDIS 原始扩散更新：(D_patch(x_noisy)-x_noisy)/sigma^2。
                        score = (denoised - noisy) / sigma ** 2
                        # 归一化的 DPS 数据一致性梯度下降；zeta 为经验系数。
                        updated = current - (dps_zeta / residual_norm[:, None, None, None]) * likelihood_grad
                        # 再加一项 score 引导；保留作者原始更新结构。
                        updated = updated + (alpha / 2) * score
                        # 只要不是“最后一个 outer 噪声层”，每个 inner-loop 末都加噪。
                        # 这是原始 PaDIS-DPS 的循环规则，而不是每层只加一次噪声。
                        if outer < len(dps_sigmas) - 1:
                            updated = updated + sqrt(alpha) * native_complex_noise(updated, system.generator)
                        padded = updated
                        # 保存每次 DPS 更新前计算的测量残差，用于后续诊断。
                        dps_records.append({
                            "outer_level": outer,
                            "inner_step": inner,
                            "sigma": sigma,
                            "measurement_residual_norm": residual_norm.detach().cpu().tolist(),
                        })
                    # 屏幕上打印的是该噪声层最后一个 inner-loop 的残差。
                    report(f"DPS tail level {outer + 1}/{len(dps_sigmas)} sigma={sigma:.5g} "
                           f"residual={residual_norm.detach().mean().item():.6g}")
            # 完成 DPS 后，去掉 padding，保留 FOV 内的最后状态。
            state = padded[:, :, fov, fov].detach()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dps_tail_seconds = perf_counter() - dps_tail_start

        # [可选] 额外再去噪一次以保存展示图；不覆盖最终主输出 state。
        if terminal:
            before = network_calls
            terminal_sigma = float(dps_sigmas[-1]) if dps_sigmas else sigmas[-1]
            last_shifts = shifts if pula_shifts_by_level is None else pula_shifts_by_level[-1]
            terminal_image = oracle.denoise(state, terminal_sigma, last_shifts[-1])
            terminal_calls = network_calls - before
        else:
            terminal_calls = 0

    # ========================= 6. 输出计时和诊断 =========================
    # 同步所有 GPU 操作后记录总重建耗时。
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timing = {"reconstruction_seconds": perf_counter() - wall_start}
    for name, intervals in events.items():
        timing[name] = sum(a.elapsed_time(b) / 1000 for a, b in intervals) if device.type == "cuda" else sum(intervals)
    # 整理前向次数、pULA CG 次数、DPS 反传次数及各阶段耗时。
    # 请注意 main_network_forwards 统计“神经网络 forward 次数”，不等同于噪声层数。
    timing.update(
        pula_variant=pula_variant, sigma_data=sigma_data, gamma=gamma,
        pa_balance=pa_records,
        logical_updates=len(cg_records) + len(dps_records),
        logical_nfe=len(cg_records) + len(dps_records),
        pula_convention=system.convention,
        cg_seconds=timing["initialization_cg_seconds"] + timing["update_cg_seconds"],
        network_forwards=network_calls,
        main_network_forwards=network_calls - diagnostic_calls - terminal_calls,
        diagnostic_network_forwards=diagnostic_calls, terminal_network_forwards=terminal_calls,
        # pULA 的 Langevin 更新数；若 K=4、pULA_levels=23，则为 92。
        langevin_updates=len(cg_records), cg_solves=1 + len(cg_records),
        cg_iterations=init_info["iterations"] + sum(row["iterations"] for row in cg_records),
        a_calls=system.forward_calls, ah_calls=system.adjoint_calls,
        # dps_records 每个条目对应一次 DC 反传；DPS24x10 = 240 次。
        network_backprops=len(dps_records), dps_updates=len(dps_records),
        dps_a_calls=len(dps_records), pula_levels=len(selected_sigmas),
        dps_tail_seconds=dps_tail_seconds,
        initialization_cg=init_info, update_cg=cg_records, dps_tail=dps_records,
    )
    timing["pula_seconds"] = timing["reconstruction_seconds"] - dps_tail_seconds
    if device.type == "cuda":
        timing.update(
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30,
        )
    # 主输出是复数图像 [B,H,W]；另有可选终态去噪图、计时、诊断数据。
    return state.squeeze(1), None if terminal_image is None else terminal_image.squeeze(1), timing, diagnostics
