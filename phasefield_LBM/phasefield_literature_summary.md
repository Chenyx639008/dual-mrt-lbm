# Phase-Field LBM 四篇文献综述：共同点、方法一致性与具体参数值

> 生成日期：2026-08-07
> 转换方式：OpenDataLoader PDF **hybrid 模式**（`--hybrid docling-fast --hybrid-mode full` + 后端 `--enrich-formula`）
> 原始输出：`phasefield_LBM/output/*.md`（含 LaTeX 公式的 Markdown）与 `*.json`（含公式边界框）

---

## 1. 公式识别质量说明（回答：LaTeX 模式？）

**是的，hybrid 模式 + `--enrich-formula` 输出的是 LaTeX 格式公式**，以 `$...$`（行内）与 `$$...$$`（独立公式块）嵌入 Markdown 中。

质量评估：

| 项目 | 质量 | 说明 |
|------|------|------|
| 希腊字母/算子 | ✅ 优秀 | `\nabla` `\partial` `\sum` `\tanh` `\ln` `\mathbf` `\omega` 均正确 |
| 分数/上下标 | ✅ 优秀 | `\frac{}{}`、`^`、`_` 正确（含字符间空格，LaTeX 可正常编译） |
| 矩阵/分段 | ✅ 良好 | `\begin{bmatrix}`、`\begin{cases}` 保留 |
| 正文文字 | ✅ 良好 | 标题、作者、DOI 准确（如 liang2018 的 DOI 10.1103/PhysRevE.97.033309） |
| 个别瑕疵 | ⚠️ 少数 | ① 变量间偶然夹入 OCR 噪声词（如 "conf" "Ocaca" "Navighscr"）；② S16 的 `ρu` 宏观公式被误识别为 `\frac{\sigma}{c_s^2}`，真实应为 $\rho\mathbf u=\sum_a\mathbf e_a f_a+\frac{\Delta t}{2}\mathbf F$（Guo 格式）；③ 大矩阵被截断 |

**结论**：公式数学结构准确、可直接用于核对实现；个别行需人工复核（总结中已修正）。

---

## 2. 四篇论文身份一览

| # | 论文 | 期刊/年 | 求解器 | 维度 | 角色 |
|---|------|--------|--------|------|------|
| 1 | **Liang et al. 2018**, *Phase-field-based LB modeling of large-density-ratio two-phase flows* | Phys. Rev. E 97, 033309 | **BGK** | 2D | Yang 2024 的 2D 母参考（核心） |
| 2 | **Yang et al. 2024**, *Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT images* | The Innovation Energy 1(1):100015 | **D3Q19 MRT** | 3D | 本项目核心母模型 |
| 3 | **Zhang et al. 2023**, *Central-moment discrete unified gas-kinetic scheme for incompressible two-phase flows with large density ratio* | J. Comput. Phys. | **Central-moment DUGKS** | 2D | 方法学参考（高稳定性方案） |
| 4 | **Zarareh et al. 2021**, *Improving the staircase approximation for wettability implementation of phase-field model: Part 1 – Static contact angle* | Comput. Math. Appl. | **BGK** | 2D | Stage 3 润湿实现参考 |

---

## 3. 各论文要点

### 3.1 Liang 2018（PRE 97, 033309）
- 双 LB 方程：一个解**保守 Allen-Cahn (AC) 方程**，一个解**不可压 NS 方程**。
- 关键创新：为 NS 方程设计了一个**简化力分布函数**，比既有 AC-LB 模型更简单、精度更高。
- 验证算例：静态液滴、层状 Poiseuille 流、spinodal 分解、液滴撞击液膜（密度比 1000，Re 20–500）。
- 伪速度量级：静态 $|\mathbf u|_{\max}\sim10^{-9}$；密度比 10–1000 时 $\sim10^{-8}$。
- 基准对比：改进 Shan-Chen $10^{-3}$、color-gradient $10^{-5}$、Cahn-Hilliard-LB $10^{-6}$ → **本文模型伪速度最小**。

### 3.2 Yang 2024（The Innovation Energy, 本项目母模型）
- 3D 微-CT 孔隙尺度的甲烷水合物分解模拟。
- **D3Q19 MRT-LB** 解保守 AC（S10）与不可压 NS（S15）；**D3Q7 MRT CST-LB** 解溶质传质（相分数指示方案）。
- 表面能润湿（S17）在复杂孔隙中的实现，球形固体颗粒验证 $\theta=60°$。
- 化学势形式毛细力：$F_c=[4\beta\phi(\phi-1)(\phi-0.5)-\kappa\nabla^2\phi]\nabla\phi$。

### 3.3 Zhang 2023（JCP, central-moment DUGKS）
- **中心矩碰撞算子**（基于连续 Maxwell 平衡分布，非离散版）改善伽利略不变性。
- 速度型分布函数 + 恒定密度，消除恢复方程中与密度比成正比的分部积分误差；压力仅出现在零阶矩。
- **Strang 分裂**处理外力项，避免界面处空间导数计算，提升稳定性。
- 保守 AC 界面捕获 + 粘度线性插值。

### 3.4 Zarareh 2021（CMWA, 润湿实现）
- 针对**圆形表面（staircase 近似）**上的润湿实现，提出两种处理流-固相互作用的方案。
- 二元系统：**表面能（cubic formulation）+ 几何润湿**；三元系统：表面能模型（显式/隐式润湿条件）。
- **隐式润湿条件**（边界节点序参数取相邻流体/固体节点平均值）显著提高接触角精度。
- 插值法（vs round-off）选取相邻节点信息更准确。

---

## 4. 共同点

1. **界面捕获全部采用保守 Allen-Cahn 方程**（Yang 明确说明改用 AC 是因 CH 需高阶微分、非局部实现受限）。
2. **表面张力均用"化学势 × ∇φ"（potential form）**（Zhang 2023 除外，其用曲率形式）。
3. **均面向大密度比两相流**（100 以上，最高 1000）。
4. **表面能润湿方案是主流**（Yang 2024 与 Zarareh 2021 都采用；Yang 因几何方案在 3D 复杂孔隙中需足够界面节点而弃用）。
5. 宏观量恢复均含**半时步力修正**（Guo 格式）：$p=\sum f+\frac{\Delta t}{2}\mathbf u\cdot\nabla\rho c_s^2$，$\rho\mathbf u=\sum\mathbf e_a f_a+\frac{\Delta t}{2}\mathbf F$。
6. 伪速度抑制是共同验证指标（Liang：$10^{-9}$ 量级）。

---

## 5. 方法一致性（公式对照）

### 5.1 保守 AC 方程 —— 完全一致

Liang (式1) 与 Yang (S10)：

$$\frac{\partial\phi}{\partial t}+\nabla\cdot(\phi\mathbf u)=\nabla\cdot M\left\{\nabla\phi-\frac{1}{W}\left[1-\tanh^2\left(\frac{1}{2}\ln\frac{\phi}{1-\phi}\right)\right]\mathbf n\right\},\quad \mathbf n=\frac{\nabla\phi}{|\nabla\phi|}$$

- 反扩散系数：$\lambda=\frac{4\phi(1-\phi)}{W}$（Liang 式4）= Yang 的 $\frac{1}{W}[1-\tanh^2(\cdot)]$（两者数学等价，后者为 $\tanh$ 剖面形式）。

### 5.2 化学势与表面张力参数 —— 完全一致

| 量 | Liang (式6,7) | Yang (S14) | Zarareh | 一致 |
|----|--------------|------------|---------|------|
| 化学势 | $\mu_\phi=4\beta\phi(\phi-1)(\phi-0.5)-k\nabla^2\phi$ | $F_c=\mu_\phi\nabla\phi$ | 同形式 | ✅ |
| 梯度系数 | $k=\frac{3}{2}\sigma W$ | $\kappa=\frac{3\sigma W}{2}$ | — | ✅ |
| 双阱系数 | $\beta=\frac{12\sigma}{W}$ | $\beta=\frac{12\sigma}{W}$ | — | ✅ |
| 力项 | $F_s=\mu_\phi\nabla\phi$ | $F_c=[4\beta\phi(\phi-1)(\phi-0.5)-\kappa\nabla^2\phi]\nabla\phi$ | — | ✅ |

### 5.3 求解器差异（关键区别）

| 论文 | AC 求解器 | NS 求解器 | 松弛格式 |
|------|----------|----------|---------|
| Liang 2018 | BGK | BGK | $\tau_f,\tau_g$ 单松弛 |
| Yang 2024 | **D3Q19 MRT** | **D3Q19 MRT** | 矩空间对角阵 |
| Zhang 2023 | central-moment DUGKS | central-moment DUGKS | 中心矩 |
| Zarareh 2021 | BGK | BGK | 单松弛 |

> ⚠️ **对 Stage 3 的关键启示**：Yang 2024（MRT）在 3D 大密度比下稳定；其 2D 母参考 Liang 2018 是 BGK。我们的 2D 实现（`pf_ns_2d`）要复现 Yang 的定量润湿，需按 **Yang 的 D3Q19 MRT 公式降维到 D2Q9 MRT**（`Λ_φ = diag(1, ω_φ, ω_φ, ω_φ, 1, ...)`）。

### 5.4 MRT 松弛矩阵与迁移率 —— Yang S5.1（Stage 3 直接实现依据）

- AC：$\Lambda_\phi=\mathrm{diag}(1,\ \omega_\phi,\ \omega_\phi,\ \omega_\phi,\ 1,\ 1,\ ...)$，迁移率
  $$M=c_s^2\left(\frac{1}{\omega_\phi}-\frac12\right)\frac{\Delta x^2}{\Delta t}$$
- AC 源项力：$F_\phi=\frac{1}{3W}\left[1-\tanh^2\left(\frac12\ln\frac{\phi}{1-\phi}\right)\right]$
- NS：$\Lambda=\mathrm{diag}(1,\ 1,\ 1,\ \omega_\nu,\ \omega_\nu,\ \omega_\nu,\ \omega_\nu,\ \omega_\nu,\ \omega_\nu,\ 1,\ ...)$，粘度
  $$\upsilon=c_s^2\left(\frac{1}{\omega_\nu}-\frac12\right)\frac{\Delta x^2}{\Delta t}$$
- 密度插值：$\rho=\rho_g+\phi(\rho_w-\rho_g)$（Yang S13）

### 5.5 润湿边界 —— Yang S17（积分）与 Zarareh（微分）等效

- **Yang 2024（表面能，积分形式）**：
  $$\phi_s=\frac{2}{a}\left(1+\frac{a}{2}-\sqrt{\left(1+\frac{a}{2}\right)^2-2a\phi_f}\right)-\phi_f,\quad a=-l_{sf}\sqrt{\frac{2\beta}{\kappa}}\cos\theta\ (\theta\neq90°)$$
  $\theta=90°$ 时 $\phi_s=\phi_f$；相邻流体节点按固体法向投影选择。
- **Zarareh 2021（表面能，微分形式，等价）**：
  $$\varepsilon_s(\phi_s)=\frac{b_1}{2}\phi_s^2-\frac{b_1}{3}\phi_s^3,\qquad \mathbf n\cdot\nabla\phi_s=-\sqrt{\frac{2\beta}{\kappa}}\,\phi_s(1-\phi_s)\cos\theta_s$$
  Young 方程：$\cos\theta_s=\frac{\sigma_{sg}-\sigma_{sl}}{\sigma}=-\frac{b_1}{\sqrt{2\kappa}\beta}$。

---

## 6. 具体参数值汇总

### 6.1 Liang 2018（静态液滴 / 层状 Poiseuille）

| 参数 | 值 |
|------|-----|
| 界面厚度 | $W=5$ |
| 表面张力 | $\sigma=0.001$ |
| 液相运动粘度 | $\nu_l=0.1$ |
| 迁移率 | $M=0.1$ |
| 密度比 | $\rho_l/\rho_g=10,\ 100,\ 150,\ 1000$ |
| 静态液滴 | $R=50,\ N_y\times N_x=200\times200$，周期性边界 |
| 层状流壁面 | halfway bounce-back |
| 伪速度 | 静态 $10^{-9}$；密度比 10–1000 时 $10^{-8}$ |

### 6.2 Yang 2024（D3Q19 MRT phase-field）

| 参数 | 表达式 |
|------|--------|
| 迁移率-松弛 | $M=c_s^2(\frac{1}{\omega_\phi}-\frac12)\frac{\Delta x^2}{\Delta t}$ |
| 粘度-松弛 | $\upsilon=c_s^2(\frac{1}{\omega_\nu}-\frac12)\frac{\Delta x^2}{\Delta t}$ |
| AC 源项 | $F_\phi=\frac{1}{3W}[1-\tanh^2(\frac12\ln\frac{\phi}{1-\phi})]$ |
| 表面张力参数 | $\beta=\frac{12\sigma}{W},\ \kappa=\frac{3\sigma W}{2}$ |
| 润湿验证 | 球形颗粒 $\theta=60°$ |
| 密度插值 | $\rho=\rho_g+\phi(\rho_w-\rho_g)$ |

### 6.3 Zarareh 2021（润湿实现）

- 界面剖面：$\phi(z)=\frac{\phi_H+\phi_L}{2}+\frac{\phi_H-\phi_L}{2}\tanh\left(\frac{2z}{W}\right)$
- 固体壁自由能：$\varepsilon_s(\phi_s)=\frac{b_1}{2}\phi_s^2-\frac{b_1}{3}\phi_s^3$
- 接触角：$\cos\theta_s=-\frac{b_1}{\sqrt{2\kappa}\beta}$
- 润湿 BC：$\mathbf n\cdot\nabla\phi_s=-\sqrt{\frac{2\beta}{\kappa}}\phi_s(1-\phi_s)\cos\theta_s$
- 方法：staircase 近似 + 插值法选择相邻节点；隐式润湿条件（取相邻流/固节点平均值）

### 6.4 Zhang 2023（central-moment DUGKS）

- 粘度插值：$\nu=\nu_l+\frac{\phi-\phi_l}{\phi_h-\phi_l}(\nu_h-\nu_l)$
- 表面张力（曲率形式）：$F_{sf}=-\sigma(\nabla\cdot\mathbf n)\,\phi\,\frac{2\rho}{(\rho_n+\rho)(\phi_n-\phi_l)}$
- 力项处理：Strang 分裂（半时步前后各施加一次）

---

## 7. 对 Phase-Field 开发的启示（衔接 Stage 3）

1. **MRT-LB AC 的 2D D2Q9 定义**：由 Yang 的 D3Q19 公式降维得到（`Λ_φ = diag(1, ω_φ, ω_φ, ω_φ, 1, 1, 1, 1, 1)`），松弛对角元前 3 个为 ω_φ（对应动量矩）、其余为 1；迁移率 $M=c_s^2(1/\omega_\phi-\frac12)\Delta x^2/\Delta t$；源项力 $F_\phi=\frac{1}{W}[1-\tanh^2(\frac12\ln\frac{\phi}{1-\phi})]$（Yang 的 1/3W 是 D3Q19 的 ω 加权，D2Q9 需按离散速度权重调整）。
2. **反扩散源必须作用在矩空间**：Liang 用 BGK 的 $F_i=(1-\frac{1}{2\tau_f})\frac{\omega_i c_i\cdot[\partial_t(\phi\mathbf u)+c_s^2\lambda]}{c_s^2}$；MRT 版本须将源项变换到矩空间（Yang S5.1 的 $\mathbf R_\phi$），这解释了之前"直接 FD AC 反扩散无效"的失败原因。
3. **润湿**：优先采用 Yang S17 积分表面能方案 + 相邻流体节点选择（球形颗粒验证 θ=60° 为基准测试）。
4. **伪速度目标**：MRT-LB AC 应达到 $10^{-8}$~$10^{-9}$ 量级（Liang 报告），远超当前直接 FD AC 的 $1.1\times10^{-4}$。

---

## 附：转换命令备忘

```bash
# 后端（常驻）
nohup opendataloader-pdf-hybrid --port 5002 --enrich-formula > /tmp/hybrid_backend.log 2>&1 &

# 客户端（公式富集必须 --hybrid-mode full）
opendataloader-pdf --hybrid docling-fast --hybrid-mode full \
  -f markdown,json -o output/ <pdf1> <pdf2> ...
```

- 输出：`*.md`（Markdown + LaTeX 公式）、`*.json`（含每元素边界框与公式 type）、`*_images/`（图）。
- 注意：CPU 环境下每篇约 1 小时（docling 模型 + easyocr），批量一次调用可复用 JVM。
