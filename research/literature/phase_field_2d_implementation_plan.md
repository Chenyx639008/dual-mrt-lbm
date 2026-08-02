# Yang 2024 相场 LB 模型与无量纲框架的 2D 实现方案

> **日期**: 2026-07-29
> **数据源**: Yang 2024 (Innov Energy) SI 逐字公式（Note S1.2, Eq. S4–S41，PDF 页 15–26 直读）
> **目标**: (1) 理解 Yang 为何从伪势 SC 转向相场；(2) 给出相场模型的完整公式体系与 2D 适配；(3) 给出无量纲分析框架的落地；(4) 针对现有代码的详细修改步骤；(5) 坑与经验
> **姊妹文档**: `literature/SC_vs_phasefield_and_3D_migration_guide.md`（学习方法论）、`code_mechanism_optimization_plan.md`（总体优化路线）

---

## 目录

1. [为什么 Yang 2024 放弃伪势 SC：先进考量的证据链](#1-为什么-yang-2024-放弃伪势-sc)
2. [相场模型完整公式体系（Yang 2024 原版，逐字）](#2-相场模型完整公式体系)
3. [2D 适配公式（D2Q9/D2Q5，对接现有代码）](#3-2d-适配公式)
4. [无量纲分析框架（2D 形成问题落地版）](#4-无量纲分析框架)
5. [代码修改详细步骤（Phase PF-0 ~ PF-5）](#5-代码修改详细步骤)
6. [坑与经验（相场专属）](#6-坑与经验)
7. [决策建议：你现在该不该切相场](#7-决策建议)

---

## 1. 为什么 Yang 2024 放弃伪势 SC

### 1.1 证据链：这不是一次"升级"，而是被坑出来的"撤退"

Yang 系列五篇论文提供了完整的决策演化证据：

| 阶段 | 论文 | SC 模型的遭遇 | 教训 |
|---|---|---|---|
| 全 SC | 2021a CEJ | intra+inter 力 + C-S EOS 实现高密度比 | 伪速度主要出现在气相（"the spurious current mainly occurs in the gas phase"） |
| **妥协** | 2022 CEJ | **砍掉 intra-component 力、密度比设为 1**（ρ_w=ρ_g=100 kg/m³） | 伪速度引起的非物理质量通量**淹没水中低扩散甲烷的真实扩散通量**——为传质精度牺牲多相流真实性 |
| 放弃 | 2024 Innov | 彻底切换 phase-field AC | SI p.7 原话：几何润湿方案"implementation complexity in three-dimensional complicated pore structure is unacceptable"；SC 的间接润湿（G_aw 调接触角）在 micro-CT 复杂曲面上无法逐点精确控制 |

**核心结论**：SC 的两大软肋——**伪速度污染低扩散传质**、**间接润湿无法控制复杂曲面接触角**——在 3D 真实几何中不可接受。相场用显式序参数 φ 一次性解决：表面张力直接给定（σ 独立参数）、润湿用表面能方案精确到点、守恒 AC 方程严格保质量。

### 1.2 相场 vs SC 的能力矩阵（与你的 mode 4 直接相关）

| 维度 | SC 伪势（你的现状） | 相场 AC（Yang 2024） | 对你哪个 mode 关键 |
|---|---|---|---|
| 界面表示 | 隐式（密度场自然分离） | 显式 φ∈[0,1] | mode 4（要精确知道"界面在哪"） |
| 界面张力 | EOS+分子力间接调 | σ 直接输入（β=12σ/W, κ=3σW/2） | mode 4（Ca 数精确控制） |
| 润湿性 | G_aw 映射（间接、需标定） | 表面能方案（逐点精确） | 多材料壁面 |
| 伪速度 | 有（气相显著） | 近零（热力学一致） | **mode 4 界面通量** |
| 密度比 | PR-EOS 可达 80+ | 插值 ρ=ρ_g+φ(ρ_w−ρ_g)，Yang 实测 10/62.5 | — |
| 质量守恒 | EOS 界面处可能密度漂移 | 守恒 AC 严格 | 你的 T1 质量守恒叙事加分 |
| 界面法向 | 需从密度梯度算 | n=∇φ/\|∇φ\| 直接可用 | mode 4 成核判据 |
| 计算成本 | 1 套 D2Q9 | 2 套格子（AC + NS） | 成本 ~1.5–2× |

### 1.3 对你的适用性判断（先给结论，论证在 §7）

- mode 0–2（分解/ghost 生长/动态平衡）：SC 够用，相场收益小
- **mode 4（气-水界面成核）：相场可能是解锁器**——界面位置、界面法向、界面张力、润湿全部显式可控，伪速度近零不污染界面通量
- 3D 未来：相场是 Yang 验证过的唯一路径

---

## 2. 相场模型完整公式体系（Yang 2024 原版，逐字）

> 以下公式逐字摘自 Yang 2024 SI Note S1.2 与 Note S5（PDF 页 15–26）。编号保持原文（S10–S41）。3D 形式；2D 适配见 §3。

### 2.1 界面捕捉：改进守恒 Allen-Cahn 方程

$$\frac{\partial \phi}{\partial t} + \nabla\cdot(\phi\mathbf{u}) = \nabla\cdot M\left\{\nabla\phi - \frac{1}{W}\left[1-\tanh^2\left(\frac{1}{2}\ln\frac{\phi}{1-\phi}\right)\right]\mathbf{n}\right\} \tag{S10}$$

- φ=1 水相，φ=0 气相；M = mobility（迁移率）；W = 界面厚度尺度；**n = ∇φ/|∇φ|**（界面法向）
- 平衡剖面自洽：tanh 形式 $\phi(z) = \frac{1}{2}[1+\tanh(2z/W)]$（可代入验证 $\frac{1}{2}\ln\frac{\phi}{1-\phi} = 2z/W$）
- 反扩散项 $[1-\tanh^2(\cdot)]$ 在界面中心（φ=0.5）取峰值 1、在两相体相为 0——**只在界面处工作**，这是"守恒"与"锐界面"兼得的关键
- 选 AC 不选 CH 的原因（SI p.5）：CH 需高阶导数与非局部实现，3D 复杂孔隙受限；守恒 AC（Chiu & Lin 2011）保证质量守恒

### 2.2 AC 方程的 MRT-LB 求解（D3Q19 原版）

$$\mathbf{f}_\phi^*(\mathbf{x},t) = \mathbf{f}_\phi(\mathbf{x},t) - \mathbf{M}^{-1}\left[\boldsymbol{\Lambda}_\phi\left(\mathbf{m}_\phi - \mathbf{m}_\phi^{eq}\right) + \Delta t\left(\mathbf{I}-\frac{\boldsymbol{\Lambda}_\phi}{2}\right)\mathbf{R}_\phi\right] \tag{S11}$$

$$\phi = \sum_\alpha f_{\phi,\alpha} \tag{S12}$$

$$\rho = \rho_g + \phi(\rho_w - \rho_g) \tag{S13}$$

**平衡矩**（D3Q19，Eq. S34）：

$$\mathbf{m}_\phi^{eq} = \left[\phi,\; \phi u_x,\; \phi u_y,\; \phi u_z,\; 0,\; 0,\; 0,\; \phi,\; 0,\; 0,\; \phi u_x c_s^2,\; \phi u_y c_s^2,\; \phi u_z c_s^2,\; \phi u_y c_s^2,\; \phi u_x c_s^2,\; \phi u_z c_s^2,\; \phi c_s^4,\; \phi c_s^4,\; \phi c_s^4\right]^T$$

**源项**（Eq. S35）：$R_\phi$ 的动量行为 $F_{\phi,i} + \partial_t(\phi u_i)$，其余按矩结构布置（高阶矩乘 $c_s^2$），其中

$$\mathbf{F}_\phi = \frac{1}{3W}\left[1-\tanh^2\left(\frac{1}{2}\ln\frac{\phi}{1-\phi}\right)\right]\mathbf{n}$$

（注意系数 **1/(3W)**：1/3 = 格子单位 $c_s^2$。）

**松弛矩阵与迁移率**：

$$\boldsymbol{\Lambda}_\phi = \mathrm{diag}(1, \omega_\phi, \omega_\phi, \omega_\phi, 1, 1, \dots, 1) \tag{S36}$$

$$M = c_s^2\left(\frac{1}{\omega_\phi}-\frac{1}{2}\right)\frac{\Delta x^2}{\Delta t} \tag{S37}$$

### 2.3 不可压 N-S 与毛细力（D3Q19 原版）

**毛细力（化学势形式）**：

$$\mathbf{F}_c = \left[4\beta\phi(\phi-1)(\phi-0.5) - \kappa\nabla^2\phi\right]\nabla\phi, \qquad \beta = \frac{12\sigma}{W},\quad \kappa = \frac{3\sigma W}{2} \tag{S14}$$

- 自洽关系：$\sigma = \sqrt{\kappa\beta/18}$，$W = \sqrt{8\kappa/\beta}$（σ、W 为输入，β、κ 派生）

**MRT-LB**（压力分布函数格式）：

$$\mathbf{f}^* = \mathbf{f} - \mathbf{M}^{-1}\left[\boldsymbol{\Lambda}\left(\mathbf{m} - \mathbf{m}^{eq}\right) + \Delta t\left(\mathbf{I}-\frac{\boldsymbol{\Lambda}}{2}\right)\mathbf{R}\right] \tag{S15}$$

**宏观量**：

$$p = \sum_\alpha f_\alpha + \frac{\Delta t}{2}\mathbf{u}\cdot\nabla\rho c_s^2, \qquad \rho\mathbf{u} = \frac{\sum_\alpha f_\alpha\mathbf{e}_\alpha}{c_s^2} + \frac{\Delta t}{2}\mathbf{F} \tag{S16}$$

总力 $\mathbf{F} = -\nabla(p - \rho c_s^2) + \mathbf{F}_c + \mathbf{F}_b$（体力）。$c_s = \Delta x/(\sqrt{3}\Delta t)$。

**平衡矩与源项**（Eq. S38/S39，19 分量，含 $\rho c_s^2 u_i u_j$、$p+\rho c_s^2|\mathbf{u}|^2$、$c_s^4(p+\rho u_i^2+\rho u_j^2-0.5\rho u_k^2)$ 等；$R$ 为 $\mathbf{u}\cdot\nabla\rho c_s^2$、$F_{b,i}c_s^2$、$\partial_i(\rho c_s^2)$ 及其组合的矩空间布置——实现时按 D3Q19 矩结构逐行照抄 SI 原文，2D 适配见 §3.2）

**松弛与粘度**：

$$\boldsymbol{\Lambda} = \mathrm{diag}(1,1,1,1,1,\; \omega_\upsilon, \omega_\upsilon, \omega_\xi, \omega_\xi, \omega_\upsilon, \omega_\xi, \omega_\upsilon,\; 1,1,\dots,1) \tag{S40}$$

$$\upsilon = c_s^2\left(\frac{1}{\omega_\upsilon}-\frac{1}{2}\right)\frac{\Delta x^2}{\Delta t} \tag{S41}$$

$\omega_\xi$ 可调以改善稳定性（非正交矩集，引同组 Wang et al. 2023 IJMF 168）。

### 2.4 润湿边界：表面能方案

$$\phi_s = \frac{2}{a}\left(1+\frac{a}{2}-\sqrt{\left(1+\frac{a}{2}\right)^2 - 2a\phi_F}\right) - \phi_F, \qquad a = -l_{sf}\sqrt{\frac{2\beta}{\kappa}}\cos\theta \quad (\theta \neq 90°) \tag{S17}$$

θ=90° 时 $\phi_s = \phi_F$。$\phi_F$ = **沿固体法向 n_s 方向的相邻流体节点**的序参数；n_s = −∇s/|∇s|（s=1 固体、0 流体）；$l_{sf}$ 为格子长度尺度（通常 = Δx）。

**为何不用几何方案**（SI p.7 原话）：几何方案需要足够节点构造界面，"its implementation complexity in three-dimensional complicated pore structure is unacceptable"。

### 2.5 浓度/热场/VOP 的角色（与你现有完全一致）

- **CST**（S18–S23）：相分数直接用 φ（不再需要 SC 密度比计算 x_w！）：$C = \phi C_w + (1-\phi)C_g$，$D = D_wD_g/[\phi D_w + (1-\phi)D_g]$，$R_{CST} = \frac{1}{4}C\frac{H-1}{H\phi+(1-\phi)}(0, \partial_x\phi, \partial_y\phi, \partial_z\phi, 0,0,0)^T$（**3D D3Q7 系数 1/4；2D D2Q5 为 1/3**）
- **热场**（S24–S27）：双分布 + 反应热源 $R_d$（$S_d = -\Pi\cdot n_h A_s \Delta H$）+ 共轭源项 $R_c$
- **VOP**（S28）：$V_h(t+\Delta t) = V_h(t) - \Pi\cdot n_h A_s V_m$；**转化规则：V_h→0 时新流体节点强制 φ=1（生成水）、C=C_eq（立即饱和）**
- **耦合顺序**（SI Fig S2）：流场 → 传质 → 传热 → 固相更新，单向无内迭代

---

## 3. 2D 适配公式

### 3.1 总体映射策略

| 模块 | Yang 2024 (3D) | 你的 2D 适配 | 与现有代码关系 |
|---|---|---|---|
| AC 界面捕捉 | D3Q19 MRT | **D2Q9 MRT**（新写）或 D2Q5 MRT（复用浓度场骨架） | 新增 |
| 不可压 NS | D3Q19 MRT（压力格式） | **D2Q9 MRT**（新写，替换 SC 流场） | **替换** `mrt_collide_two_components_gpu` |
| 浓度 CST | D3Q7 MRT | **保持现有 D2Q5 MRT-CST**（仅 x_w 来源改 φ） | 改 ~10 行 |
| 热场 DDF | D3Q7 MRT | **保持现有 D2Q5 MRT-DDF**（不动） | 不动 |
| Kang 反应边界 | 同 2D | **不动**（hydrate.cu:822-988） | 不动 |
| VOP + ghost 重建 | 同 2D | **不动**（hydrate_vop.cu 全部） | 不动 |
| 润湿 | 表面能 + 26 邻居查表 | **表面能 + 8 邻居最近方向** | 替换 G_aw 映射 |

**关键洞察**：Yang 2024 的四模块中，**只有流场（AC+NS）需要重写**；浓度、热场、Kang 边界、VOP 与你的 D2Q5/ghost 体系同构——这与 `SC_vs_phasefield_and_3D_migration_guide.md` §4.3 的"80% 代码可复用"结论一致。

### 3.2 AC 方程的 2D 实现（两条路径）

#### 路径 A：D2Q5 MRT（推荐先做——复用现有 M5 骨架，风险最低）

AC 方程（S10）与浓度方程（S18）同为对流-扩散型标量方程，**可直接镜像你现有的 `kernel_collide_conc` 结构**：

- 分布函数 $f_{\phi,k}$，k=0..4（与 g_in 同布局 `idx5(k,x,y)`）
- 矩向量 $\mathbf{m}_\phi = M_5 \mathbf{f}_\phi = [\phi, j_x, j_y, e, p]^T$（复用现有 M5/Minv5 矩阵！）
- **平衡矩**：$\mathbf{m}_\phi^{eq} = [\phi,\; \phi u_x,\; \phi u_y,\; \tfrac{3}{4}\phi,\; 0]^T$（与浓度场同构）
- **弛豫矩阵**：$\Lambda_\phi = \mathrm{diag}(1, \omega_\phi, \omega_\phi, 1, 1)$
- **迁移率**：$M = c_{s,5}^2\left(\dfrac{1}{\omega_\phi}-\dfrac{1}{2}\right)\dfrac{\Delta x^2}{\Delta t}$，$c_{s,5}^2 = 1/3$
- **源矩**（注入动量行，Guo 式 (I−Λ/2) 前置因子）：

$$R_{\phi} = \left[0,\; F_{\phi,x} + \partial_t(\phi u_x),\; F_{\phi,y} + \partial_t(\phi u_y),\; 0,\; 0\right]^T$$

$$F_\phi = \frac{c_{s,5}^2}{W}\left[1-\tanh^2\left(\frac{1}{2}\ln\frac{\phi}{1-\phi}\right)\right]\mathbf{n}, \qquad \mathbf{n} = \frac{\nabla\phi}{|\nabla\phi|}$$

- **φ 截断**：$\phi \leftarrow \min(\max(\phi, \epsilon), 1-\epsilon)$，$\epsilon = 10^{-6}$（防 ln(0) 奇异）
- **∇φ**：各向同性二阶差分（D2Q9 权重 $w_i e_i \phi(x+e_i)/c_s^2$ 形式，与 LBM.cu 中现有梯度算子一致）
- **时间导数项**：$\partial_t(\phi u_i) \approx [(\phi u_i)^n - (\phi u_i)^{n-1}]/\Delta t$（一阶后差，需存上一步 φu）

**代价声明**：D2Q5 只有轴向 4 方向，界面法向 n 的各向同性弱于 D2Q9——**对圆形/曲面界面（气泡）精度有限**。用于原型验证与方法论学习；若 mode 4 气泡界面膜模拟对界面质量敏感，升级路径 B。

#### 路径 B：D2Q9 MRT（Yang 精神的 2D 对应——8 方向各向同性）

平衡矩按 S34 降级到 D2Q9 标准矩集（排序对齐 `LBM.h` 现有 M 矩阵）：

$$\mathbf{m}_\phi^{eq} = \left[\phi,\; \phi u_x,\; \phi u_y,\; \phi,\; 0,\; 0,\; 0,\; \phi u_x c_s^2,\; \phi u_y c_s^2\right]^T$$

（对应矩序 $[\phi, j_x, j_y, e, \varepsilon, q_x, q_y, p_{xx}\text{-类}, p_{xy}\text{-类}]$；具体按 LBM.h 的 M 行序调整——**实现时第一步是打印 M 的每行确认矩序**）

源矩 $R_\phi$：动量行放 $F_{\phi,i} + \partial_t(\phi u_i)$，能量/高阶行按 S35 结构置 $c_s^2$ 倍或 0。

> **参考实现文献**：Liang et al. (2018) PRE 97, 033309 给出 D2Q9 相场 MRT 的完整矩定义（正是 Yang 2024 refs 18–20 的 2D 母本）。路径 B 实施前建议先核对该文。

### 3.3 不可压 N-S 的 2D 实现（D2Q9）

采用压力分布函数格式（Yang S15–S16 的 2D 对应；Guo et al. 2000 族）：

**平衡分布**（分布函数形式，无歧义）：

$$f_0^{eq} = w_0\left[\frac{p}{c_s^2} - \frac{\rho(u_x^2+u_y^2)}{2c_s^2}\right]$$

$$f_i^{eq} = w_i\left[\frac{p}{c_s^2} + \rho\left(\frac{\mathbf{e}_i\cdot\mathbf{u}}{c_s^2} + \frac{(\mathbf{e}_i\cdot\mathbf{u})^2}{2c_s^4} - \frac{u_x^2+u_y^2}{2c_s^2}\right)\right], \quad i=1..8$$

**宏观量**（S16 的 2D 版）：

$$p = \sum_i f_i + \frac{\Delta t}{2}\mathbf{u}\cdot\nabla\rho c_s^2, \qquad \rho\mathbf{u} = \sum_i f_i\mathbf{e}_i + \frac{\Delta t}{2}\mathbf{F}$$

（D2Q9 标准 BGK 形式下 $\rho\mathbf{u} = \sum f_i e_i + \frac{\Delta t}{2}F$；Yang 的 $/c_s^2$ 来自其非标准速度定义，实现时统一为一种约定并在注释中写死）

**总力**：$\mathbf{F} = -\nabla(p - \rho c_s^2) + \mathbf{F}_c + \mathbf{F}_b$，$\mathbf{F}_c$ 按 S14（2D：∇²φ 用 9 点各向同性拉普拉斯，∇φ 同 §3.2）

**密度插值**（S13）：$\rho = \rho_g + \phi(\rho_w - \rho_g)$；**粘度插值**：$\nu = \nu_g + \phi(\nu_w - \nu_g)$（Yang 未明说但为相场惯例；或直接取 $\nu(\phi)$ 调和平均）

**体力**：$F_b = \rho G_x$（对接你现有 Gx 驱动）

### 3.4 2D 表面能润湿（S17 的 8 邻居简化）

Yang 3D 需要 26 邻居查表；2D 只需 8 邻居：

```
对每个固体邻居节点 s：
  ① n_s = -∇s/|∇s|  （s 为固体指示函数，中心差分）
  ② 在 8 个离散方向 e_i 中选出与 n_s 点积最大的方向 e_k
  ③ 取该方向的相邻流体节点 φ_F = φ(x + e_k)
  ④ 代入 S17 得 φ_s，写入固体节点
```

边界处理顺序：**每步 AC 流步完成后、碰撞前**刷新固体节点的 φ_s（保证 ∇φ 在界面处使用正确润湿）。

**多材料扩展**（你的石英 θ=30°/水合物 θ=80°）：a 按 mat 取不同 cosθ——直接对接 `d_wall_mat`，与现有 G_aw 多材料映射一一对应。

### 3.5 与 hydrate 管线的耦合点（修改清单）

| # | 耦合点 | 现状（SC） | 相场改法 | 位置 |
|:-:|---|---|---|---|
| 1 | 浓度场相分数 x_w | `kernel_collide_conc:742-767` 用 rB/rT>0.5 判界面 | `x_w = φ`（直接读） | hydrate.cu:737-767 |
| 2 | CST 源项 ∇x_w | 隐含在 src_cst 判据 | ∇φ（AC 模块已算，传入） | 同上 |
| 3 | mode 4 界面判据 | rB/rT 双邻居判据 | **φ∈[0.01,0.99] 即为界面**（更精确！） | 新 is_gas_water_interface |
| 4 | VOP 分解转化 | `kernel_reinit_new_fluid` 邻域平均 | 追加 φ=1（Yang S1.2.4 规则） | hydrate_vop.cu:106 |
| 5 | VOP 形成转化 | `kernel_reinit_new_hydrate` | 固体内 φ 冻结（不参与 AC 演化） | hydrate_vop.cu:423 |
| 6 | 润湿 | G_aw/GBw 映射表 | a(mat, θ) 表面能参数表 | LBM.cu 润湿上传处 |
| 7 | 流场密度 | A/B 双组分 rho_A/rho_B | 单场 ρ=ρ_g+φ(ρ_w−ρ_g) | 全局 |
| 8 | 气-水界面张力 | PR-EOS + GAB/GBA/kappa/sigmaA | σ 直接输入（β=12σ/W, κ=3σW/2） | configs/*.yaml 新参数 |

### 3.6 相场参数表（2D 起始值）

| 参数 | 物理含义 | 推荐起始值（lattice） | 依据 |
|---|---|---|---|
| W | 界面厚度 | **3.0–4.0 lu**（tanh 剖面 10–90% 宽 ~1.1W ≈ 3.3–4.4 lu） | LB 相场惯例；Yang 未明示 |
| ω_φ | AC 弛豫率 | 由目标 M 反算（M·Δt/Δx² ~ 0.01–0.1 → ω_φ ≈ 1.6–1.9） | S37 |
| σ_latt | 表面张力（格子） | **按目标 Ca 反推，勿用物理值**（Yang γ=5×10⁻⁶ kg/s² 是折减值！） | §6-坑 3 |
| β, κ | 12σ/W, 3σW/2 | 派生 | S14 |
| ρ_w/ρ_g | 密度比 | 先 10（Yang 2024 甲烷算例）→ 稳定后 62.5–80 | S13 + Yang 实测 |
| l_sf | 润湿长度尺度 | 1.0 lu | S17 |

---

## 4. 无量纲分析框架

### 4.1 Yang 的无量纲数体系（逐字定义 + 出处）

| 数 | 定义 | 出处 | 物理意义 |
|---|---|---|---|
| Ca | $\rho_{ch}\nu U/\gamma$ | 2024 Eq. S9 | 粘性力/毛管力（气泡是否移动） |
| Pe | $UL/D$ | 2024 Eq. S9；2022 Eq.20（分母 $D_g$）；2023 Eq.30（分母 $D_w$） | 对流/扩散 |
| Da | $k_0 L/D$ | 2024 Eq. S9；2022 Eq.21 | 反应/扩散 |
| φ² (Thiele) | $k_C L/D$ | 2021a Eq.12/20 | **与 Da 同义**（2021a 称 Thiele，2022 起称 Da） |
| Da_T | $k_C L\,\Delta H\,\Delta C/(\lambda\Delta T)$ | 2022 Eq.22 | 反应热效应/热传导 |
| Pr | $\rho_{ch}\nu c_p/\lambda$ | 2024 Eq. S9 | 动量/热扩散 |
| Sc | $\nu/D$ | 2021a Eq.12 | 动量/质量扩散 |
| Le | $\lambda/(\rho c_p D)$ | 2021a Eq.12 | 热/质扩散比 |

**regime 判据**（Yang 实测）：φ²<O(1) 动力学控制、>O(10) 扩散控制；Pe~O(10⁻²) 对流开始显著；Ca<O(10⁻³) 毛管钉扎气泡。

### 4.2 形成方向的无量纲数（你的论文工具）

| 数 | 定义 | 物理解读 | regime 预期 |
|---|---|---|---|
| $\phi_f^2 = Da_f$ | $k_f L/D_w$ | 形成速率/CH₄ 扩散 | <1 动力学控制（均匀生长）；>10 扩散控制（壳层/膜状） |
| Pe | $UL/D_w$（**声明分母 D_w**） | 对流供给/扩散供给 | >10⁻² 对流供给显著 |
| $Da_{T,f}$ | $k_f L\,\Delta H\,\Delta C/(\lambda\Delta T)$ | 形成放热/热传导（符号：放热，取 \|ΔH\|） | >1 热反馈显著（形成自抑制） |
| Ca | $\rho\nu U/\gamma$ | 气泡迁移（mode 4 关键） | <10⁻³ 气泡钉扎→膜均匀包裹 |
| 过饱和度 | $S = C_m/C_{eq}(T)$ | 形成驱动力 | 成核/生长的输入 |
| 成核势垒 | $G^* = \Delta G^*/RT$ | 成核难易（mode 3/4） | 指数级影响 P_nuc |
| Sc, Pr, Le | 同上 | 物性比 | 报告用 |

### 4.3 代码嵌入点（自动计算与输出）

**位置**：`sim_utils.cu:762-798 update_hydrate_diagnostics`（每次 VTK 输出时执行）

新增诊断量（host 端计算，无需核函数）：

```cpp
// 需要的新输入：L_char（特征长度=粒径，yaml 新增 geometry.L_char，默认 20 lu）
// 运行时量：k_f = k0_form_latt * exp(-Ea_form_over_R / T_avg)
//           U_char = 流体节点 |u| 的 RMS（MX.ux, MX.uy）
//           D_latt = d_D_latt（已有 __constant__，host 侧从 P 重算）
double phi_f2   = k_f * L_char / D_latt;                 // 形成 Thiele/Da
double Pe_num   = U_char * L_char / D_latt;              // 对流/扩散
double DaT_f    = k_f * L_char * d_latent_H_latt * dCm
                  / (alpha_latt * dT_char);              // 放热/导热
double Ca_num   = rho_avg * nu_latt * U_char / sigma_latt;
```

输出到 log.txt + VTK fielddata。文件：`sim_utils.cu:762-798`（~60 行）、`sim_utils.h RuntimeParams`（+L_char）、`config.py` flatten。

### 4.4 用无量纲数组织参数扫描（形成区制图）

```
扫描矩阵（对标 Yang 2022 Fig.21 的 12+ 算例设计）：
  φ_f² ∈ {0.1, 1, 10, 100}   ×  S_w ∈ {0, 0.2, 0.4, 0.6}   ×  Pe ∈ {10⁻³, 10⁻², 10⁻¹}

实现方式：不再扫 k0_form/Ea_form（有量纲），而是
  for target_phi_f2 in [...]: k0_form = phi_f2 * D_w / L_char（反算）
  
预期区制（假设，待验证）：
  φ_f²<1:  均匀形成（动力学控制，全域同时生长）
  φ_f²>10: 扩散控制 → 壳层优先生长（ghost 邻域 Cm 枯竭）
  S_w>0.5 + 高 Pe: 气泡迁移 → 界面成核位点随气泡移动（mode 4）
```

---

## 5. 代码修改详细步骤

### Phase PF-0：独立相场模块骨架（1 天，零风险）

**原则**：新文件、新二进制，**不动现有 SC 代码一行**。

| 步骤 | 内容 | 文件 |
|---|---|---|
| 0.1 | 新建 `include/phase_field.h`：`PF_dev` 结构体（f_phi[NX·NY·5 或 9], f_ns[NX·NY·9], phi, rho, ux, uy, Fx_c, Fy_c, phi_prev）、`__constant__ d_W, d_sigma_latt, d_beta, d_kappa, d_omega_phi, d_rho_w_latt, d_rho_g_latt`、函数声明 | 新 |
| 0.2 | 新建 `src/phase_field.cu`：alloc/free/init + `step_phase_field()` 宿主函数 | 新 |
| 0.3 | `build.py` 新增目标 `mcmp_sim_phasefield`（`_PF_SOURCES = _BASE_SOURCES + phase_field.cu`） | build.py:24-25 仿 hydrate |
| 0.4 | `configs/phasefield.yaml`：W, sigma_latt, rho_w, rho_g, nu_w, nu_g, omega_phi | 新 |
| 0.5 | `main.cu` 新增 `--phase-field` 分支（#ifdef 保护） | main.cu |

### Phase PF-1：AC 方程实现与验证（2–3 天）

| 步骤 | 内容 | 验证 |
|---|---|---|
| 1.1 | `kernel_collide_phi` + `kernel_stream_phi`（路径 A：镜像 `kernel_collide_conc` 骨架；m_eq=[φ,φux,φuy,¾φ,0]，源矩动量行注入 F_φ+∂_t(φu)） | — |
| 1.2 | `kernel_compute_grad_phi`（各向同性差分）+ `kernel_compute_F_phi`（含 tanh、φ 截断 ε=1e-6） | — |
| 1.3 | 存储 `phi_u_prev`（时间导数后差） | — |
| 1.4 | **验证 1：静态平面界面**——初始化 φ=tanh 剖面，无流速，演化 10k 步 | 剖面保持 W、无漂移、Σφ 守恒（漂移<0.1%） |
| 1.5 | **验证 2：静态液滴**——圆形 φ 初始化，无外力 | 液滴稳定、剖面 tanh、无伪速度环流（max\|u\|<10⁻⁵ lu） |

### Phase PF-2：不可压 NS + 毛细力（2–3 天）

| 步骤 | 内容 | 验证 |
|---|---|---|
| 2.1 | `kernel_collide_ns` + `kernel_stream_ns`（§3.3 平衡分布；MRT 复用 LBM.h 的 M/Minv） | — |
| 2.2 | `kernel_compute_capillary_force`（S14：β,κ,∇²φ 9 点拉普拉斯）+ Guo 力项 | — |
| 2.3 | **验证 3：Laplace 定律**——半径 R∈{15,20,25,30} 液滴，测 Δp | Δp·R/σ = 1（2D 柱面），误差<5% |
| 2.4 | **验证 4：双液滴合并** | 合并后单滴、总 Σφ 守恒 |

### Phase PF-3：表面能润湿（1–2 天）

| 步骤 | 内容 | 验证 |
|---|---|---|
| 3.1 | `kernel_wetting_boundary`（§3.4：n_s → 8 邻居最近方向 → S17 → φ_s；按 d_wall_mat 取 θ） | — |
| 3.2 | **验证 5：固面液滴接触角** θ∈{30°,60°,90°,120°} | 平衡后拟合接触角误差<3°（对标 Yang 2024 Fig S6 的 θ=60° 验证） |

### Phase PF-4：hydrate 管线耦合（2–3 天）

| 步骤 | 内容 | 验证 |
|---|---|---|
| 4.1 | `kernel_collide_conc` 的 x_w 来源改为 φ（hydrate.cu:737-767 改 ~10 行；保留 src_cst 弛豫） | 单气泡 Henry 平衡与 SC 版一致 |
| 4.2 | `kernel_reinit_new_fluid` 追加 φ=1；`kernel_reinit_new_hydrate` 冻结固体区 φ（flag<-1 跳过 AC 碰撞） | — |
| 4.3 | **验证 6：单球分解**（与 SC 版同参数对比） | Vh 衰减曲线、Cm 场、T 场与 SC 版定性一致；无 NaN |
| 4.4 | **验证 7：双系统对照**（SC vs PF 的 mode 0 基线） | 论文素材："两种界面模型给出一致分解动力学" |

### Phase PF-5：形成模式 + 无量纲框架（3–5 天）

| 步骤 | 内容 | 验证 |
|---|---|---|
| 5.1 | mode 1/3 形成管线在 PF 底上跑通（复用 hydrate_vop.cu 全部） | 单球形成 + T1 守恒容差 5% |
| 5.2 | mode 4 界面成核：`is_gas_water_interface` 改用 **φ∈[0.01,0.99]**（比 rB/rT 判据更精确） | 单气泡膜成核 |
| 5.3 | 无量纲诊断嵌入（§4.3，~60 行） | log 输出 φ_f²/Pe/Da_T,f/Ca |
| 5.4 | 形成区制图首轮扫描（§4.4，GPU 并行 12 算例） | 区制图雏形 |

**工作量汇总**：新代码 ~800-1000 行（AC ~250 + NS ~350 + 润湿 ~120 + 耦合 ~100 + 诊断 ~80），改动现有代码 <50 行。验证 7 个里程碑。

---

## 6. 坑与经验

### 6.1 Yang 2024 自己趟过的坑（照单全收）

| # | 坑 | Yang 的处理 | 你的对策 |
|:-:|---|---|---|
| 1 | 表面张力物理值在 Δx=10 μm 下 Ca 无法解析 | **γ=5×10⁻⁶ kg/s² 格子折减**（比物理低 4 个量级） | 按目标 Ca=ρνU/γ 反推 σ_latt；论文中声明折减（Yang 未声明是疏漏，你可以做得更好） |
| 2 | CH 方程高阶导数 + 非局部 | 选守恒 AC | 同（已定） |
| 3 | 几何润湿方案在复杂 3D 不可实现 | 表面能方案 | 同（2D 也统一用表面能，保持代码路径单一） |
| 4 | 高 Re/大密度比稳定性 | 非正交 M 矩阵（Wang 2023）+ ω_ξ 可调 | 2D 先用现有正交 M；失稳时再引入 |
| 5 | 模块强耦合 | 单向解耦、无内迭代（"moderate rate" 前提） | 同（你现有框架已如此） |

### 6.2 实现级坑（你的代码语境）

| # | 坑 | 后果 | 对策 |
|:-:|---|---|---|
| 6 | φ=0/1 时 ln(φ/(1−φ)) 奇异 | NaN | **必须截断** φ∈[ε,1−ε]，ε=1e-6 |
| 7 | 忘记 ∂_t(φu) 时间导数项 | 界面位置 O(Δt) 系统误差 | 存 phi_u_prev，一阶后差 |
| 8 | W 取 1（想"薄界面"） | 界面数值振荡、法向失真 | W≥3；剖面验证先行（PF-1.4） |
| 9 | M 过大 | 界面非物理漂移/扩散 | M·Δt/Δx² ≤ 0.1（ω_φ≥1.6） |
| 10 | θ→0°/180° 时 S17 根号内为负 | NaN | a 计算后 clamp；θ∈[5°,175°] 外推 |
| 11 | φ 与 pointsflag 双轨不同步 | 浓度/热场看到错误相分布 | 转化核函数中**同时**更新 φ 与 flag（§3.5-4/5） |
| 12 | AC 质量泄漏（边界处） | Σφ 缓慢漂移 | 每周输出 Σφ 诊断；域边界 no-flux（∂_n φ=0） |
| 13 | D2Q5 路径 A 的界面法向各向异性 | 圆形气泡变"方" | 仅用于原型；mode 4 正式算例用路径 B（D2Q9） |
| 14 | 粘度插值方式 | 界面处 ν 突变失稳 | 用调和平均 ν=ν_wν_g/(φν_w+(1−φ)ν_g)（同 CST 的 D 调和） |

### 6.3 不要做的事

- ❌ 不要把 Yang 的 D3Q19 M 矩阵（S33 非正交）照抄到 2D——2D 用你现有正交 M
- ❌ 不要删 SC 代码——相场是**并行原型**（mcmp_sim_phasefield），主代码 SC 继续用于 mode 0–3
- ❌ 不要在 PF 原型未过 Laplace/接触角验证前接入 hydrate 管线
- ❌ 不要忽略 σ 折减声明——审稿人会算 Ca

---

## 7. 决策建议

### 7.1 诚实的成本-收益表

| 路径 | 成本 | 收益 | 时机 |
|---|:---:|---|:---:|
| **留在 SC**（主线） | 0 | mode 0–3 全部可用；Yang 2D 论文同级 | 现在 |
| **无量纲框架**（§4） | ~60 行 + 分析脚本 | **论文分析的"灵魂工具"，与 SC/PF 无关** | **立即（最高性价比）** |
| **PF 原型**（PF-0~3） | ~700 行 + 1 周 | 方法论学习 + Laplace/接触角验证 | mode 4 开发前 |
| **PF 耦合**（PF-4~5） | ~300 行 + 1 周 | mode 4 界面成核的精确界面 | 仅当 SC 伪速度污染被量化证实 |

### 7.2 推荐路径

```
立即（本周）：§4 无量纲框架嵌入（SC 代码上直接做，与相场无关）
   ↓
Phase 3.3 前置验证（code_mechanism_optimization_plan.md）：
   量化 SC 伪速度 vs 界面扩散通量
   ↓
判据 A：伪速度 << D/Δx  → 留在 SC，mode 4 用 rB/rT 判据
判据 B：伪速度 ~ D/Δx 或更大 → 启动 Phase PF-0~5，mode 4 走相场
```

**这就是"从 Yang 的经验中学到最多、踩坑最少"的方式**：他花了 3 年（2021→2024）才发现 SC 在界面传质精度上的天花板——你用他的结论做**前置量化判据**，把 3 年的弯路压缩成 1 天的验证算例。

---

## 附录 A：Yang 2024 SI 公式索引（本文件引用对照）

| 内容 | 原式 | 本文件章节 |
|---|---|---|
| 无量纲数定义 | Eq. S9 | §4.1 |
| 守恒 AC 方程 | Eq. S10 | §2.1 |
| AC-LB 演化/φ/ρ | Eq. S11–S13 | §2.2 |
| 毛细力 F_c（β,κ） | Eq. S14 | §2.3 |
| NS MRT-LB/p/u | Eq. S15–S16 | §2.3 |
| 表面能润湿 | Eq. S17 | §2.4 / §3.4 |
| CST 方程/LB/R_CST | Eq. S18–S23 | §2.5 |
| 热场 R_d/R_c | Eq. S25–S26 | §2.5 |
| VOP 更新 | Eq. S28 | §2.5 |
| 平衡浓度 | Eq. S29 | （与代码一致 ✓） |
| D3Q19 速度/矩阵 | Eq. S32–S33 | §2.2（不照抄） |
| AC 平衡矩/源矩 | Eq. S34–S35 | §3.2 路径 B 映射源 |
| Λ_φ/M | Eq. S36–S37 | §3.2 |
| NS 平衡矩/源矩 | Eq. S38–S39 | §3.3（概念来源） |
| Λ/ν | Eq. S40–S41 | §3.3 |

## 附录 B：与既有文档的关系

- `literature/SC_vs_phasefield_and_3D_migration_guide.md`：学习路线与 30 天计划（**为什么学**）；本文是**公式+代码级实现**（怎么写）
- `code_mechanism_optimization_plan.md`：总体优化 Phase 0–5；本文的 Phase PF-x 是其 Phase 3.3 判据触发后的分支
- `literature/yang_papers_pdf_deep_analysis.md` §2.5：Yang 2024 的 3D 原版细节与坑
