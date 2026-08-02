# 相场 LBM 开发计划（2D 从简到复杂 → hydrate 集成）

> **日期**: 2026-08-02
> **作者**: 项目组
> **位置**: 在 **dual-mrt-lbm 框架**内开发（纯流动相场模型 + 基准测试），完成验证后集成进 hydrate 求解器
> **状态**: 📋 计划（待执行）
> **一句话总结**: 弃 SC 伪势、改用**相场 LB**（Yang 2024 路径）在 2D 从零搭起——单相 NS 基线 → 保守 Allen-Cahn 界面捕捉 → 两相耦合（液滴/Laplace/共存/伪势电流）→ 润湿 → 两相流动基准 → 集成 hydrate 求解器 → 兑现 mode 4 界面成核。**JAX 先行验证 + CUDA 生产轨**双轨推进。

---

## 1. 为什么换相场（背景与动机，诚实版）

### 1.1 SC 伪势的实测瓶颈（已在 hydrate 求解器量化）
- SC 气-水界面伪速度 = 扩散速度 D/Δx 的 **460-6500×**（`formation-phase1/research/phase3_mode4_interface_nucleation.md` 实测）。
- 计划预设的"降密度比/调 G"缓解不足（GAB 降到 0.06 仍 461×）；降密度比反而更糟。
- → mode 4（气-水界面成核）在 SC 上**物理不可靠**，物理闸门已关闭。

### 1.2 Yang 的路径（2024 弃 SC 换相场，本计划的模板）
Yang 弃 SC 的 5 个技术原因（见 `research/literature/SC_vs_phasefield_and_3D_migration_guide.md`）：
1. **3D 曲面伪势电流不可接受**（2D 可控到 ~1e-3，3D 复杂曲面失控）。
2. **润湿性在复杂 3D 几何难精确控制**（SC 靠 G_aw 间接标定；相场用表面能方案几何精确）。
3. **表面张力 σ 需独立于密度比控制**（相场 ρ(φ) 直接插值 + σ 独立参数）。
4. **质量守恒严格保证**（保守 Allen-Cahn 保证 ∫φdV 守恒）。
5. **工程实现更模块化**（AC 与 N-S 共享格子速度/变换矩阵；D2Q7 从 D3Q19 提取）。

### 1.3 为什么在 dual-mrt-lbm 开发（而不是 hydrate 文件夹）
| 理由 | 说明 |
|---|---|
| 纯流动基准的隔离 | 网格无关/液滴/Laplace/共存/两相流**均无水合物依赖**，放在 hydrate 文件夹会污染已验证基线（97 测试 + 近完美守恒） |
| 双轨架构 | **JAX 镜像验证轨**（算法先 JAX 验证，成本低 10×）+ CUDA 生产轨 |
| 统一模型注册 | `ModelDefinition` + `ModelRegistry`，新增 `pf` 族干净 |
| 已有对照 | SCMP（Huang & Wu 2016）可做相场 vs SC 的伪势电流对照 |
| validation/ 目录 | contact_angle / mesh_convergence / surface 已是基准测试的家 |

---

## 2. 技术基础（Yang 2024 相场公式速查）

### 2.1 守恒 Allen-Cahn（界面捕捉，D2Q9 标量演化）
$$\frac{\partial\phi}{\partial t} + \nabla\cdot(\phi\mathbf{u}) = \nabla\cdot\left[M\left(\nabla\phi - \frac{1}{W}\left[1-\tanh^2\left(\frac{1}{2}\ln\frac{\phi}{1-\phi}\right)\right]\mathbf{n}\right)\right]$$
- $\phi$：序参数（0=气, 1=液）；$M$：迁移率；$W$：界面宽度；$\mathbf{n}=\nabla\phi/|\nabla\phi|$。
- 保守性由该特殊构造保证（Yang 2024 Note；避免 CH 高阶导数问题）。

### 2.2 表面张力（化学势形式毛细力）
$$F_c = \left[4\beta\phi(\phi-1)(\phi-0.5) - \kappa\nabla^2\phi\right]\nabla\phi,\qquad \beta=\frac{12\sigma}{W},\ \kappa=\frac{3\sigma W}{2}$$

### 2.3 密度与物性插值
$$\rho(\phi) = \rho_g + \phi(\rho_w - \rho_g), \qquad \mu(\phi) = \mu_g + \phi(\mu_w - \mu_g)$$

### 2.4 表面能润湿（几何精确接触角）
$$\phi_s = \frac{a}{2}\left(1 + a^2 - \sqrt{(1+a^2)^2 - 2a\phi_f}\right) - \phi_f,\qquad a = -\frac{l_{sf}}{2}\sqrt{\frac{\kappa}{\beta}}\cos\theta$$
- $l_{sf}$：固体表面虚拟层厚度；$\theta$：接触角；对每个曲面点直接计算所需 $\phi_s$。

### 2.5 格子速度/浓度/温度（与 hydrate 集成用）
- 流场/AC：D2Q9（共享格子速度与变换矩阵）。
- 浓度/温度：D2Q5（2D）或 D3Q7=D3Q19 前 7 方向（3D，系数 1/4）。
- 参考 Yang 2024 Table S3：26 邻居查表润湿（3D）。

---

## 3. 双轨开发策略（JAX 先行 + CUDA 生产）

```
JAX 轨（jax_lbm/pf/）：算法验证、参数探索、基准生成 → 与 CUDA 轨逐阶段对照
CUDA 轨（lbm_mrt/solver/）：生产级（sm_120，RTX 5090），最终集成 hydrate
```

- **每阶段先 JAX**：写 `jax_lbm/pf/phase_field.py`（d2q9_bgk.py 同风格），跑通基准、确认物理正确。
- **再 CUDA**：按 JAX 验证过的算法写 CUDA kernel，用 JAX 结果做逐位/逐点对照。
- **验收**：每阶段有明确 benchmark + 定量容差（见 §5）。

---

## 4. 阶段设计（从简到复杂）

### 阶段 0：单相基线（最基本 LBM 渗流特性）🟢
**目标**：D2Q9 MRT 不可压 N-S 正确（这是所有后续的地基）。
**实现**：
- D2Q9 MRT 碰撞 + 标准流步 + 周期/壁面边界。
- 无相场、无两相——纯单相流动。
**Benchmark**：
| 测试 | 验收标准 |
|---|---|
| Poiseuille 流（平板/管道） | 速度剖面 vs 解析解 < 1% |
| 方腔驱动流（lid-driven cavity） | 中线速度 vs Ghia 基准（误差 < 2%） |
| 网格无关性 | 粗/中/细网格收敛率 ≈ 二阶 |
**交付**：`jax_lbm/pf/phase_field.py`（单相核）+ JAX 测试 + 可选 CUDA `pf_ns_2d` 二进制。

### 阶段 1：保守 Allen-Cahn 界面捕捉（无流动）🟢
**目标**：φ 场在静止下正确演化、界面锐化、质量守恒。
**实现**：
- D2Q9 标量演化保守 AC（§2.1），无 u 对流项（u=0）。
- 初始：圆形液滴 φ 分布（双曲正切剖面，宽度 W）。
**Benchmark**：
| 测试 | 验收标准 |
|---|---|
| 界面剖面 | tanh 解析剖面匹配（W 控制） |
| φ 守恒 | ∫φdV 漂移 < 0.1%（保守性验证） |
| 无流动静止 | 液滴不漂移、不形变（伪势电流 = 0 量级） |
**交付**：AC 核 + 测试。

### 阶段 2：AC + N-S 耦合两相流 🟡（核心）
**目标**：耦合界面捕捉 + 流动 + 表面张力，两相平衡与基准。
**实现**：
- 密度插值 ρ(φ)（§2.3）、表面张力力项（§2.2 化学势形式）、双分布（φ + N-S）。
- 关键：**伪势电流测量**（对比 SC 的 460-6500×，相场应达 ~1e-3 或更低）。
**Benchmark**：
| 测试 | 验收标准 |
|---|---|
| 静态液滴平衡 | 平衡后形状稳定、质量守恒 |
| **Laplace 压力** | ΔP = σ/R 验证（R 扫描，误差 < 5%） |
| **密度共存** | 平衡 φ→0/1，ρ_g/ρ_w 精确 |
| **伪势电流** | 界面 |u| 相对 SC 大幅下降（目标 ≪ 扩散速度量级） |
| 网格无关性 | 液滴半径/压力差收敛 |
**交付**：AC+NS 耦合核（JAX + CUDA）+ 基准脚本（`validation/phasefield/`）。

### 阶段 3：润湿（表面能方案）🟡
**目标**：接触角几何精确控制（§2.4）。
**Benchmark**：
| 测试 | 验收标准 |
|---|---|
| 接触角标定 | θ ∈ {30°, 60°, 90°, 120°, 150°}，模拟 vs 设定 < 2° |
| 液滴铺展 | 亲水/疏水动态铺展趋势正确 |
**交付**：表面能润湿边界（JAX + CUDA）。

### 阶段 4：两相流动基准 🟡
**目标**：动态两相流验证（从静态到流动）。
**Benchmark**：
| 测试 | 验收标准 |
|---|---|
| 气泡剪切流 | 气泡形变 vs 文献（Capillary 数扫描） |
| Couette/通道两相流 | 速度/相分布合理 |
| **网格无关性**（系统） | 关键量（界面位置/压差）网格收敛 |
| 接触线移动 | 动态接触角定性正确 |
**交付**：两相流动基准集（`validation/phasefield/`）。

### 阶段 5：集成 hydrate 求解器 🔴（跨仓库迁移）
**目标**：把验证好的相场流场搬进 `formation-phase1`，替换 SC 流核。
**实现**：
- 将 pf 流场（AC + N-S + 表面张力 + 润湿）移植到 `formation-phase1/lbm_mrt/solver/`。
- 保持浓度场（D2Q5 + Kang 边界）、热场、VOP、形成管线**不动**（这些已验证）。
- 流场替换后重跑：mode 1/3 质量守恒（V1 < 0.1%）、分解 V5/V6、惰性球 0.00%。
**验收**：hydrate 回归全绿 + 守恒不变。

### 阶段 6：mode 4 界面成核兑现 🔴
**目标**：相场下界面伪速度可忽略 → 气-水界面成核物理成立。
**实现**：
- 复用 `formation-phase1` 已就绪的 `is_gas_water_interface` 判据 + `P0_interface` 管线（模式 4 机制已实现，物理闸门在 SC 下 FAIL）。
- 在相场流场上启用 mode 4：Sw<1.0 界面膜铺展、ghost 增厚。
**验收**：界面成核仅发生在气-水界面、无伪速度污染、膜铺展物理合理。

---

## 5. 里程碑与验收汇总

| 里程碑 | 阶段 | 验收 | 预计 |
|---|---|---|---|
| M0 单相 NS | 0 | Poiseuille <1%、方腔 <2% | 1-2 天 |
| M1 AC 界面捕捉 | 1 | φ 守恒 <0.1%、剖面匹配 | 2 天 |
| M2 两相平衡 | 2 | Laplace <5%、共存精确、伪势电流≪SC | 3-4 天 |
| M3 润湿 | 3 | 接触角 <2° | 2 天 |
| M4 两相流动 | 4 | 网格收敛、动态基准 | 3 天 |
| M5 hydrate 集成 | 5 | 守恒回归全绿 | 3-5 天 |
| M6 mode 4 兑现 | 6 | 界面成核物理合理 | 2 天 |

---

## 6. 仓库组织（dual-mrt-lbm 内）

```
jax_lbm/pf/                  # JAX 相场轨（阶段 0-4 算法）
  phase_field.py             # 单相 NS（阶段0）+ AC（阶段1）+ 耦合（阶段2+）
  tests/                     # JAX 基准测试（Poiseuille/液滴/Laplace/共存）
lbm_mrt/unified/models.py    # pf 族注册（PF_MODELS）
validation/phasefield/       # 基准脚本与结果（Laplace/共存/网格无关/接触角）
research/phasefield_development_plan.md   # 本文档
research/literature/         # 从 hydrate 仓库复制的相场/文献文档
```

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| AC 保守性在离散下漂移 | 用保守 AC 格式 + ∫φdV 逐阶段监测（M1 验收） |
| 相场 spurious current 仍超阈值 | M2 伪势电流基准先行；必要时调 W/M/σ 或换化学势离散 |
| CUDA 实现与 JAX 不一致 | 双轨逐阶段逐点对照；JAX 结果作为 golden reference |
| 集成 hydrate 时破坏守恒 | 阶段 5 用既有 V1-V9 验证套件回归锁定 |
| 迁移成本高 | 阶段 5 前确保 JAX/CUDA 完全一致，集成是"搬运"非"重写" |

---

## 8. 参考文献（本项目内）

- `research/literature/SC_vs_phasefield_and_3D_migration_guide.md`（复制自 hydrate 仓库）
- `research/literature/yang_papers_pdf_deep_analysis.md`（复制自 hydrate 仓库）
- `research/phase_field_2d_implementation_plan.md`（复制自 hydrate 仓库，Yang SI 逐字公式）
- Yang 2021a/2021b/2022/2023/2024（详见 yang_papers 分析）
