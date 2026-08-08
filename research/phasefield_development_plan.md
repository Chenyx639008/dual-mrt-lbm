# 相场 LBM 开发计划（2D 从简到复杂 → hydrate 集成）

> **日期**: 2026-08-02 · **更新**: 2026-08-06（v4: 阶段 3 阻塞确认, NS 壁面完成）
> **作者**: 项目组
> **位置**: 在 **dual-mrt-lbm 框架**内开发（纯流动相场模型 + 基准测试），完成验证后集成进 hydrate 求解器
> **状态**: 🔄 阶段 0/1/2 完成 + CUDA pf_ns_2d；阶段 3 阻塞于 **MRT-LB AC**（直接 FD 无法定量湿壁）
> **一句话总结**: 弃 SC 伪势、改用**相场 LB**（Yang 2024 路径）在 2D 从零搭起——单相 NS 基线 → 保守 Allen-Cahn 界面捕捉 → 两相耦合（液滴/Laplace/共存/伪势电流）→ 润湿 → 两相流动基准 → 集成 hydrate 求解器 → 兑现 mode 4 界面成核。**JAX 先行验证 → CUDA 生产落地**双轨推进，每阶段 JAX 通过后再写 CUDA。
参考文献十分重要，请你在阅读 Yang 2024 的论文时，我给你我初步的一个 Yang 2024 的论文的md文档：/home/server/software/dual-mrt-lbm/research/literature/Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag.md这个请你仔细参考。

> **进度总览 (2026-08-06)**:
> - ✅ **阶段 0** (单相 NS): Poiseuille <2% + 方腔 vs Ghia（JAX 测试通过）
> - ✅ **阶段 1** (保守 AC): 守恒 + 剖面（JAX 测试通过）
> - ✅ **阶段 2** (AC+NS 耦合): 9 项 JAX 测试 + Laplace R²=0.9996 / 共存 / 伪速度 1.1e-4
> - ✅ **CUDA pf_ns_2d**: 编译成功, JAX↔CUDA 交叉验证 phi=3e-10（机器精度）
> - ✅ **NS 无滑移壁面**: `coupled_ac_ns_step(wall=...)` 已实现（向后兼容，回归全绿）
> - 🔴 **阶段 3 定量湿壁**: 阻塞于 **MRT-LB 保守 AC（Yang S11）**——直接 FD AC 无法定量重现 S17 接触角（证据见 `phasefield_stage3_summary.md` v2）
> - 📋 **阶段 4** (两相流动) → 🔴 **阶段 5/6** (hydrate)
> - 阶段总结: `research/phasefield_stage2_summary.md` / `stage3_summary.md`
---

## 0. CUDA 二进制策略：与现有体系对齐

### 0.1 框架已有的三套 CUDA 二进制

| 二进制 | 家族 | 物理 | 编译命令 |
|--------|------|------|---------|
| `mcmp_huang_256` | scmp | Huang & Wu (2016) SCMP, CS-EOS | `uv run lbm-build --huang` |
| `mcmp_sim` | mcmp | MCMP 两相流, PR-EOS | `uv run lbm-build` |
| `mcmp_sim_hydrate` | mcmp | MCMP + 水合物热-浓度-VOP | `uv run lbm-build --hydrate` |

### 0.2 相场需要新增的 CUDA 二进制

| 二进制 | 家族 | 物理 | 编译命令 | 阶段 |
|--------|------|------|---------|:----:|
| `pf_ns_2d` | pf | 相场 D2Q9 MRT N-S + AC + 表面张力 + 润湿 | `uv run lbm-build --pf` | 0-4 |

**设计原则**：
- **单一 `pf_ns_2d` 二进制覆盖阶段 0-4**：通过 `pf_mode` 参数门控（0=单相 NS, 1=AC only, 2=AC+NS耦合, 3=+润湿, 4=+流动基准）。
- **编译期用 `-DPF_BUILD` 宏隔离**：与 `-DHUANG_256_BUILD` / `-DHYDRATE_ENABLE` 互斥，避免符号冲突。
- **params.txt 接口复用现有协议**：新增 `pf_*` 前缀键（`pf_mode`, `pf_W`, `pf_M`, `pf_sigma`, `pf_rho_g`, `pf_rho_w`, `pf_tau` 等），通过 `load_params_txt()` 读取到新增 `PFParams` 结构体。
- **源代码位置**：`lbm_mrt/solver/src/pf_ns_2d.cu` + `lbm_mrt/solver/include/pf_ns_2d.h`（新建，独立于现有 LBM.cu）。
- **JAX 先行原则**：CUDA 代码仅在 JAX 验证通过后才开始编写，JAX 输出作为 CUDA 的 golden reference。

### 0.3 二进制编译链

```bash
# Phase-field 编译（阶段 0 即可编译，后续阶段增量添加 kernel）
uv run lbm-build --pf                    # pf_ns_2d, 默认 256²
uv run lbm-build --pf --grid 128x64      # 自定义网格
uv run lbm-build --pf --grid 400         # 网格收敛测试用
```

### 0.4 CUDA 代码组织结构

```
lbm_mrt/solver/
├── include/
│   ├── LBM.h              # 现有 D2Q9/MRT 常量（pf 复用）
│   ├── sim_utils.h         # RuntimeParams + PFParams 扩展
│   ├── pf_ns_2d.h          # 🆕 相场结构体 + pf_mode 门控 + kernel 声明
│   └── ...
├── src/
│   ├── LBM.cu              # 现有 SC/MCMP 内核（不改）
│   ├── main.cu             # 入口（🆕 新增 pf dispatch 路径）
│   ├── sim_utils.cu        # 参数读写（🆕 新增 pf_* key 解析）
│   ├── pf_ns_2d.cu         # 🆕 相场 CUDA 内核（阶段 0-4 增量）
│   └── ...
└── build.py                # 🆕 新增 --pf 编译选项
```

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
┌─────────────────────────────────────────────────────────────┐
│  每阶段的执行顺序（严格遵守）                                   │
│                                                             │
│  Step 1: JAX 实现物理逻辑                                     │
│          jax_lbm/pf/phase_field.py                           │
│          ↓                                                   │
│  Step 2: JAX 基准测试                                        │
│          jax_lbm/tests/test_pf_stage{N}.py                   │
│          pytest 全绿 → 物理正确性确认                          │
│          ↓                                                   │
│  Step 3: JAX 参数探索（可选）                                  │
│          vmap 批量扫 W/σ/M → 确定最优参数区间                   │
│          ↓                                                   │
│  Step 4: CUDA 实现                                           │
│          lbm_mrt/solver/src/pf_ns_2d.cu                      │
│          以 JAX 输出为 golden reference，逐 kernel 对照         │
│          ↓                                                   │
│  Step 5: CUDA 基准测试                                       │
│          validation/phasefield/ 脚本                          │
│          JAX vs CUDA 逐点对比 → 误差 < 1e-6                    │
│          ↓                                                   │
│  Step 6: 模型注册 + CLI 集成                                  │
│          lbm_mrt/unified/models.py → PF_MODELS               │
│          uv run lbm run pf_xxx → 一键运行                      │
└─────────────────────────────────────────────────────────────┘
```

**关键原则**：
- **绝不跳过 JAX 直接写 CUDA**：JAX 中 10 分钟能发现的 bug，CUDA 中可能需要 2 天。
- **JAX 结果是金标准**：CUDA 输出的每个标量都必须与 JAX 一致到机器精度（或 1e-6 相对误差）。
- **一阶段一交付**：每个阶段的 JAX + CUDA + 测试 + 注册全部完成后，再进入下一阶段。

---

## 4. 阶段设计（从简到复杂，每阶段 JAX → CUDA）

### 阶段 0：单相基线（最基本 LBM 渗流特性）🟢

**目标**：D2Q9 MRT 不可压 N-S 正确（所有后续的地基）。

**JAX 实现**（`jax_lbm/pf/phase_field.py`）：
- D2Q9 MRT 碰撞 + 标准流步 + 周期/壁面边界。
- 无相场、无两相——纯单相流动。
- ✅ 已实现：`run_poiseuille()` + `collision_bgk` 步进。

**JAX 基准测试**（`jax_lbm/tests/test_pf_stage0.py`）：
| 测试 | 验收标准 | 状态 |
|------|---------|:----:|
| Poiseuille 速度剖面 vs 解析解 | L2 相对误差 < 2% | ✅ |
| 无滑移壁面速度 = 0 | \|u\| < 1e-6 at walls | ✅ |
| 质量守恒（周期流） | Δρ/ρ < 1e-6 | ✅ |
| 方腔驱动流 vs Ghia 基准 | 中线速度误差 < 2% | 📋 |
| 网格无关性 | 64²/128²/256² 收敛率 ≈ 二阶 | 📋 |

**CUDA 实现**（`lbm_mrt/solver/src/pf_ns_2d.cu`，`pf_mode=0`）：
- 单 kernel：`pf_collide_stream_bgk`（BGK 碰撞 + 流步在同一个 kernel 内）。
- 复用 `LBM.h` 中的 D2Q9 常量（`e[9][2]`, `w[9]`, `opp[9]`）。
- 输出 VTK：rho, ux, uy（与 Huang SCMP 相同字段名，便于后处理脚本复用）。
- JAX vs CUDA 对照：Poiseuille 出口速度剖面逐点误差 < 1e-6。

**模型注册**：`pf_ns_single_2d`（✅ 已注册）。

**编译**：
```bash
uv run lbm-build --pf                    # → pf_ns_2d
uv run lbm run pf_ns_single_2d           # 一键运行
```

---

### 阶段 1：保守 Allen-Cahn 界面捕捉（无流动）🟢

**目标**：φ 场在静止下正确演化、界面锐化、质量守恒。

**JAX 实现**（`jax_lbm/pf/phase_field.py`）：
- `conservative_ac_step()` — 散度形式离散，Lax-Wendroff 对流 + 中心差分扩散。
- `tanh_interface_profile()` — 初始双曲正切液滴。
- ✅ 已实现。

**JAX 基准测试**（`jax_lbm/tests/test_pf_stage0.py`）：
| 测试 | 验收标准 | 状态 |
|------|---------|:----:|
| ∫φdV 守恒 | 漂移 < 2% (200步) | ✅ |
| φ ∈ [0, 1] 有界性 | min ≥ −0.1, max ≤ 1.1 | ✅ |
| 液滴不漂移（u=0） | 质心移动 < 1 格点 | 📋 |
| 界面剖面 vs tanh 解析 | W 参数匹配 | 📋 |

**CUDA 实现**（`pf_ns_2d.cu`，`pf_mode=1`）：
- 新增 kernel：`pf_ac_step`（对标 JAX `conservative_ac_step`）。
- 内存布局：`phi[ny][nx]`（与 JAX 的 (ny, nx) 布局一致）。
- 对照：JAX 第 200 步 φ 场 vs CUDA 第 200 步 φ 场，逐点误差 < 1e-6。

**无模型注册**（阶段 1 是中间验证步骤，不独立注册模型）。

---

### 阶段 2：AC + N-S 耦合两相流 🟡（核心）

**目标**：耦合界面捕捉 + 流动 + 表面张力，两相平衡与基准。
> 🎯 **这是整个计划中最重要的阶段**——Laplace 定律是两相流模型的"身份证"。本阶段的四项基准测试**完全对齐 Huang SCMP 验证文档**（`HUANG_SCMP_VALIDATION_GUIDE.md`）的四大验证体系，确保相场 vs SC 的可对比性。

**JAX 实现**（`jax_lbm/pf/phase_field.py`）：
- `coupled_ac_ns_step()` — 一步耦合更新：
  1. 密度插值 ρ(φ)（§2.3）
  2. 化学势表面张力 F_c（§2.2）
  3. AC 更新（调用 `conservative_ac_step`）
  4. NS 碰撞 + 流 + Guo 力项（复用 `collision_bgk`）
- 初始：圆形液滴 φ 分布（`tanh_interface_profile` 的 2D 版）。
- 边界：全周期（阶段 2 先不做润湿）。

**JAX 基准测试**（`jax_lbm/tests/test_pf_stage2.py`）：

| # | 测试（对齐 Huang） | 相场扫掠参数 | 验收标准 | 对应 Huang 验证 |
|:--:|------|------|---------|------|
| ②a | **Laplace 定律** | R ∈ {20, 30, 40}, σ=0.01, W=3 | ΔP = σ/R, R² ≥ 0.99, intercept≈0 | §2.1 Laplace |
| ②b | **σ-W 解耦**（相场特有）| W ∈ {2, 3, 4, 5}, 固定 σ=0.01 | σ 不随 W 变化（±5%） | §2.2 σ-解耦（类比） |
| ②c | **共存密度** | W ∈ {2, 3, 4, 5}, 平界面 | φ→0/1 精确，ρ_g/ρ_w 稳定 | §2.3 共存曲线 |
| ②d | **伪势电流** | 单点 R=30, σ=0.01 | \|u\|_max < 1e-3（vs SC 的 ~1e-2） | §2.4 Spurious |

> **相场 vs SC 验证参数对照**：
>
> | 物理量 | Huang SCMP | 相场 | 说明 |
> |--------|-----------|------|------|
> | 表面张力控制 | ε ∈ {0, −0.24, −0.48, −0.72, −0.96} | σ ∈ {0.001, 0.005, 0.01, 0.05} | 相场 σ 是直接参数 |
> | 界面厚度控制 | T_reduced ∈ {0.70, 0.80, 0.90} | W ∈ {2, 3, 4, 5} | 相场 W 是直接参数 |
> | 密度比 | ρ_l/ρ_g 由 T_r + CS-EOS 决定 | ρ_w/ρ_g 直接设定 | 相场可任意设密度比 |
> | 解耦验证 | σ vs (1−6k₁) | σ vs W（应无关） | 相场天然解耦 |
> | 共存提取 | 平界面 slab, init_mode=2 | 平界面 φ profile | 平界面法 |
> | 伪势电流 | 单点 R=51, Tr=0.70 | 单点 R=30, σ=0.01 | 目标 ≪ SC 的 460× |

**CUDA 实现**（`pf_ns_2d.cu`，`pf_mode=2`）：
- 新增 kernel：`pf_coupled_ac_ns_step`（对标 JAX `coupled_ac_ns_step`）。
- 使用共享内存优化邻域访问（D2Q9 流步和 ∇φ 梯度共用同一批邻域）。
- 对照：JAX 第 5000 步的 (ρ, ux, uy, φ) 四场 vs CUDA，逐点误差 < 1e-6。

**模型注册**：`pf_ac_ns_droplet_128`（✅ 已注册）；新增变体用于不同 W/σ 扫掠。

**验证脚本**：`validation/phasefield/` 下新建四个脚本（对标 Huang `06_run_huang_validation_suite.py`）：
```
validation/phasefield/
├── laplace_law.py          # ΔP vs 1/R 扫掠 + 线性回归
├── sigma_w_decoupling.py   # σ vs W 独立性验证
├── coexistence.py           # 平界面 φ → ρ_g, ρ_w
├── spurious_current.py     # |u|_max 测量 + SC 对照
└── run_all_benchmarks.py   # 🆕 一键全验证（对标 06_run_huang_validation_suite.py）
```

---

### 阶段 3：润湿（表面能方案）🟡

**目标**：接触角几何精确控制（§2.4）。

**现状 (2026-08-06)**：
- ✅ `coupled_ac_ns_step(wall=...)` NS 无滑移壁（半程反弹，向后兼容，回归全绿）
- ✅ `surface_energy_phi_s` S17 公式（4 项测试：θ=90° 中性/亲水拉水/疏水斥水/有界）
- ⚠️ `conservative_ac_step_walls` 无 y 环绕壁面 AC stencil（机制正确：强制 φ_s=1 可铺展）
- 🔴 **定量接触角被直接 FD 保守 AC 阻塞**：S17 ghost 与反扩散/毛细力失衡 → 过冲或界面冻结，
  三种角度无法区分（详见 `phasefield_stage3_summary.md` v2）

**JAX 基准测试**（`jax_lbm/tests/test_pf_stage3.py`）：
| 测试 | 验收标准 | 状态 |
|------|---------|:----:|
| S17 公式单元测试（θ=90°/亲水/疏水/有界）| 方向正确、有界 | ✅ 4 项 |
| 接触角标定 θ ∈ {30°, 60°, 90°, 120°, 150°} | 模拟 vs 设定 < 2° | 🔴 需 MRT-LB AC |
| 90° 中性（壁面无偏好）| 液滴保持半球形 | 🔴 需 MRT-LB AC |
| 液滴铺展趋势 | 亲水铺展、疏水收缩，定性正确 | 🔴 需 MRT-LB AC |

**前置任务：MRT-LB 保守 AC（Yang S11）** — 这是定量湿壁的硬前置：
- 按 `literature/phase_field_2d_implementation_plan.md` §3.2 Path B 实现
- ⚠️ 关键：源项 R 必须保证 ΣR_vs=0（质量守恒），需按 S35 补高阶矩（纯动量源不守恒，已计算验证）
- 顺带解决直接 FD 的 W≥5 稳定性限制（MRT-LB 可在 W=3-4 稳定）
- 验证链：MRT-LB AC 单独（u=0）稳定+守恒 → 耦合 NS → 湿壁接触角

**CUDA 实现**（`pf_ns_2d.cu`，`pf_mode=3`）：
- 新增 kernel：`pf_wetting_bc`（底部壁面 ghost 层 φ 修正）。
- 对照：JAX 平衡后接触角 vs CUDA，误差 < 0.5°。

**模型注册**：新增 `pf_ac_ns_droplet_theta60` / `_theta120` 等变体。

---

### 阶段 4：两相流动基准 🟡

**目标**：动态两相流验证（从静态到流动）。

**JAX 实现 + 基准**：
| 测试 | 验收标准 | 扫掠参数 |
|------|---------|---------|
| 气泡剪切流 | 气泡形变 vs 文献 | Ca ∈ {0.1, 0.5, 1.0} |
| 通道两相 Poiseuille | 速度/相分布合理 | 密度比 ∈ {10, 100, 1000} |
| 网格无关性（系统） | ΔP/R 收敛率 ≈ 二阶 | 64²/128²/256² |
| 接触线移动 | 动态 θ 定性正确 | θ₀=60°, Ca=0.01 |

**CUDA 实现**（`pf_ns_2d.cu`，`pf_mode=4`）：
- 添加体积力驱动 + 出口边界（Zou-He 或对流）。
- 对照：JAX 速度剖面 vs CUDA，逐点误差 < 1e-6。

**验证脚本**：`validation/phasefield/dynamic_benchmarks.py`

---

### 阶段 5：集成 hydrate 求解器 🔴（跨仓库迁移）

**目标**：把验证好的相场流场搬进 `formation-phase1`，替换 SC 流核。

**实现**：
- 将 `pf_ns_2d.cu` 的 AC + N-S + 表面张力 + 润湿内核移植到 `formation-phase1/lbm_mrt/solver/`。
- 保持浓度场（D2Q5 + Kang 边界）、热场、VOP、形成管线**不动**（这些已验证）。
- 流场替换后重跑：mode 1/3 质量守恒（V1 < 0.1%）、分解 V5/V6、惰性球 0.00%。

**验收**：hydrate 回归全绿 + 守恒不变。

---

### 阶段 6：mode 4 界面成核兑现 🔴

**目标**：相场下界面伪速度可忽略 → 气-水界面成核物理成立。

**实现**：
- 复用 `formation-phase1` 已就绪的 `is_gas_water_interface` 判据 + `P0_interface` 管线。
- 在相场流场上启用 mode 4：Sw<1.0 界面膜铺展、ghost 增厚。

**验收**：界面成核仅发生在气-水界面、无伪速度污染、膜铺展物理合理。

---

## 5. 里程碑与验收汇总

| 里程碑 | 阶段 | JAX 验收 | CUDA 验收 | 预计 |
|---|---|---|---|---|
| M0 单相 NS | 0 | Poiseuille <2%, 方腔 <2% | JAX vs CUDA 逐点 < 1e-6 | 1-2 天 |
| M1 AC 界面捕捉 | 1 | φ 守恒 <2%, 剖面匹配 | JAX vs CUDA φ 场 < 1e-6 | 2 天 |
| **M2 两相平衡** | **2** | **Laplace R²≥0.99, 伪势电流≪SC, 共存精确** | **JAX vs CUDA 四场 < 1e-6** | **3-4 天** |
| M3 润湿 | 3 | θ 误差 <2° | JAX vs CUDA θ < 0.5° | 2 天 |
| M4 两相流动 | 4 | 网格收敛 ≈ 二阶, 动态基准 | JAX vs CUDA 速度剖面 < 1e-6 | 3 天 |
| M5 hydrate 集成 | 5 | — | 守恒回归全绿 | 3-5 天 |
| M6 mode 4 兑现 | 6 | — | 界面成核物理合理 | 2 天 |

### 阶段 2 详细验收矩阵（最关键的阶段）

| 验证项 | 扫掠空间 | JAX 验收标准 | CUDA 对照标准 | 产出 |
|--------|---------|-------------|-------------|------|
| ②a Laplace | R∈{20,30,40}, σ=0.01 | ΔP=σ/R, R²≥0.99 | 同 JAX | `laplace_law.pdf` |
| ②b σ-W 解耦 | W∈{2,3,4,5}, σ=0.01 | σ 不随 W 变 ±5% | 同 JAX | `sigma_w_decoupling.pdf` |
| ②c 共存密度 | W∈{2,3,4,5} 平界面 | ρ_g, ρ_w 精确 | 同 JAX | `coexistence_curve.pdf` |
| ②d 伪势电流 | R=30, σ=0.01 | \|u\|_max < 1e-3 | 同 JAX | `spurious_comparison.pdf` |

---

## 6. 仓库组织（dual-mrt-lbm 内）

```
jax_lbm/pf/                       # JAX 相场轨（阶段 0-4 算法）
  phase_field.py                  # 单相 NS + AC + 耦合（逐阶段追加函数）
  __init__.py
jax_lbm/tests/
  test_pf_stage0.py               # ✅ 阶段 0+1: Poiseuille + AC 守恒 (5 tests)
  test_pf_stage2.py               # 📋 阶段 2: Laplace/σ-W/共存/伪势电流 (4+ tests)
  test_pf_stage3.py               # 📋 阶段 3: 接触角标定 (3+ tests)
  test_pf_stage4.py               # 📋 阶段 4: 动态两相流 (4+ tests)
lbm_mrt/solver/
  include/pf_ns_2d.h              # 📋 相场结构体 + pf_mode 门控 + kernel 声明
  src/pf_ns_2d.cu                 # 📋 相场 CUDA 内核（阶段 0-4 增量添加）
  src/main.cu                     # 🔧 新增 pf dispatch 路径
  src/sim_utils.cu                # 🔧 新增 pf_* key 解析
  build.py                        # 🔧 新增 --pf 编译选项
lbm_mrt/unified/models.py         # pf 族注册（✅ pf_ns_single_2d + pf_ac_droplet_2d）
validation/phasefield/            # 📋 基准脚本与结果
  laplace_law.py                  # ΔP vs 1/R 扫掠（对标 Huang ②a）
  sigma_w_decoupling.py           # σ vs W 独立性（对标 Huang ②b）
  coexistence.py                  # φ 平界面共存（对标 Huang ②c）
  spurious_current.py             # |u|_max 测量（对标 Huang ②d）
  run_all_benchmarks.py           # 🆕 一键全验证
research/
  phasefield_development_plan.md  # 本文档
  literature/                     # Yang 论文分析 + SC vs 相场对比
  HUANG_SCMP_VALIDATION_GUIDE.md  # Huang 验证参考（对比基准）
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
| `pf_ns_2d.cu` 与现有 `LBM.cu` 符号冲突 | `-DPF_BUILD` 宏隔离，独立编译单元 |
| Huang 验证脚本无法直接复用 | 相场参数命名不同（σ vs ε, W vs T_r），需独立脚本但结构对齐 |

---

## 8. 执行顺序（AI 可自动化执行的流水线）

```
Phase 1: 阶段 0 收尾
  ├── [ ] test_pf_stage0.py 补全方腔驱动流测试
  ├── [ ] pf_ns_2d.h 头文件（结构体 + kernel 声明）
  ├── [ ] pf_ns_2d.cu 阶段 0 kernel（pf_mode=0, BGK 碰撞+流步）
  ├── [ ] sim_utils.cu pf_* key 解析
  ├── [ ] main.cu pf dispatch 路径
  ├── [ ] build.py --pf 选项
  ├── [ ] JAX vs CUDA Poiseuille 对照
  └── [ ] 验收: pytest jax_lbm/tests/test_pf_stage0.py -v 全绿

Phase 2: 阶段 1 CUDA 落地
  ├── [ ] pf_ns_2d.cu 阶段 1 kernel（pf_mode=1, AC step）
  ├── [ ] JAX vs CUDA φ 场对照
  └── [ ] 验收: pytest jax_lbm/tests/test_pf_stage0.py -v 全绿 (5 tests)

Phase 3: 阶段 2 JAX 实现 ⭐ 核心
  ├── [ ] coupled_ac_ns_step() 实现（化学势表面张力 + AC + NS）
  ├── [ ] test_pf_stage2.py ②a Laplace 测试
  ├── [ ] test_pf_stage2.py ②b σ-W 解耦测试
  ├── [ ] test_pf_stage2.py ②c 共存密度测试
  ├── [ ] test_pf_stage2.py ②d 伪势电流测试
  ├── [ ] JAX 参数探索: vmap 扫 W/σ 空间
  └── [ ] 验收: 4 项 JAX 测试全绿

Phase 4: 阶段 2 CUDA 落地
  ├── [ ] pf_ns_2d.cu 阶段 2 kernel（pf_mode=2, AC+NS 耦合）
  ├── [ ] JAX vs CUDA 四场对照（ρ, ux, uy, φ）
  ├── [ ] validation/phasefield/ 四个验证脚本
  ├── [ ] pf_ac_ns_droplet_128 模型注册 + CLI 验证
  └── [ ] 验收: uv run lbm run pf_ac_ns_droplet_128 成功

Phase 5: 阶段 3 定量润湿（⛔ 前置：MRT-LB 保守 AC）
  ├── [x] MRT-LB AC 首次尝试（2026-08-06）：质量守恒✅ + W=3/4 稳定✅，但液滴溶解（反扩散源失效）
  ├── [ ] **先获取 Liang et al. 2018 (PRE 97, 033309) 或 Yang SI 精确 D2Q9 矩/源/弛豫定义**（不要猜）
  ├── [ ] MRT-LB 保守 AC 正确实现: f* = f − M⁻¹[Λ(m−m_eq) + (I−Λ/2)R]
  ├── [ ] 源项 R 质量守恒验证（ΣR_vs=0）—— 纯动量源不守恒，需按 S35 补高阶矩
  ├── [ ] MRT-LB AC 单独基准: 液滴稳定 + ∫φ 守恒 + W=3-4 稳定
  ├── [ ] MRT-LB AC 耦合 NS（替换直接 FD）→ 重跑 Laplace/共存/伪速度回归
  ├── [ ] S17 湿壁 + 接触角标定 θ∈{30,60,90,120,150} < 2°
  ├── [ ] CUDA pf_mode=3 湿壁 kernel + JAX↔CUDA 对照
  └── [ ] 模型注册 pf_ac_ns_droplet_theta60/120

Phase 5b: 阶段 4 两相流动基准（依赖 MRT-LB AC 或可先用直接 FD @W≥5 探索）
  ├── [ ] 气泡剪切流（Ca 扫描）
  ├── [ ] 通道两相 Poiseuille（密度比扫描）
  ├── [ ] 系统网格收敛（64²/128²/256²）
  └── [ ] 接触线移动（动态 θ）

Phase 6: 阶段 5+6 hydrate 集成
  └── ...（跨仓库迁移）
```

---

## 9. 参考文献（本项目内）

- `research/literature/SC_vs_phasefield_and_3D_migration_guide.md`（复制自 hydrate 仓库）
- `research/literature/yang_papers_pdf_deep_analysis.md`（复制自 hydrate 仓库）
- `research/phase_field_2d_implementation_plan.md`（复制自 hydrate 仓库，Yang SI 逐字公式），
- `research/HUANG_SCMP_VALIDATION_GUIDE.md`（Huang SCMP 四大验证参考基准）

-/home/server/software/dual-mrt-lbm/research/literature/Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag.md这个是yang 2024的论文，里面有Yang 2024的公式和验证方法介绍，公式有乱码并不准确，准确的请参考research/phase_field_2d_implementation_plan.md。
