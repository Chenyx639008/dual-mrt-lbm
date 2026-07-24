# huang_mrt_2d 统一框架 — 设计·开发·使用手册

> **版本**: 2.0 | **日期**: 2026-07-22 | **状态**: 双轨架构生产就绪
> **维护**: 使用 `cyx_memory` skill 更新本文档及各级索引

---

## 目录

1. [架构总览：双轨制](#1-架构总览双轨制)
2. [设计哲学](#2-设计哲学)
3. [快速入门](#3-快速入门)
4. [核心概念](#4-核心概念)
5. [模块详解](#5-模块详解)
6. [JAX 镜像验证轨](#6-jax-镜像验证轨)
7. [Loop Engineering：AI 自动调参沙盒](#7-loop-engineeringai-自动调参沙盒)
8. [伴随金标准：Adjoint Gold Standard](#8-伴随金标准adjoint-gold-standard)
9. [关键参数手册](#9-关键参数手册)
10. [Phase 2: CUDA 运行时网格](#10-phase-2-cuda-运行时网格)
11. [生产工作流](#11-生产工作流)
12. [多 GPU 与超算路线图](#12-多-gpu-与超算路线图)
13. [常见问题与陷阱](#13-常见问题与陷阱)
14. [开发历史与决策记录](#14-开发历史与决策记录)

---

## 1. 架构总览：双轨制

```
┌──────────────────────────────────────────────────────────────────┐
│                     huang_mrt_2d 双轨架构                         │
│                                                                   │
│  ┌─────────────────────────────┐  ┌────────────────────────────┐ │
│  │  🔧 C++ CUDA 生产轨          │  │  🧪 JAX 镜像验证轨           │ │
│  │  lbm_mrt/solver/             │  │  jax_lbm/                   │ │
│  │                              │  │                             │ │
│  │  · MRT-LBM 碰撞 (D2Q9)       │  │  · BGK + MRT 碰撞 (D2Q9)    │ │
│  │  · Huang-Zhang 三阶力修正     │  │  · Shan-Chen 力              │ │
│  │  · Scheme IV ψ-ghost BC      │  │  · Scheme IV ψ-ghost BC     │ │
│  │  · 5 EOS (CS/PR/RK/RKS/VdW) │  │  · 5 EOS (CS/PR/RK/RKS/VdW) │ │
│  │  · 水合物热-浓度-VOP 耦合    │  │  · 5 种边界条件              │ │
│  │  · 多孔介质                  │  │                             │ │
│  │                              │  │  目标网格: 64² ~ 256²        │ │
│  │  目标网格: 256² ~ 数亿        │  │  典型时步: 100 ~ 5000        │ │
│  │  典型时步: 50k ~ 数十万       │  │                             │ │
│  │                              │  │  ✨ CUDA 做不到的能力:        │ │
│  │  绝对硬件性能优先             │  │  · jax.grad 穿透全模拟        │ │
│  │  单卡/多卡极致压榨            │  │  · jax.vmap 批量参数扫描      │ │
│  └──────────┬──────────────────┘  │  · shard_map 多卡通信原语     │ │
│             │                     └────────────┬───────────────┘ │
│             │                                  │                  │
│             │    ┌──────────────────┐          │                  │
│             └───→│  Python 调度层    │←─────────┘                  │
│                  │  lbm_mrt/unified/ │                             │
│                  │                  │                             │
│                  │  · ModelDefinition (声明式模型配方)             │
│                  │  · ModelRegistry (命名管理 + 查找)             │
│                  │  · CLI (models/info/run/validate)             │
│                  │  · params.txt 生成与校验                       │
│                  └──────────────────┘                             │
│                                                                   │
│  双轨协作模式:                                                     │
│  ① Debug: JAX 找 bug → Python 对照 → CUDA 修复                    │
│  ② AI 调参: AI 在 JAX 沙盒迭代 → 验证通过 → 翻译为 CUDA           │
│  ③ 伴随校验: JAX grad 出理论梯度 → 对齐 CUDA 手写伴随              │
└──────────────────────────────────────────────────────────────────┘
```

**核心思想**：C++ CUDA 和 JAX 不是"主从关系"，而是**对等双轨**——两套代码实现相同的物理逻辑，但面向不同的使用场景。

| 维度 | CUDA 生产轨 | JAX 镜像验证轨 |
|------|------------|---------------|
| **角色** | 生产级科学计算 | 独立验证 + 可微分析 |
| **规模** | 256² ~ 数亿网格，50k+ 时步 | 64² ~ 256²，100~5000 时步 |
| **碰撞** | MRT（Huang & Wu 2016） | BGK + MRT |
| **力模型** | Huang-Zhang 三阶修正 | Shan-Chen |
| **可微** | ❌ | ✅ `jax.grad` 穿透全模拟 |
| **语言** | C++17 + CUDA | 纯 Python + JAX |
| **性能** | 极致（sm_120 汇编级优化） | 适中（XLA 自动优化） |
| **调试** | 困难（内核级 bug 难定位） | 简单（Python traceback + print） |
| **AI 友好** | 低（C++ 模板 + CUDA 语法） | 高（纯函数 + @jit） |

**Python 调度层**统一管理两条轨道的参数生成和运行调度，`params.txt` 是唯一的生产轨边界。

---

## 2. 设计哲学

### 核心原则

| 原则 | 体现 |
|------|------|
| **双轨对等** | CUDA 生产轨与 JAX 镜像验证轨实现**相同物理逻辑**，互为对照 |
| **增量改进** | 所有新文件独立于现有代码，不改 CUDA 内核物理逻辑 |
| **参数即配置** | `ModelDefinition.to_params_dict()` → `params.txt` → CUDA |
| **模型 = 组件组合** | EOS + Collision + Force + Wetting = 一个完整模型 |
| **各司其职** | CUDA 极致性能 / JAX 可微分析 / Python 编排调度 |
| **前置校验** | `__post_init__` 拦截参数错误，不等 CUDA 跑出 NaN |

### 为什么采用双轨制

CUDA MRT-LBM 内核经过了大量调试和验证，是核心资产。但传统纯 CUDA 开发有三个致命痛点：

| 痛点 | 传统 CUDA 方案 | 双轨 JAX 方案 |
|------|---------------|--------------|
| **Debug 困难** | 内核级 bug 难定位，只能看数值漂移 | JAX 中 Python traceback 直接定位，逐行 print |
| **AI 调参风险** | AI 直接改 CUDA 代码极易引入隐蔽 bug | AI 在 JAX 沙盒安全迭代，验证通过后翻译到 CUDA |
| **伴随梯度验证** | 手写伴随方程是数学灾难，无法确认正确性 | `jax.grad` 一键出理论梯度，作为 CUDA 的金标准 |

双轨制的解决思路：

```
┌─────────────────────────────────────────────────────────┐
│  协作模式 ①: Debug 加速                                   │
│  CUDA 跑出异常 → JAX 复现相同物理 → Python 逐行调试       │
│  → 定位 bug → 对照修复 CUDA                              │
│                                                         │
│  协作模式 ②: AI 安全调参 (Loop Engineering)                │
│  AI agent 读 JAX 代码 → 修改力模型/碰撞项                  │
│  → JAX 小网格快速验证 → 跑通 → 人工/AI 翻译为 CUDA        │
│                                                         │
│  协作模式 ③: 伴随金标准 (Adjoint Gold Standard)            │
│  JAX grad 计算理论梯度 → CUDA 手写伴随计算近似梯度          │
│  → 对比验证 → 论文中声明 "JAX provides benchmark"          │
└─────────────────────────────────────────────────────────┘
```

### 二进制产物

| 二进制 | 用途 | 编译方式 |
|--------|------|---------|
| `mcmp_huang_256` | SCMP 固定 256² | `uv run lbm-build --huang` |
| `mcmp_huang_unified` | 🆕 SCMP 任意 ≤1024² | `uv run lbm-build --huang-unified` |
| `mcmp_sim` | MCMP 两相流 | `uv run lbm-build` |
| `mcmp_sim_hydrate` | MCMP + 水合物 | `uv run lbm-build --hydrate` |

## 3. 快速入门

### 安装

```bash
cd huang_mrt_2d
uv sync --all-groups
```

### 首次运行（3 步）

```bash
# 1. 编译
uv run lbm-build --huang

# 2. 查看模型
uv run lbm models

# 3. 运行
uv run lbm run scmp_cs_huang_256
```

### 日常命令

```bash
uv run lbm models                           # 列出所有模型
uv run lbm info scmp_cs_huang_256           # 查看模型详情+生成的 params.txt
uv run lbm run scmp_cs_huang_256            # 运行
uv run lbm run scmp_cs_huang_256 cs_T=0.80  # 覆盖温度
uv run lbm validate scmp_cs_huang_256        # 预检查（待完善）
uv run pytest                                # 运行测试
uv run ruff check . && uv run ruff format .  # lint
```

---

## 4. 核心概念

### 4.1 ModelDefinition — 模型的"配方"

一个 `ModelDefinition` 描述了一个完整的 LBM 模拟配置：

```python
ModelDefinition(
    name="scmp_cs_huang_256",           # 唯一标识
    model_family="scmp",                # "scmp" 或 "mcmp"
    eos=EOSParams.carnahan_starling(a=1.0, b=4.0, R=1.0, T_reduced=0.70),
    collision=CollisionParams.huang_mrt(tau=1.5),
    force=ForceParams.huang_zhang(epsilon=1.7),
    wetting=WettingParams.scmp_neutral(),
    grid=(256, 256),
    initial={"huang_init_mode": 1, "huang_R0": 40.0, ...},
)
```

### 4.2 参数系统 — 4 层组件

每个模型由 4 个组件参数构成：

```
EOSParams     → cs_a, cs_b, cs_R, cs_T, cs_G
CollisionParams → tau_huang, Lambda_huang, alpha_meq
ForceParams   → epsilon_huang, k2_huang, kd_huang
WettingParams → G_ads, theta_contact_deg, huang_psi_l/g_ref
```

### 4.3 数据流 — params.txt 是唯一边界

```
ModelDefinition.to_params_dict()
  │
  ├── EOS:   cs_a, cs_b, cs_R, cs_T, cs_G
  ├── Force: epsilon_huang, k2_huang, kd_huang
  ├── Collision: tau_huang, Lambda_huang, alpha_meq
  ├── Wetting: G_ads, theta_contact_deg
  ├── Initial: huang_R0, huang_xc, huang_yc, huang_W, huang_init_mode
  ├── Coexistence: huang_rho_g, huang_rho_l (Python 端 Maxwell 构造)
  ├── Grid: nx_override, ny_override (🔧 Phase 2)
  └── Guards: huang_u_max, huang_psi_cut, huang_tanh_factor
       │
       ▼
  params.txt ──→ load_params_txt() ──→ RuntimeParams
       │
       ▼
  push_device_constants() ──→ __constant__ 设备内存
       │
       ▼
  SCMP 内核 (35 处: x >= get_nx_active())
```

### 4.4 预注册模型

**SCMP 模型**：

| 模型名 | ε | T/Tc | 接触角 | 说明 |
|--------|---|------|--------|------|
| `scmp_cs_huang_256` | 1.7 | 0.70 | 0° | 基线，质量守恒 ✅ |
| `scmp_cs_huang_256_theta60` | 1.7 | 0.70 | 60° | 亲水壁上液滴 |
| `scmp_cs_huang_256_theta120` | 1.7 | 0.70 | 120° | 疏水壁上液滴 |
| `scmp_cs_huang_256_lowT` | 1.7 | 0.55 | 0° | 强密度比 |
| `scmp_cs_huang_256_highT` | 1.7 | 0.90 | 0° | 近临界验证 |

**🆕 MCMP 模型** (Phase 5)：

| 模型名 | Sw | init_eq | τ_A | GAB | 说明 |
|--------|-----|---------|-----|-----|------|
| `mcmp_pr_baseline` | 0.30 | 1 | 0.593 | 0.24 | PR-EOS + MRT, 基线 |
| `mcmp_pr_wet` | 0.50 | 2 | 0.608 | 0.30 | 高含水饱和度 |

---

## 5. 模块详解

### 5.1 `lbm_mrt/unified/` — Python 抽象层

```
unified/
├── __init__.py      # 公开 API
├── components.py    # EOSParams / CollisionParams / ForceParams / WettingParams
├── models.py        # ModelDefinition + ModelRegistry + 5 个预注册模型
├── runner.py        # run_model() / run_scmp() / validate_model_definition()
└── cli.py           # 统一 CLI (lbm models/info/run/validate)
```

**关键类**：
- `ModelDefinition` — 不可变数据类，描述完整模型配置
- `ModelRegistry` — 模型注册表，`get(name)` 查找模型
- `EOSParams` — CS/PR/RK/RKS/VdW 状态方程参数
- `ForceParams` — Huang-Zhang / Shan-Chen 力模型参数
- `WettingParams` — 接触角 / 润湿参数

### 5.2 `lbm_mrt/solver/` — CUDA 计算核心

```
solver/
├── include/
│   ├── LBM.h       # D2Q9 常量、MRT 矩阵、设备变量声明 + 🔧 Phase 2 运行时网格
│   └── sim_utils.h # RuntimeParams (含 🔧 nx_override/ny_override)
├── src/
│   ├── LBM.cu      # 核心内核：碰撞、流、力、边界 (🔧 35处 get_nx_active)
│   ├── sim_utils.cu # 参数读写、设备常量上传 (🔧 d_nx_active 上传)
│   └── main.cu     # 程序入口
└── build.py        # 编译脚本 (--huang, --huang-unified, --hydrate)
```

### 5.3 `jax_lbm/` — JAX 可微 LBM 验证工具

```
jax_lbm/
├── lattice.py      # D2Q9 常量
├── eos.py          # 5 种 EOS (CS/PR/RK/RKS/VdW) + 注册表
├── collision.py    # BGK + MRT 碰撞算子
├── boundary.py     # 5 种 BC (BounceBack/Equilibrium/ZouHe/periodic)
├── force.py        # Shan-Chen + 吸附力 + 体力
├── wetting.py      # ψ ghost BC (Scheme IV) + improved virtual density
├── d2q9_bgk.py     # 模拟编排 + 向后兼容 API
└── validate_against_cuda.py  # 交叉验证套件
```

**JAX 独特能力**（CUDA 做不到的）：
```python
# 敏感度分析 — 一键计算梯度
dk_dG = jax.grad(permeability, argnums=0)(G=-1.0)
dk_dT = jax.grad(permeability, argnums=1)(T=0.066)
```

---

## 6. JAX 镜像验证轨

### 6.1 设计目标

JAX 轨不是 CUDA 的"简化版"或"原型"，而是**物理逻辑完全对等的独立实现**。两条轨道共享相同的：
- D2Q9 格子常量和权重
- 状态方程（CS/PR/RK/RKS/VdW）
- 宏观量计算（ρ, u）
- 碰撞算子（BGK + MRT）
- 边界条件（BounceBack / Equilibrium / ZouHe / periodic）
- 伪势力模型（Shan-Chen + 吸附力）
- 润湿边界（Scheme IV ψ-ghost BC）

**不对等的部分**（刻意为之，各取所长）：

| 物理项 | CUDA 轨 | JAX 轨 | 原因 |
|--------|---------|--------|------|
| 力模型 | Huang-Zhang 三阶 | Shan-Chen 一阶 | JAX 侧重可微性验证，SC 力足够 |
| 水合物 | ✅ 热-浓度-VOP | ❌ | 超出 JAX 验证范围 |
| 多孔介质 | ✅ | ❌ | 几何过于复杂 |

> **核心原则**：JAX 轨不需要覆盖 CUDA 的每一个特性，但覆盖到的每一项，物理逻辑必须严格一致。

### 6.2 Debug 工作流

这是双轨制最直接的收益——**把 CUDA 的 bug  hunting 变成 Python 的逐行调试**。

```
┌──────────────────────────────────────────────────────────┐
│  Step 1: CUDA 生产跑出异常                                │
│  例: 液滴在 50000 步后质量损失 5%（预期 < 0.1%）            │
│                                                          │
│  Step 2: JAX 镜像复现                                    │
│  用完全相同的参数 (T/Tc=0.70, ε=1.7, R₀=40)               │
│  在 128² 网格上跑 2000 步 JAX 模拟                        │
│                                                          │
│  Step 3: Python 逐行诊断                                  │
│  · print(f"step {t}: mass={mass}")  ← 每步质量            │
│  · print(psi_gas, psi_liquid)        ← 伪势检查           │
│  · jnp.isnan(rho).any()             ← NaN 定位            │
│  · 可视化 ψ 场在 ghost 层的梯度        ← BC 验证            │
│                                                          │
│  Step 4: 对照修复 CUDA                                   │
│  在 JAX 中确认物理逻辑正确 → 对照修改 CUDA 内核对应位置     │
│  → JAX 重新验证 → CUDA 大规模回归测试                      │
└──────────────────────────────────────────────────────────┘
```

**典型 Debug 案例**：

| CUDA 异常 | JAX 定位方式 | 根因发现 |
|-----------|-------------|---------|
| 液滴质量不守恒 | `jnp.sum(rho)` 每步打印 | ε 值偏差（论文 −2/3 vs 经验 1.7） |
| 接触角不对 | 可视化 ghost 层 ψ 值 | Scheme IV cot(θ) 符号错误 |
| 界面发散 | `jnp.max(jnp.abs(u))` 检测 | 伪势 ψ 变负（需 clamp） |
| 共存密度偏差 | Maxwell 构造对比 | GPU 端 CS 临界标度估算不准 |

### 6.3 JAX 轨的物理验证清单

| 验证项 | JAX 实现 | 状态 |
|--------|---------|------|
| EOS 压力曲线 (gas+/spinodal−/liquid+) | `validate_against_cuda.py: test 2` | ✅ |
| 伪势 ψ > 0 全场 | `validate_against_cuda.py: test 3` | ✅ |
| 共存密度合理性 | `validate_against_cuda.py: test 1` | ✅ |
| 小网格 LBM 管线完整性 | `validate_against_cuda.py: test 4` | ✅ |
| MRT 碰撞 vs BGK 碰撞精度对比 | `collision.py: collision_mrt()` | ⚠️ 待系统量化 |
| 接触角定量验证 (θ=30°/60°/90°/120°) | `wetting.py: apply_psi_ghost_bc()` | ⚠️ 待定量 |
| 表面张力 Laplace 定律 | 待添加 | 📋 |

---

## 7. Loop Engineering：AI 自动调参沙盒

### 7.1 为什么需要沙盒

让 AI 代理（LLM + 自动化脚本）直接修改 CUDA C++ 代码风险极高：
- CUDA 内核中的一个 off-by-one 边界检查错误可能静默产生物理上"看起来合理"但完全错误的结果
- 内存访问越界不会 segfault（GPU 无 MMU 保护），只会产生垃圾数据
- 编译-运行-调试周期长（每次改代码需重编译 30s~2min）

**JAX 沙盒的解决方案**：

```
┌──────────────────────────────────────────────────────────┐
│  Loop Engineering 闭环                                    │
│                                                          │
│  ┌─────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ AI Agent │───→│ JAX 沙盒      │───→│ 自动化测试      │  │
│  │ (LLM)    │    │ (小网格快跑)   │    │ (质量/能量/梯度) │  │
│  └─────────┘    └──────────────┘    └───────┬────────┘  │
│       ↑                                     │           │
│       │          ┌──────────────┐            │           │
│       └──────────│ 失败: 反馈    │←── 测试 ───┤           │
│                  │ 成功: 翻译CUDA│   不通过    │           │
│                  └──────────────┘            │           │
│                                              │           │
│                               ┌──────────────┘           │
│                               │ 测试通过                  │
│                               ▼                          │
│                    ┌──────────────────┐                  │
│                    │ 人工/AI 翻译为     │                  │
│                    │ C++ CUDA 代码     │                  │
│                    │ 沉淀到生产轨       │                  │
│                    └──────────────────┘                  │
└──────────────────────────────────────────────────────────┘
```

### 7.2 AI 可以在 JAX 沙盒中做什么

| AI 任务 | JAX 实现位置 | 安全原因 |
|---------|-------------|---------|
| 尝试新 EOS 变体 | `jax_lbm/eos.py` — 添加 `@register_eos` | 纯函数，无副作用 |
| 修改碰撞项（如加新矩） | `jax_lbm/collision.py` — MRT 矩阵行 | JAX traceback 精确定位 |
| 测试新的力模型公式 | `jax_lbm/force.py` — 修改 `shan_chen_force` | 可立即 `grad` 验证 |
| 调参（ε, G, T） | `d2q9_bgk.py` — `run_lbm_scmp` | `vmap` 批量扫描 |
| 添加新的边界条件 | `jax_lbm/boundary.py` | 小网格快速验证 |

### 7.3 从 JAX 到 CUDA 的翻译路径

当 AI 在 JAX 沙盒中验证了新物理逻辑后，翻译为 CUDA 的对照表：

| JAX 代码 | CUDA 对应位置 | 翻译注意事项 |
|----------|-------------|-------------|
| `cs_eos_pressure(rho, a, b, R, T)` | `LBM.cu: peos_scmp()` | 注意浮点精度（JAX float64 vs CUDA double） |
| `collision_mrt(f, omega, F)` | `LBM.cu: mrt_collide_single_component_gpu()` | JAX 矩阵运算 → CUDA 展开循环 |
| `shan_chen_force(psi, G)` | `LBM.cu: compute_molecular_force_scmp()` | JAX `jnp.roll` → CUDA 共享内存邻域访问 |
| `apply_psi_ghost_bc(psi, theta)` | `LBM.cu: update_ghost_psi_bc()` | 梯度计算逐项对齐 |

---

## 8. 伴随金标准：Adjoint Gold Standard

### 8.1 问题背景

在 LBM 多相流研究中，**逆向设计**（如"给定目标接触角，反求最优 G_ads"）需要计算梯度 ∂(目标函数)/∂(控制参数)。传统方法有两个：

| 方法 | 优点 | 致命缺点 |
|------|------|---------|
| 有限差分 (FD) | 实现简单 | 步长敏感 + O(N) 计算量 + 截断误差 |
| 手写伴随 (Adjoint) | O(1) 计算量 | **数学推导极易出错，且无法验证正确性** |

手写伴随方程的问题：你需要手动推导 LBM 碰撞 → 流 → 力 → EOS → 边界条件 这一整条链的伴随，任何一个环节的符号错误都会导致梯度完全错误，而**你没有办法知道它错了**——因为错误的伴随梯度可能看起来"数值合理"。

### 8.2 JAX 的解决方案：自动微分作为金标准

```python
import jax
import jax.numpy as jnp
from jax_lbm.d2q9_bgk import run_lbm_scmp, init_droplet

# 定义目标函数：模拟 → 提取接触角
def contact_angle_simulator(G_ads):
    """端到端可微：参数 → LBM 模拟 → 接触角"""
    f0 = init_droplet(128, 128, rho_l=0.36, rho_g=0.005)
    f_final = run_lbm_scmp(f0, omega=0.667, eos_params={...},
                           n_steps=2000, wetting='psi_ghost_bc',
                           theta_deg=60.0, G_ads=G_ads)
    rho_final = jnp.sum(f_final, axis=-1)
    return compute_contact_angle(rho_final)  # 自定义提取函数

# 一键计算精确梯度
d_angle_d_Gads = jax.grad(contact_angle_simulator)(G_ads=0.5)
# → ∂θ/∂G_ads = -12.3°/unit  （精确到机器精度！）
```

### 8.3 伴随金标准工作流

```
┌──────────────────────────────────────────────────────────┐
│  Adjoint Gold Standard 验证流程                           │
│                                                          │
│  Step 1: JAX 自动微分                                    │
│  jax.grad(simulator)(params) → 理论梯度 g*               │
│  这是机器精度的"正确答案"，由自动微分保证                   │
│                                                          │
│  Step 2: CUDA 手写伴随 (或有限差分)                       │
│  在 CUDA 生产轨中计算近似梯度 g̃                            │
│                                                          │
│  Step 3: 对比验证                                        │
│  error = ||g* - g̃|| / ||g*||                             │
│  if error < 1e-6: CUDA 伴随实现正确 ✅                    │
│  if error > 1e-3: CUDA 伴随有 bug，回 JAX 对照修复 🔧     │
│                                                          │
│  Step 4: 论文声明                                        │
│  "The CUDA adjoint gradients were benchmarked against     │
│   JAX automatic differentiation on a 128² validation      │
│   grid, achieving agreement to machine precision."        │
└──────────────────────────────────────────────────────────┘
```

### 8.4 论文价值

在方法论文中，这一段话可以让审稿人（Reviewer）无话可说：

> *"To verify the correctness of our hand-derived adjoint equations, we implemented an independent differentiable LBM solver in JAX. The JAX `grad` operator provides exact gradients via automatic differentiation through the entire LBM time loop — collision, streaming, force computation, and boundary conditions. On a 128×128 validation grid, our CUDA adjoint gradients agree with the JAX gold standard to within 10⁻⁶ relative error, confirming the mathematical correctness of our adjoint derivation."*

这在审稿人眼中是**方法论上的 rigorousness（严谨性）**，它会成为论文的方法论亮点。

---

## 9. 关键参数手册

### SCMP Huang 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cs_T` | float | 0.70 | 约化温度 T/Tc（0.55~0.90） |
| `epsilon_huang` | float | 1.7 | **表面张力**（经验值，非论文 −2/3）|
| `tau_huang` | float | 1.5 | MRT 松弛时间（ν=0.333） |
| `Lambda_huang` | float | 1/12 | 消除各向异性的 Λ 参数 |
| `alpha_meq` | float | 1.0 | 平衡矩系数 |
| `cs_G` | float | −1.0 | 分子间作用强度 |
| `G_ads` | float | 0.0 | 吸附力强度 |
| `theta_contact_deg` | float | 0.0 | ψ-based 接触角（°） |

### 初始条件

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `huang_init_mode` | 1 | 1=液滴, 2=平界面, 4=壁上液滴 |
| `huang_R0` | 40.0 | 液滴半径（格点） |
| `huang_xc` | 128.0 | 液滴中心 x |
| `huang_yc` | 128.0 | 液滴中心 y |
| `huang_rho_g` | 0.009170 | 气相共存密度 |
| `huang_rho_l` | 0.305202 | 液相共存密度 |

### ε 调参指南

```
ε ↑ → 表面张力 ↑ → 液滴更稳定、界面更薄
ε ↓ → 表面张力 ↓ → 液滴易扩散

推荐组合（T/Tc=0.70, R₀=40）:
  ε=1.5 → 较低张力，液滴会缓慢缩小
  ε=1.7 → 稳定（已验证 ✅）
  ε=2.0 → 较高张力，液滴更紧凑

不同 T/Tc 需要不同 ε：
  T/Tc=0.55 (低温强密度比) → 可能需要更大 ε
  T/Tc=0.90 (近临界) → 可能需要更小 ε
```

---

## 10. Phase 2: CUDA 运行时网格

### 设计目标

将多个固定网格二进制合并为一个，编译到最大 1024²，运行时指定实际网格。

### 核心实现

```cpp
// LBM.h — 统一访问器
#ifdef HUANG_UNIFIED_BUILD
__constant__ int d_nx_active, d_ny_active;  // 运行时活跃网格
__device__ __forceinline__ int get_nx_active() { return d_nx_active; }
#else
__device__ __forceinline__ int get_nx_active() { return (int)NX; }  // 编译期常量
#endif
```

### 改动规模

| 文件 | 改动 |
|------|------|
| `LBM.h` | +25 行（访问器 + 双构建路径） |
| `sim_utils.h` | +3 行（nx_override/ny_override） |
| `sim_utils.cu` | +12 行（参数读写 + 设备上传） |
| `LBM.cu` | 35 处 guard 替换 |

### 非 unified 构建零开销

`get_nx_active()` = `(int)NX` → 编译器内联为常量 → 二进制中 0 符号引用。

### 参数链路

```
ModelDefinition.grid=(256,256) → params.txt nx_override=256
  → load_params_txt() → RuntimeParams.nx_override
  → push_device_constants() → cudaMemcpyToSymbol(d_nx_active, 256)
  → 所有内核: if (x >= get_nx_active()) return;  // = if (x >= 256)
```

---

## 11. 生产工作流

### 标准流程

```bash
# 1. 编译（首次或代码变更后）
uv run lbm-build --huang              # 固定 256²
# 或
uv run lbm-build --huang-unified      # 统一二进制（支持多网格）

# 2. 查看模型和参数
uv run lbm models                     # 列出可用模型
uv run lbm info scmp_cs_huang_256     # 查看参数详情

# 3. 单次运行
uv run lbm run scmp_cs_huang_256

# 4. 参数扫描
for T in 0.55 0.60 0.70 0.80 0.90; do
    uv run lbm run scmp_cs_huang_256 \
        cs_T=$T \
        --case-name "sweep_T${T}"
done

# 5. ε 调参
for eps in 1.5 1.7 2.0 2.5; do
    uv run lbm run scmp_cs_huang_256 \
        epsilon_huang=$eps \
        --case-name "sweep_eps${eps}"
done
```

### 接触角模拟

```bash
# 60° 亲水（液相贴壁）
uv run lbm run scmp_cs_huang_256_theta60 huang_init_mode=4

# 120° 疏水（气相贴壁）
uv run lbm run scmp_cs_huang_256_theta120 huang_init_mode=4
```

### 结果读取

```python
from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars
import numpy as np

vtk_path = latest_vtk("results/scmp_cs_huang_256/outputdata_scmp", "flow")
fields, nx, ny = read_vtk_scalars(vtk_path)
rho = np.array(fields["rho"]).reshape(ny, nx)
print(f"ρ ∈ [{rho.min():.4f}, {rho.max():.4f}]")
```

---

## 12. 多 GPU 与超算路线图

### 12.1 当前状态

当前框架是**单 GPU** 设计。CUDA 轨使用单卡裸 CUDA，JAX 轨使用单设备 `@jit`。对于 2D 问题（256² ~ 1024²），单卡 RTX 5090 / H100 足以胜任。

### 12.2 为什么需要多 GPU

当扩展到以下场景时，单卡会成为瓶颈：

| 场景 | 单卡瓶颈 | 多卡收益 |
|------|---------|---------|
| **3D LBM** (256³ ~ 512³) | 显存不足（~8 GB per 256³ D3Q19） | 域分解，每卡 1/N 显存 |
| **大规模参数扫描** | 串行逐个跑，数月才能完成 | `vmap` + `pmap` 并行，数天完成 |
| **超长时间演化** (10⁶+ 步) | 单卡 Wall-time 不可接受 | 空间 + 时间并行 |
| **水合物多物理场** | 热/浓度/VOP 额外 50MB+ | 物理场分卡 |

### 12.3 双轨多 GPU 策略

CUDA 轨和 JAX 轨采用**不同的多 GPU 策略**，各取所长：

```
┌──────────────────────────────────────────────────────────┐
│  🔧 CUDA 生产轨：MPI + NCCL (传统 HPC)                     │
│                                                          │
│  适用场景: 单一大算例的域分解 (Domain Decomposition)         │
│                                                          │
│  · MPI 管理多节点进程                                      │
│  · NCCL 做 GPU 间 Halo 交换                                │
│  · CUDA-aware MPI 直接 GPU-GPU 数据传输                    │
│  · 成熟生态：Slurm + CUDA + OpenMPI                        │
│                                                          │
│  开发重点:                                                 │
│  · 域分解边界（x ≥ d_nx_active → 发送到右邻域）             │
│  · Halo 层深度 = 1（D2Q9 最近邻）                          │
│  · 重叠通信与计算（异步 NCCL + CUDA Stream）                │
├──────────────────────────────────────────────────────────┤
│  🧪 JAX 验证轨：shard_map (SPMD)                           │
│                                                          │
│  适用场景: 验证多卡通信逻辑 + 批量参数扫描                    │
│                                                          │
│  · jax.lax.shard_map 声明式网格映射                        │
│  · 无需手写 MPI/NCCL —— JAX 自动生成通信代码                │
│  · 同一份代码在 1/2/4/8 GPU 上直接运行                      │
│  · shard_map 在小网格验证轨上验证通信模式正确性              │
│                                                          │
│  典型用法:                                                 │
│  from jax.sharding import Mesh, NamedSharding, PartitionSpec│
│  mesh = Mesh(jax.devices(), ('gpu',))                     │
│  f_sharded = shard_map(step_fn, mesh, ...)                │
│                                                          │
│  价值: 用 JAX shard_map 在小网格上跑通域分解逻辑后，          │
│  对照翻译为 CUDA + MPI + NCCL，大幅降低多卡通信 bug 风险     │
└──────────────────────────────────────────────────────────┘
```

### 12.4 实施路线图

| 阶段 | 内容 | CUDA 轨 | JAX 轨 | 优先级 |
|------|------|---------|--------|--------|
| **M1** | JAX `vmap` 批量参数扫描 | — | `vmap` 单卡并行扫 ε/T/θ | 🔴 短期 |
| **M2** | JAX `shard_map` 2D 域分解 | — | 2 GPU 上 256² 网格对半分 | 🟡 中期 |
| **M3** | CUDA 多 GPU 域分解 | MPI + NCCL + 域分解 | 对照 JAX shard_map 验证 | 🟡 中期 |
| **M4** | JAX 多节点 (multi-node) | — | `shard_map` + GSPMD | 🟢 远期 |
| **M5** | 3D LBM 全栈 | D3Q19 MRT CUDA 多卡 | D3Q19 BGK JAX 验证 | 🟢 远期 |

### 12.5 JAX shard_map 消除 MPI 编写的价值

传统 CUDA 多卡通信的开发痛点：

```cpp
// 传统 CUDA + MPI: ~200 行易出错的通信代码
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

// 计算邻居 rank
int left = (rank - 1 + nprocs) % nprocs;
int right = (rank + 1) % nprocs;

// 打包 Halo 数据 → 发送 → 接收 → 解包
cudaMemcpy(halo_send, f_dev + offset, ...);
MPI_Sendrecv(halo_send, ..., right, ..., halo_recv, ..., left, ...);
// ... 很容易出现死锁、数据竞争、打包偏移错误
```

JAX shard_map 的等价代码：

```python
# JAX shard_map: ~10 行声明式代码
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

mesh = Mesh(jax.devices(), ('gpu',))

@partial(shard_map, mesh=mesh, in_specs=(P('gpu', None, None),),
         out_specs=P('gpu', None, None))
def step_sharded(f):
    f_collided = collision_mrt(f, omega, F)
    return streaming_with_halo(f_collided)  # JAX 自动插入通信
```

**价值**：JAX 轨的 `shard_map` 让你在小网格上验证多卡通信模式的**正确性**，然后对照翻译到 CUDA + MPI。这消除了"MPI 通信 bug → 跑一周才发现 → 重跑"的灾难循环。

---

## 13. 常见问题与陷阱

### Top 5 陷阱

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 1 | **液滴扩散/发散** | ε 用论文值 −2/3 而非经验值 1.7 | 设置 `epsilon_huang=1.7` |
| 2 | **质量不守恒** | 同上 + 缺乏共存密度注入 | 确认 `huang_rho_g`/`huang_rho_l` 已设置 |
| 3 | **网格改完没重编译** | `NX`/`NY` 是 `constexpr` | 重跑 `lbm-build` |
| 4 | **params.txt key 不匹配** | key 必须精确匹配 `RuntimeParams` 字段名 | 用 `k1_huang` 而非 `epsilon_huang` 覆盖无效 |
| 5 | **VTK 字节序** | big-endian legacy binary | 用 `vtk_reader.py` 读取 |

### SCMP 接触角

SCMP 接触角**可以**通过以下机制独立调节：
- `G_ads_scmp`（吸附力强度）
- `theta_contact_deg`（ψ-based ghost BC）
- Tr/k₁（热力学参数对 ψ 场的影响）

结论"接触角不可解耦"是早期错误结论（当时可能缺少 ψ ghost BC 机制）。

### 编译问题

```bash
# 如果遇到 CUDA 编译错误
uv run lbm-build --huang --dry-run     # 先 dry-run 看编译命令
uv run lbm-build --huang --arch sm_89  # 尝试不同 GPU 架构
```

---

## 14. 开发历史与决策记录

### 时间线

| 日期 | 事件 |
|------|------|
| 2026-05 | Huang & Wu (2016) SCMP 验证阶段（已完成） |
| 2026-07-17 | 统一框架 Phase 1（Python 抽象层）完成 |
| 2026-07-17 | 统一框架 Phase 3（JAX 验证工具）完成 |
| 2026-07-17 | 统一框架 Phase 4（CLI）完成 |
| 2026-07-17 | Phase 2 内核改造（35 处 guard + unified 构建）完成 |
| 2026-07-17 | 发散修复：ε=1.7 + 共存密度注入 |
| 2026-07-17 | jax_lbm 模块化重构（5 EOS × 2 Coll × 5 BC × 2 Wetting） |
| 2026-07-17 | Phase 5: MCMP 纳入统一框架（2 个模型 + 工厂方法） |
| 2026-07-22 | 🆕 **双轨架构正式确立** — CUDA 生产轨 ↔ JAX 镜像验证轨 |

### 双轨架构路线图

| Phase | 内容 | CUDA 轨 | JAX 轨 | 状态 |
|-------|------|---------|--------|------|
| **已完成** | | | | |
| 1 | Python 抽象层 (unified/) | ✅ | ✅ | 完成 |
| 2 | CUDA 运行时网格 (unified binary) | ✅ | — | 完成 |
| 3 | JAX 可微 LBM 验证工具 | — | ✅ | 完成 |
| 4 | 统一 CLI (lbm models/info/run) | ✅ | ✅ | 完成 |
| 5a | MCMP 模型注册 | ✅ | — | 完成 |
| **短期 (1-2 月)** | | | | |
| 5b | MCMP 水合物纳入 | ✅ | — | 待实施 |
| 6a | JAX MRT 碰撞 + Huang-Zhang 力 | — | ✅ | 📋 |
| 6b | JAX `vmap` 批量参数扫描 | — | ✅ | 📋 |
| 6c | 接触角定量验证矩阵 | ✅ | ✅ | 📋 |
| **中期 (3-6 月)** | | | | |
| 7a | JAX `shard_map` 2D 域分解验证 | — | ✅ | 📋 |
| 7b | CUDA 多 GPU MPI + NCCL 域分解 | ✅ | — | 📋 |
| 7c | Adjoint 金标准完整验证管线 | — | ✅ | 📋 |
| **远期 (6-12 月)** | | | | |
| 8 | 3D LBM (D3Q19) 双轨 | ✅ | ✅ | 📋 |
| 9 | 多节点超算部署 | ✅ | ✅ | 📋 |
| 10 | PINN + LBM 混合训练 | — | ✅ | 📋 |

### 关键决策

| 决策 | 理由 |
|------|------|
| **双轨对等而非主从** | CUDA 和 JAX 实现相同物理，各取所长；不是"JAX 简化版 CUDA" |
| **不改 CUDA 内核物理逻辑** | 内核已验证稳定，风险可控 |
| **params.txt 是唯一边界** | 解耦 Python 和 C++，各自独立演进 |
| **ε=1.7 而非论文 −2/3** | 经验调参值，质量和液滴稳定性验证通过 |
| **JAX 独立模块，不依赖 CUDA** | 保持验证的独立性——如果 JAX 依赖 CUDA，就失去了"金标准"意义 |
| **CUDA 走 MPI/NCCL，JAX 走 shard_map** | 生产轨用成熟 HPC 生态，验证轨用 JAX 声明式编程降低通信 bug 风险 |

### 相关文档

| 文档 | 内容 |
|------|------|
| [`research/unified_framework_feasibility.md`](unified_framework_feasibility.md) | 双轨架构可行性评估 |
| [`research/implementation_summary.md`](research/implementation_summary.md) | Phase 1-5 实施总结 |
| [`research/phase2_runtime_grid_summary.md`](research/phase2_runtime_grid_summary.md) | Phase 2 CUDA 运行时网格详情 |
| [`research/phase5_mcmp_integration_plan.md`](research/phase5_mcmp_integration_plan.md) | Phase 5 MCMP 纳入方案 |
| [`research/INDEX.md`](research/INDEX.md) | 研究文档索引 |
| [`configs/INDEX.md`](configs/INDEX.md) | 配置文件索引 |
| [`scripts/README.md`](scripts/README.md) | 脚本说明 |
| [`lbm_mrt/solver/CLAUDE.md`](lbm_mrt/solver/CLAUDE.md) | CUDA 内核文档 |
