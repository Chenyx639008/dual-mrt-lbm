# Yang 方法论深度剖析：SC→相场转变 + 2D→3D 迁移 + 自研代码路线图

> **日期**: 2026-07-28
> **目的**: 理解 Yang 从伪势 SC-LBM 切换到相场 LB 的技术原因、掌握 2D→3D 的实现路径、制定自研代码计划
> **前提**: 已读完 Yang 全部 5 篇论文和 SI

---

## 目录

1. [SC-LBM vs 相场 LB：Yang 为什么切换？](#1-sc-lbm-vs-相场-lb)
2. [Yang 的 2D→3D 迁移路径](#2-yang-的-2d3d-迁移路径)
3. [自研代码：分阶段实现计划](#3-自研代码分阶段实现计划)
4. [与你的现有代码的衔接策略](#4-与你的现有代码的衔接策略)

---

## 1. SC-LBM vs 相场 LB：Yang 为什么切换？

### 1.1 两种模型的本质区别

| 维度 | 伪势 SC-LBM (Yang 2021-2023) | 相场 LB (Yang 2024) |
|------|------------------------------|---------------------|
| **界面表示** | 隐式（通过密度场自动分离） | 显式（order parameter $\phi \in [0,1]$） |
| **界面厚度** | 由分子力强度决定（~3-5 格子） | 由 $W$ 参数显式控制 |
| **表面张力** | 通过 EOS + 分子力参数间接控制 | 直接参数 $\sigma$（界面能） |
| **控制方程** | N-S + EOS（无额外方程） | N-S + AC 方程（额外一个 PDE） |
| **密度比** | 通过选择 EOS 实现（可达 1000:1） | 通过 $\rho(\phi)$ 插值实现 |
| **伪势电流** | 有（尤其在 3D 曲面处严重） | 几乎无（热力学一致） |
| **润湿性实现** | $G_{aw}$ 分子力调接触角 | 表面能方案（几何精确） |
| **质量守恒** | 守恒（但 EOS 可能导致密度漂移） | 保守 AC 方程保证 |
| **3D 扩展性** | 难（伪势电流 + EOS 调参复杂） | 相对容易（AC 方程是标量方程） |
| **计算量** | 较小（无额外 PDE） | 较大（多一个 D3Q19 的 AC 方程） |

### 1.2 Yang 切换的 5 个技术原因

#### 原因 1：3D 中的伪势电流（Spurious Current）无法接受

SC 模型在 2D 中的伪势电流可以通过 MRT + Li 力项控制到可接受范围（~10⁻³ 量级）。
但在 **3D 曲面处**（如颗粒表面），伪势电流显著增大，导致：
- 非物理的界面扰动 → 气泡/水分布不稳定
- 虚假的对流 → 浓度场和温度场被污染
- 尤其对于微CT 真实结构中的复杂曲面，SC 模型几乎无法控制伪势电流

Yang 2024 Note S1 中的表述：
> *"SC model...the spurious current mainly occurs in the gas phase"* (Yang 2021a)
> *"In the present work, the phase field LB model is used to capture the gas-water multiphase flow pattern"* (Yang 2024)

#### 原因 2：润湿性在复杂 3D 几何中难以精确控制

SC 模型的润湿性通过 $G_{aw}$ 分子力系数调节——这是一个"间接"参数：
- 接触角与 $G_{aw}$ 的关系需要通过 Young's 方程 + 一系列模拟来标定
- 在 3D 复杂曲面（如微CT 的沙粒表面），无法为每个曲面点设置精确的 $G_{aw}$

Yang 2024 的相场方案使用**表面能方案**（Surface-Energy Scheme）：
$$\phi_s = \frac{a}{2}\left(1 + a^2 - \sqrt{\left(1+a^2\right)^2 - 2a\phi_f}\right) - \phi_f, \quad a = -\frac{l_{sf}}{2}\sqrt{\frac{\kappa}{\beta}}\cos\theta$$

这是**精确几何**的润湿实现——对每个曲面点直接计算所需的 order parameter，
而不是通过调节分子力系数间接逼近。

#### 原因 3：界面张力在 3D 中需要独立于密度比控制

SC 模型的一个固有问题：表面张力 $\sigma$ 和密度比都依赖于 EOS 的选择。
你不能独立调整 $\sigma$ 而保持密度比不变。在 3D 中，毛细力是关键物理——
你需要精确控制 $\sigma$ 来得到正确的 $Ca$ 数。

相场模型解耦了这两个参数：
- 密度比：$\rho(\phi) = \rho_g + \phi(\rho_w - \rho_g)$（直接插值）
- 表面张力：$\sigma$ 作为独立参数输入

#### 原因 4：质量守恒的严格保证

SC 模型的 EOS（如 C-S EOS）在气-水界面处可能出现密度振荡，导致局部质量不守恒。
Yang 2024 使用的**保守 Allen-Cahn 方程**：
$$\frac{\partial\phi}{\partial t} + \nabla\cdot(\phi\mathbf{u}) = \nabla\cdot\left[M\left(\nabla\phi - \frac{1}{W}\left[1-\tanh^2\left(\frac{1}{2}\ln\frac{\phi}{1-\phi}\right)\right]\mathbf{n}\right)\right]$$

通过特殊构造保证了 $\int\phi dV$ 的守恒，避免了 CH 方程的高阶导数问题。

#### 原因 5：数值实现的工程考虑

Yang 2024 是一个 **3D MPI 并行**代码（C++，in-house developed）。相场模型有以下工程优势：
- AC 方程的 D3Q19 LBM 实现与 N-S 方程的 D3Q19 LBM 实现**共享同一套格子速度**和变换矩阵
- D3Q7 CST/DDF 是从 D3Q19 前 7 个方向提取的——进一步减少代码量
- 表面能方案的润湿处理在代码层面比 SC 模型更模块化

### 1.3 我们的判断：你应该继续用 SC-LBM（现阶段）

| 方面 | SC-LBM (2D) | 相场 LB (2D) | 判断 |
|------|:----------:|:----------:|:---:|
| 你的代码存量 | ✅ 已有 | ❌ 需从零写 | **保持 SC-LBM** |
| 2D 伪势电流 | 可控 | 可忽略 | SC-LBM 足够好 |
| 你的 GPU 优势 | ✅ MRT SC 已优化 | ❌ 需重写 CUDA kernel | **保持 SC-LBM** |
| 形成方向创新 | ✅ 已实现 | ❌ 需重新实现 | **保持 SC-LBM** |
| 论文所需精度 | 够用 | 更好 | SC 够用 |
| 未来 3D 扩展 | 困难 | 更可行 | **3D 时切换到相场** |

**策略**：2D 阶段保持 SC-LBM（你的代码已经成熟），3D 阶段再切换到相场（跟随 Yang 2024 的路径）。

---

## 2. Yang 的 2D→3D 迁移路径

### 2.1 迁移全景

```
            2D SC-LBM 体系                    3D 相场 LB 体系
           (Yang 2021-2023)                  (Yang 2024)
  ┌──────────────────────────┐      ┌──────────────────────────┐
  │ D2Q9 MRT SC    (流场)    │  →   │ D3Q19 MRT Phase-field    │
  │ D2Q5 CST/BGK   (浓度)    │  →   │ D3Q7 MRT CST             │
  │ D2Q5 DDF       (温度)    │  →   │ D3Q7 MRT DDF             │
  │ VOP            (固相)    │  →   │ VOP            (不变)     │
  │ Kang BC        (反应)    │  →   │ Kang BC        (不变)     │
  │ Kim-Bishnoi    (动力学)  │  →   │ Kim-Bishnoi    (不变)     │
  │ C++ 单核       (计算)    │  →   │ C++ MPI        (并行)     │
  └──────────────────────────┘      └──────────────────────────┘
```

### 2.2 逐模块的 2D→3D 变化细节

#### 模块 1：流场（最大的变化）

**2D (Yang 2021a)**:
```
D2Q9 MRT SC-LBM
- 格子速度: 9 个方向 (e_x, e_y)
- 变换矩阵: M[9×9]
- EOS: Carnahan-Starling
- 力项: Li et al. 格式 (2012)
- 密度比: 1000:1 (通过 C-S EOS)
```

**3D (Yang 2024)**:
```
D3Q19 MRT Phase-field LB
- 格子速度: 19 个方向 (e_x, e_y, e_z)
- 变换矩阵: M[19×19] (Note S5, Eq.S33)
- 两个 LBM: N-S LB + AC LB (共享 D3Q19)
- N-S 方程: 不可压缩格式 (He et al. 1999)
- AC 方程: 保守 Allen-Cahn (Liang et al. 2023)
- 密度: ρ(φ) = ρ_g + φ(ρ_w - ρ_g) 线性插值
- 表面张力: σ = 12β/W, κ = 3σW/2 (直接参数)
- 毛细力: F_c = [4βφ(φ-1)(φ-0.5) - κ∇²φ]∇φ
```

**关键代码量估算**：
- D3Q19 MRT 核函数：~200 行
- AC 方程核函数：~150 行
- 两者共享 M 矩阵和 e 数组
- 总流场代码量：2D SC-LBM 的 ~2.5 倍

#### 模块 2：浓度场

**2D**: D2Q5 CST (5 方向) → **3D**: D3Q7 CST (7 方向)

| 量 | 2D | 3D |
|----|:--:|:--:|
| 格子速度数 | 5 | 7 (=D3Q19 前 7 个) |
| 变换矩阵 | M5[5×5] | N[7×7] |
| CST 源项 | 2D 梯度 | 3D 梯度 |
| Kang BC | 2D 邻居搜索 | 3D 邻居搜索 |

D3Q7 变换矩阵 N[7×7]（Yang 2024 Note S5 Eq.S42）:
```
N = [1  1  1  1  1  1  1
     0  1 -1  0  0  0  0
     0  0  0  1 -1  0  0
     0  0  0  0  0  1 -1
     6 -1 -1 -1 -1 -1 -1
     0  2  2 -1 -1 -1 -1
     0  0  0  1  1 -1 -1]
```

#### 模块 3：温度场

与浓度场完全相同：**D2Q5 DDF → D3Q7 DDF**

唯一差异：温度场有共轭源项和反应热源项（浓度场有 CST 源项和 Kang 反应边界）。

#### 模块 4：固相更新

**VOP 在 2D 和 3D 中相同**！这是最简单的迁移。
$$V_h(t+\Delta t) = V_h(t) - \Pi \cdot n_h \cdot A_s \cdot V_M$$

唯一的区别是 $A_s$ 的计算——3D 中需要 3D 的表面积。

#### 模块 5：并行计算

Yang 2024 从 C++ 单核 → C++ MPI。

**你的优势**：你已经有 CUDA GPU 代码。GPU 在 3D 中的优势比 2D 更大：
- 2D: 300×300 = 9 万节点 → GPU 过饱和
- 3D: 400×400×400 = 6400 万节点 → GPU 充分利用

### 2.3 验证链条（2D→3D 的关键保障）

Yang 的 3D 模型不是从零建立的——他先在 2D 中完成了所有的物理验证，然后逐模块迁移到 3D：

```
2D 验证（Yang 2021a/2021b/2022）
  ├── 多相反应输运 vs 解析解 (Eq.S50, 2021b) ✅
  ├── 共轭传热 vs 解析解 (Eq.S51, 2021a) ✅
  ├── 微CT 实验对比 (2021a Fig.16) ✅
  └── 确认 φ²=21.6 为扩散控制 ✅
        ↓
     所有物理机制在 2D 中已充分验证
        ↓
3D 验证（Yang 2024）
  ├── 多相反应输运 vs 解析解 (Note S5 Fig.S7) ✅ (误差 2.9%)
  ├── 共轭传热 vs 解析解 (Note S5 Fig.S8) ✅ (误差 0.8%)
  ├── 接触角验证 (Fig.S6, θ=60°) ✅
  ├── 网格收敛 (Fig.S9, Δx=20/10/5μm) ✅
  └── 微CT 实验对比 (Fig.2, 1.6% 误差) ✅
```

**你应该遵循相同的逻辑**：2D 验证完整后再做 3D。

---

## 3. 自研代码：分阶段实现计划

### 3.1 总体策略：2D SC-LBM 先行，3D 相场 LB 后续

```
Phase A (当前 - 4 周): 完善 2D SC-LBM（你现有代码）
  ├── T1 质量守恒修复
  ├── 修正参数体系
  ├── 基准验证 BM0-BM5
  └── 形成动力学机制分析 + 区制图
        ↓ "2D 机制全部搞清楚"
Phase B (4-8 周): 搭建 2D 相场 LB 原型（学习为目的）
  ├── 2D 保守 AC 方程 LBM（D2Q9）
  ├── 2D N-S 不可压缩 LBM（D2Q9）
  ├── 表面能润湿方案
  └── 验证：接触角 + Laplace 定律 + 单球形成
        ↓ "相场方法 2D 掌握"
Phase C (8-16 周): 3D 相场 LB + GPU（你的核心贡献）
  ├── D3Q19 相场 LB（流场+AC方程）
  ├── D3Q7 CST/DDF（浓度+温度）
  ├── 3D VOP + Ghost 重建
  ├── CUDA 3D kernel 优化
  └── 3D 形成/分解全耦合模拟
```

### 3.2 Phase A 详细计划（2D SC-LBM 完善）

**这是你现在最紧急的工作**。在这阶段，你不需要修改流场模型。

| 周 | 任务 | 代码文件 |
|----|------|---------|
| 1 | T1 质量守恒 + Vm 修正 | `hydrate_vop.cu`, `hydrate.cu` |
| 2 | BM0-BM5 验证通过 | `03_validate_benchmarks.py` |
| 3 | 无量纲分析嵌入（φ², Pe 自动计算） | `sim_utils.cu` 输出诊断 |
| 4 | 形成机制分析 + 区制图 | 新分析脚本 |

### 3.3 Phase B 详细计划（2D 相场 LB 原型）

**这是一个学习项目**——目的是理解相场方法，不是替代你的 SC-LBM。

#### 3.3.1 子模块 1：保守 AC 方程 LBM（2D, D2Q9）

**理论知识**（参考 Yang 2024 Note S1.2.1）：

保守 Allen-Cahn 方程：
$$\frac{\partial\phi}{\partial t} + \nabla\cdot(\phi\mathbf{u}) = \nabla\cdot\left[M\left(\nabla\phi - \frac{1}{W}\left[1-\tanh^2\left(\frac{1}{2}\ln\frac{\phi}{1-\phi}\right)\right]\mathbf{n}\right)\right]$$

D2Q9 MRT 实现（退化自 Yang 2024 的 D3Q19 版本）：

```
变换矩阵 M2D[9×9]: (D2Q9 标准 MRT 矩阵，与你现有的相同)
平衡矩 m_eq: [φ, φux, φuy, φ, 0, 0, 0, φ, 0]^T
源项 R_φ: 与 AC 方程的右端项对应
Λ_φ: diag(1, ω_φ, ω_φ, 1, 1, 1, 1, 1, 1)
```

**实现步骤**：
1. 从你现有的 `LBM.h` 和 `hydrate.h` 中借鉴 MRT 核函数结构
2. 写 `kernel_ac_collide` 和 `kernel_ac_stream`
3. 测试：单液滴在静止流场中的平衡形状
4. 验证：接触角 θ=30°, 60°, 90° 三种情况

**预计代码量**：~200 行 CUDA

#### 3.3.2 子模块 2：不可压缩 N-S LBM（2D, D2Q9）

**理论知识**（He et al. 1999 格式）：

不可压缩 N-S 方程的 LBM 与压缩版本的主要区别在于平衡分布函数的构造。

**实现步骤**：
1. 写 `kernel_ns_collide` (不可压缩格式)
2. 写毛细力计算 `kernel_capillary_force`：$F_c = [4\beta\phi(\phi-1)(\phi-0.5) - \kappa\nabla^2\phi]\nabla\phi$
3. 测试：静止液滴的 Laplace 定律（$p_{in} - p_{out} = \sigma/R$）

**预计代码量**：~250 行 CUDA

#### 3.3.3 子模块 3：表面能润湿方案（2D）

**理论知识**（Yang 2024 Eq.S17）：

$$\phi_s = \frac{a}{2}\left(1 + a^2 - \sqrt{(1+a^2)^2 - 2a\phi_f}\right) - \phi_f, \quad a = -\frac{l_{sf}}{2}\sqrt{\frac{\kappa}{\beta}}\cos\theta$$

其中 $\phi_f$ 是沿表面法向量方向的相邻流体节点的 order parameter。

**实现步骤**：
1. 对每个固体邻居节点，计算表面法向量 $\mathbf{n}_s = -\nabla s / |\nabla s|$
2. 找到 $\mathbf{n}_s$ 方向上的相邻流体节点
3. 读取该流体节点的 $\phi_f$
4. 计算固体的 $\phi_s$，设置 $g_{in}$ 以维持 $\phi_s$

**预计代码量**：~100 行 CUDA

### 3.4 Phase C 详细计划（3D 相场 LB + GPU）

**这是你博士论文可能不需要 但长期来看最有价值的工作。**

#### 3.4.1 关键技术挑战

| 挑战 | 3D 特定难度 | 解决方法 |
|------|-----------|---------|
| D3Q19 M 矩阵 | 19×19 矩阵 | Yang 2024 Eq.S33 直接复制 |
| D3Q7 N 矩阵 | 7×7 变换 | Yang 2024 Eq.S42 |
| 3D 邻居搜索 | 19 个方向 | 比 D2Q9 复杂 2× |
| 3D 表面法向量 | 3D 梯度 | 有限差分 |
| GPU 显存 | 400³=64M 节点 | 需要 fp32 或半精度 |
| 3D VTK 输出 | 文件大 | Legacy Binary 格式 |

#### 3.4.2 模块优先级

```
最高优先级（必须自己写）:
  ✅ D3Q19 MRT 相场 LB（核函数 + 流步）
  ✅ D3Q7 MRT CST/DDF
  ✅ 3D VOP + Ghost 重建
  ✅ 3D Kang BC

中优先级（可从 2D 迁移）:
  ✅ Kim-Bishnoi 动力学（与 2D 相同逻辑）
  ✅ 相平衡（与 2D 相同公式）
  ✅ VTK 输出（Legacy Binary）

低优先级（可用库）:
  - 3D 可视化（ParaView）
  - 几何读取（扩展你的 .plt reader 到 3D）
```

---

## 4. 与你的现有代码的衔接策略

### 4.1 你的代码体系架构

```
lbm_mrt/solver/
├── include/
│   ├── LBM.h          ← D2Q9 常量、结构体、MRT矩阵
│   ├── hydrate.h      ← D2Q5 常量、Therm_dev/Conc_dev/VOP_dev
│   ├── sim_utils.h    ← RuntimeParams, HydrateHost
│   └── steady_monitor.cuh
└── src/
    ├── LBM.cu          ← D2Q9 SC-LBM 核函数
    ├── hydrate.cu      ← 热场 + 浓度场 + 反应边界
    ├── hydrate_vop.cu  ← VOP 固相更新 + 形成管线
    ├── sim_utils.cu    ← 运行循环 + 几何 + VTK
    └── main.cu         ← 程序入口

你的代码已经很好的模块化——这是扩展的优势。
```

### 4.2 添加相场模块的最小侵入方案

```
新增文件（不修改现有 SC-LBM 代码）:

lbm_mrt/solver/
├── include/
│   └── phase_field.h     ← 相场 AC 方程常量、结构体、D2Q9/D3Q19 扩展
└── src/
    └── phase_field.cu    ← AC 核函数 + 不可压缩 N-S + 表面能润湿

编译模型（新增编译目标）:
  $ uv run lbm-build --phase-field   ← 产出 mcmp_sim_phasefield
```

### 4.3 模块复用矩阵

| 模块 | SC-LBM (现有) | 相场 LB (新建) | 可复用？ |
|------|:----------:|:----------:|:---:|
| 热场 | D2Q5 MRT-DDF | D2Q5/D3Q7 MRT-DDF | ✅ 完全复用 |
| 浓度场 | D2Q5 MRT-CST | D2Q5/D3Q7 MRT-CST | ✅ 完全复用 |
| VOP | VOP 双向 | VOP 双向 | ✅ 完全复用 |
| Ghost 重建 | 5 步序列 | 同 | ✅ 完全复用 |
| 反应动力学 | Kim-Bishnoi | Kim-Bishnoi | ✅ 完全复用 |
| Kang BC | 邻居搜索 | 邻居搜索 | 🟡 邻居方向数不同 |
| VTK 输出 | Legacy Binary | Legacy Binary | ✅ 完全复用 |
| 参数系统 | RuntimeParams | RuntimeParams | ✅ 完全复用 |
| GPU 内存管理 | cudaMalloc | cudaMalloc | ✅ 完全复用 |
| 流场 | D2Q9 SC | D2Q9/D3Q19 PF | ❌ 不可复用 |

**关键发现**：你 **80% 的代码**可以复用！唯一需要重写的是流场模块。

---

## 5. 学习资源与顺序

### 5.1 推荐学习顺序（30 天计划）

```
Day 1-5:  理论基础
  ├── Allen-Cahn 方程推导（Yang 2024 Note S1.2.1, Refs 5-15）
  ├── 相场法 LBM 实现（Yang 2024 Note S5.1）
  └── 保守 AC vs 非保守 AC vs CH 方程的区别

Day 6-10: 2D 原型
  ├── 写 D2Q9 AC-LBM（对标 Yang 2024 的 D3Q19 → 退化到 2D）
  ├── 写 D2Q9 不可压缩 N-S LBM
  └── 验证：Laplace 定律 + 接触角

Day 11-15: 表面能润湿 + 基准
  ├── 实现表面能润湿方案（Yang 2024 §S5.2）
  ├── 验证：不同接触角 (30°, 60°, 90°, 120°)
  └── 验证：多相反应输运基准 (Yang 2024 Fig.S7)

Day 16-20: 集成热/浓度/VOP
  ├── 将你现有的 D2Q5 热场/浓度场接入相场流场
  ├── VOP + Ghost 重建保持不变
  └── 验证：单球分解（对比 SC-LBM 结果）

Day 21-25: 3D 原型
  ├── D3Q19 M 矩阵 + 核函数
  ├── D3Q7 N 矩阵 + 核函数
  └── 验证：3D 液滴 Laplace 定律

Day 26-30: 3D 全耦合
  ├── 3D 多孔介质几何
  ├── 3D 分解/形成模拟
  └── 性能 profiling + CUDA 优化
```

### 5.2 关键参考文献（按学习顺序）

| 顺序 | 文献 | 学什么 |
|:---:|------|--------|
| 1 | Yang 2024 Note S1.2.1 | 相场 LB 的控制方程和假设 |
| 2 | Yang 2024 Note S5.1 (Eq.S10-S41) | D3Q19 相场 LB 的完整数值细节 |
| 3 | Yang 2024 Note S5.2 (Eq.S17, Fig.S5-S6) | 表面能润湿方案 |
| 4 | Liang et al. (2023) PRE | 保守 AC 方程的 LBM 实现 |
| 5 | He et al. (1999) JCP | 不可压缩 N-S LBM 格式 |
| 6 | Wang et al. (2023) IJMF | D3Q19 非正交 MRT 相场模型 |

---

## 6. 总结：你现在应该做什么？

### 立即（本周）

```
✅ 完成 2D SC-LBM 的 T1（质量守恒）+ 参数修正
   这是你博士论文的必需品

🟡 不要现在就开始写相场代码
   先完成 2D 分析，否则精力分散
```

### 短期（下月）

```
🟡 Phase B: 写 2D 相场 LB 原型（学习目的）
   每周投入 1-2 天，作为副线推进

🟡 目标是理解相场方法，不是替代 SC-LBM
```

### 中期（3-6 月）

```
🟢 Phase C: 3D 相场 LB + GPU
   如果你的博士需要 3D 结果，这是必经之路
   如果只需要 2D 结果，Phase C 可以放到博士后
```

### 关键判断

**你的博士论文**只需要 2D 结果（加上充分的验证和分析），**不需要 3D**。
Yang 2024 已经明确量化了 2D vs 3D 的差异（>50% 渗透率误差），你可以在论文中引用这个结论来说明 2D 的局限性，同时声明 GPU 使 3D 扩展可行。

**3D 相场 LB + GPU 是你毕业后可以做的**，这是你相比于 Yang（用 CPU-MPI）的最大技术优势。
