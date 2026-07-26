# Huang & Wu (2016) 压力张量：完整推导、代码实现与验证指南

> 基于论文 §5.2 *Determination of the pressure tensor* 及 Fig. 3-5
> 日期: 2026-05-20

---

## §1 这篇文章的压力张量方法在做什么？

### 核心问题

伪势 LBM 中，**宏观压力张量不能直接从密度场导出**——你需要一个自洽的方式来定义它。论文 §5.2 给出了两种形式：

| 形式 | 定义 | 特点 |
|------|------|------|
| **离散型** (Eq. 34) | P^discrete_αβ = ρc_s² δ_αβ + (G/2)ψ Σ_i w_F·ψ(x+e_i)·e_iα·e_iβ | LBM 中**实际生效**的力学平衡 |
| **连续型** (Eq. 40-43) | 对离散型做 Taylor 展开 → 连续极限形式 | 解析对照用，验证数值是否正确 |

### 为什么用离散型？

因为 **离散型才是 LBM 中真正满足力学平衡** 的压力张量。用连续型算出的 σ 会和离散型有偏差——这个偏差恰好就是论文要纠正的。

---

## §2 k_d = −1/12 的结果是什么？

### 论文 Eq. 39-43 的推导链

从离散压力张量出发（Eq. 34），做 Taylor 展开得到连续形式：

```
P_continuum = [ρc_s² + (G/2)ψ² + (G/12)|∇ψ|²]·I          ← 各向同性部分
            + (G/6)∇ψ∇ψ                                     ← 各向异性部分
            + (G/12)(1 + 12k_d)ψ∇²ψ·I                       ← k_d 修正项
```

关键结果（Eq. 43）：**k_d 控制连续型压力张量的 ∇²ψ 项系数**。

### k_d = −1/12 的特殊之处

当 k_d = −1/12 时：

```
(G/12)(1 + 12·(−1/12))ψ∇²ψ = (G/12)(0)ψ∇²ψ = 0
```

**∇²ψ 项恰好消失**。此时连续型压力张量简化为：

```
P_continuum(k_d=−1/12) = [ρc_s² + (G/2)ψ² + (G/12)|∇ψ|²]I + (G/6)∇ψ∇ψ
```

这个简化形的压力张量与 CS-EOS 的热力学一致性最好（Fig. 3 验证了这一点）。

### 物理直觉

| k_d 值 | 效果 |
|--------|------|
| −1/12 | **标准值**：共存密度与 Maxwell 构建一致（Fig. 3 黑线→红点重合）|
| > −1/12 | ρ_l 偏高，ρ_g 偏低（密度比增大）|
| < −1/12 | ρ_l 偏低，ρ_g 偏高（密度比减小）|

论文 Fig. 3 展示了：k_d = −1/12 时，离散型压力张量（Eq. 34）给出的共存曲线与 Maxwell 热力学构建**几乎完全重合**。这证明了 LBM 的力学平衡条件与热力学平衡一致。

---

## §3 如何用压力张量验证表面张力？

### 三种方法（按精度排序）

#### 方法 1：压力张量积分法（最精确，论文 Fig. 3 使用）

从 VTK 输出读取 `p_xx`（法向应力）和 `p_yy`（切向应力）：

```python
# 一维平界面情况
sigma = ∫ (p_n - p_t) dz          # z 垂直于界面

# 二维液滴情况（极坐标）
sigma = ∫[0→∞] (r/R)² · (p_n(r) − p_t(r)) dr
```

这个积分直接来自力学平衡条件 ∇·P = 0（Eq. 35-36）。

**你的代码中的实现**：`compute_pressure_tensor_scmp` 输出了 `p_xx`、`p_yy`、`p_xy` 到 VTK，可以直接用。

#### 方法 2：Laplace 定律（论文 Fig. 5 使用）

```python
# 静态液滴情况
σ = (p_in − p_out) × R           # δp = σ/R
```

其中 p_in、p_out 是液滴内外压力平台值。**前提**：压力张量已正确计算。

你当前的 `fit_pressure_inside_outside()` 和 `laplace_sigma()` 用的就是这个方法。

#### 方法 3：ψ 积分法（Eq. 62，解析对照用）

```python
σ = −(G/6)(1−6k₁) ∫ (dψ/dr)² dr
```

这是从离散压力张量解析推导的结果，不需要压力场——只依赖 ρ(r) → ψ(r)。

你当前的 `compute_sigma_from_rho()` 用的就是这个方法。

### 三种方法的对比与适用场景

| 方法 | 需要的数据 | 精度 | 适用场景 | 你的代码 |
|------|-----------|------|---------|---------|
| 方法 1 | VTK p_xx, p_yy | 最高 | 验证力学平衡 | `compute_pressure_tensor_scmp` 输出 ✅ |
| 方法 2 | VTK rho + 几何 | 高 | 快速 Laplace 拟合 | `laplace_sigma()` ✅ |
| 方法 3 | VTK rho + CS-EOS 参数 | 中 | 解析对照/不依赖压力场 | `compute_sigma_from_rho()` ✅ |

**推荐策略**：用方法 1（压力张量积分）作为基准，方法 2（Laplace 定律）用于快速扫描，方法 3（ψ 积分）用于解析对照。

---

## §4 当前代码是否妥善处理了压力张量？

### ✅ 做对了的地方

```
compute_pressure_tensor_scmp (LBM.cu:2323-2384):
├── 离散型压力张量 (Eq. 34): COMPLETE ✓
│   └── pd_xx, pd_yy = cs²·ρ + 0.5·G·ψ·Σ w_F·ψ_nb·eα·eβ
├── Q_m 修正 (Eq. 55-56): COMPLETE ✓
│   ├── q_xx = (k1·Fx² + k2·F²) / (G·ψ²)
│   └── q_yy = (k1·Fy² + k2·F²) / (G·ψ²)
├── 输出到 VTK: COMPLETE ✓
│   └── p_xx, p_yy, p_xy 全部写入 VTK 文件
└── k_d = −1/12 内嵌于 CS-EOS + 基线 MRT ✓
```

### ❌ 做错了/缺失的地方

| 问题 | 严重度 | 说明 |
|------|--------|------|
| k_d 硬编码 −1/12 | 🟡 | 正确但不可调——如果你想验证 k_d ≠ −1/12 的密度比效应，需要修改 |
| Q_m 修正中未含 k_d 项 | 🟡 | Eq. 55-56 的完整形式含 k_d，但 k_d=−1/12 时此项为 0，所以当前实现等价于正确 |
| `compute_sigma_from_rho` (Python) 只用 Eq. 62 | 🟡 | 没有实现方法 1（压力张量积分），但你已有 VTK p_xx/p_yy 数据，可以做 |
| 压力张量归一化因子 | 🟢 | Eq. 34 的 0.5·G 系数需确认与 MRT 矩阵归一化一致——目前似乎正确 |

### 验证收敛性：三种方法应一致

如果代码正确，对同一个液滴 case：

```
σ_method1 (压力张量积分) ≈ σ_method2 (Laplace) ≈ σ_method3 (ψ 积分)
```

三条路径独立算出的 σ 应相差 < 1%。这是你验证代码正确性的**最有说服力的测试**。

---

## §5 这个方法的学术价值与可学之处

### 方法论创新

**1. 离散 vs 连续的一致性**

传统方法（Li 等）直接用连续型压力张量公式。Huang 的方法是：
- 从**离散层面**定义压力张量（Eq. 34）
- 推导连续极限（Eq. 40-43）
- 用参数 k_d 桥接两者

**价值**：这解决了伪势 LBM 中长期存在的"热力学一致性"问题——即数值共存的 ρ_l/ρ_g 与 Maxwell 构建之间存在偏差（Fig. 3 展示了 k_d=−1/12 时两者重合）。

**2. Q_m 修正的解耦设计**

```
传统方法（Li）:
    改 σ → 同时也改了 ρ_l/ρ_g   ← 耦合

Huang 方法:
    改 k₁/k₂ → 只改 σ，ρ_l/ρ_g 不动   ← 解耦 ✓
    改 k_d   → 只改 ρ_l/ρ_g           ← 独立控制
```

**价值**：表面张力和密度比的**独立调节**是该方法的核心优势，也是你选择 Huang 方法的最重要原因。

**3. 第三阶力项展开**

Q_m 是从力的离散格式做**三阶 Taylor 展开**推导出来的（Eq. 57-59），而非经验性地添加。这保证了：
- 二阶精度（与 MRT 碰撞一致）
- 各向同性（k₂ 项）
- 不影响质量/动量守恒（slot 0/3/5 为 0）

### 可迁移的思想与技巧

1. **矩空间的物理直觉**：每个 MRT slot 对应一个物理量（e, ε, jx, qx, jy, qy, pxx, pxy）。力项在矩空间的投影 → 清晰的物理含义。

2. **离散定义 + Taylor 展开** → 解析对照：不假设连续极限成立，从离散形式出发推导，用解析形式对照验证。

3. **参数分组验证**：论文把 k₁、k₂、k_d 分成功能组，各自独立验证。你的验证套件可以借鉴这个分组思想。

---

## §6 实用操作指南

### 如何用压力张量做表面张力验证

#### 步骤 1：确保 VTK 包含压力张量

```bash
uv run lbm-build --huang  # 当前代码已输出 p_xx, p_yy, p_xy ✅
```

#### 步骤 2：跑一个液滴 case

```bash
uv run lbm-run --case-name ptest_R40 --app lbm_mrt/solver/mcmp_huang_256 \
    --config configs/huang_scmp.yaml \
    huang_R0=40.0 cs_T=0.70 epsilon_huang=-0.666667
```

#### 步骤 3：三种方法算 σ

```python
from lbm_mrt.io.vtk_reader import read_vtk_scalars
from lbm_mrt.validation.analytical import compute_sigma_from_rho, detect_interface_radius
import numpy as np

# 读 VTK
fields, nx, ny = read_vtk_scalars("outputdata_scmp/flow_final.vtk")
rho = fields["rho"]
p_xx = fields["p_xx"]
p_yy = fields["p_yy"]

# 检测界面中心
center, R = detect_interface_radius(rho, nx, ny)

# 方法 2: Laplace 定律
# (你的 laplace_sigma 已实现)

# 方法 3: ψ 积分
sigma_psi = compute_sigma_from_rho(rho, center, k1=1/12, cs_T=0.70)

# 方法 1: 压力张量积分（需实现）
def sigma_from_pressure_tensor(rho, p_xx, p_yy, center, nx, ny):
    """σ = ∫ (p_n − p_t) dr  across interface"""
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    r_vals = np.sqrt((x_idx-center[0])**2 + (y_idx-center[1])**2)
    # p_n ≈ p_xx (法向), p_t ≈ p_yy (切向) — 一维近似
    p_diff = np.abs(p_xx - p_yy)  # 只在界面区非零
    # 径向积分
    r_max = min(nx, ny) * 0.45
    dr = 0.5
    r_bins = np.arange(0, r_max, dr)
    sigma = 0.0
    for i in range(len(r_bins)-1):
        mask = (r_vals >= r_bins[i]) & (r_vals < r_bins[i+1])
        if mask.sum() > 10:
            sigma += np.mean(p_diff[mask]) * dr
    return sigma

sigma_P = sigma_from_pressure_tensor(rho, p_xx, p_yy, center, nx, ny)
print(f"σ via pressure tensor: {sigma_P:.6e}")
print(f"σ via psi integral:   {sigma_psi:.6e}")
print(f"Difference:            {abs(sigma_P-sigma_psi)/sigma_P*100:.2f}%")
```

#### 步骤 4：一致性检验

三种独立方法算出的 σ 应相差 < 1–2%。如果不一致：

| 症状 | 可能原因 |
|------|---------|
| σ_Laplace ≠ σ_psi | `compute_sigma_from_rho` 中的 k₁ 或 cs_T 错误 |
| σ_P ≠ σ_Laplace | 压力张量输出中 Q_m 修正的系数问题 |
| 方法 1 偏大/偏小 | Eq. 34 中 0.5·G 系数或 w_F 归一化 |
