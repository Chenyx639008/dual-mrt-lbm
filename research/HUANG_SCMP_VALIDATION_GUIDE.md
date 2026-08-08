# Huang & Wu (2016) SCMP 验证 — 完整操作手册

> **二进制**: `mcmp_huang_256` (256×256, D2Q9 MRT)
> **EOS**: Carnahan-Starling (a=1.0, b=4.0, R=1.0)
> **验证产出**: `results/validation/`（原 `results/20260522/`）
> **最后更新**: 2026-07-25

---

## 目录

1. [快速开始](#1-快速开始)
2. [四大验证详解](#2-四大验证详解)
3. [单 Case 运行](#3-单-case-运行)
4. [参数选择指南](#4-参数选择指南)
5. [⚠️ 液滴不稳定的排查与解决](#️-液滴不稳定的排查与解决)
6. [后处理与分析](#6-后处理与分析)
7. [已知 Bug 与修复历史](#7-已知-bug-与修复历史)
8. [参数流：YAML → CSV → params.txt → GPU](#8-参数流yaml--csv--paramstxt--gpu)
9. [文件地图](#9-文件地图)
10. [常见问题速查表](#10-常见问题速查表)

---

## 1. 快速开始

### 1.1 编译

```bash
uv run lbm-build --huang
# → lbm_mrt/solver/mcmp_huang_256 (3 MB, sm_120, 256×256)

# 自定义网格（用于网格收敛测试）
uv run lbm-build --huang --grid 100
uv run lbm-build --huang --grid 400
```

### 1.2 一键全验证

```bash
# 生成设计 CSV + 批量运行 + 分析 + 报告
uv run python scripts/06_run_huang_validation_suite.py --all --run
```

### 1.3 单 Sweep 运行

```bash
uv run python scripts/06_run_huang_validation_suite.py --sweep laplace --run
uv run python scripts/06_run_huang_validation_suite.py --sweep decoupling --run
uv run python scripts/06_run_huang_validation_suite.py --sweep coexistence --run
uv run python scripts/06_run_huang_validation_suite.py --sweep spurious --run
```

### 1.4 后处理已有结果

```bash
uv run python scripts/06_run_huang_validation_suite.py --analyze results/validation
uv run python scripts/06_run_huang_validation_suite.py --report results/validation
```

---

## 2. 四大验证详解

### 2.1 Laplace 定律 — $\Delta P = \sigma / R$

| 项目 | 内容 |
|------|------|
| **物理** | 2D 静液滴，验证 $\Delta P = \sigma/R$ 过原点线性 |
| **几何** | 256×256 全周期，`huang_init_mode=1` (液滴) |
| **扫掠** | ε ∈ {0, −0.24, −0.48, −0.72, −0.96} × R ∈ {30, 40, 50, 60} = 20 cases |
| **固定** | Tr=0.90, τ=1.5, Λ=1/12 |
| **判据** | 每组 R² ≥ 0.99, intercept ≈ 0 |
| **收敛** | ~100,000 步（液滴形态 + max\|u\| 稳定） |
| **产出图** | `laplace_fig5.pdf` — ΔP vs 1/R, 5 条 ε 线 |

**原理图**:

```
┌────────────────────────────────┐
│   气体 (ρ_g, p_out)             │
│      ┌──────────┐              │
│      │ 液滴     │  R           │  ΔP = p_in − p_out = σ/R
│      │ ρ_l, p_in│              │
│      └──────────┘              │
└────────────────────────────────┘
```

**分析算法**:
1. `detect_interface_radius()`: 中密度 ($\frac{\rho_l+\rho_g}{2}$) 等值面 → 圆拟合 → $R_{\text{meas}}$
2. `fit_pressure_inside_outside()`: $r < 0.6R$ 区平均 → $p_{\text{in}}$, $r > 1.5R$ 区平均 → $p_{\text{out}}$
3. `laplace_sigma()`: $\Delta P = \sigma/R$ 过原点线性回归 → $\sigma$, $R^2$, intercept

### 2.2 σ-解耦 — $\sigma \propto (1-6k_1)$

| 项目 | 内容 |
|------|------|
| **物理** | 固定 (Tr, R)，扫 $k_1$，验证 $\sigma$ 只取决于 $k_1$，与 $k_2$ 无关 |
| **几何** | 256×256, R=40 液滴, Tr=0.70 |
| **扫掠** | $k_1 \in \{0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15\}$ = 7 cases |
| **判据** | $\sigma$ vs $(1-6k_1)$ R² ≥ 0.99; $\rho_l$, $\rho_g$ 漂移 < 1% |
| **产出图** | `sigma_vs_1minus6k1.pdf` |

**核心关系**:
$$\varepsilon = -8(k_1 + k_2), \quad \sigma \propto (1-6k_1)$$

| $k_1$ | ε | $1-6k_1$ | 表面张力 |
|-------|---|----------|---------|
| 0 | 0 | 1.0 | $\sigma_{\max}$ |
| 0.0833 (1/12) | −0.667 | 0.5 | $\sigma_{\max}/2$ |
| −0.213 | **1.7** | 2.275 | 2.3×增强 |

### 2.3 共存曲线 — $\rho_l, \rho_g$ vs $T_r$

| 项目 | 内容 |
|------|------|
| **物理** | 平界面，提取 $\rho_l(T_r)$, $\rho_g(T_r)$，对比 Maxwell 理论 |
| **几何** | 256×256, `huang_init_mode=2` (平界面 slab) |
| **扫掠** | $T_r \in \{0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95\}$ = 8 cases |
| **固定** | ε=1.7 (最优), τ=1.5 |
| **判据** | 密度偏差 < 2% (ε=1.7 下实测 < 0.03%) |
| **产出图** | `coexistence_curve.pdf` — semilog $\rho$ vs $T_r$ |

> **重要**: ε=1.7 是从 ε ∈ {1.0, 1.2, 1.3, 1.5, 1.7, 2.0} 扫描中确定的最优值。

### 2.4 Spurious Currents — 虚假速度

| 项目 | 内容 |
|------|------|
| **物理** | 静态液滴应无流动；测量残余 $\|\mathbf{u}\|_{\max}$ |
| **几何** | 256×256, R=51 (0.2×NY) |
| **扫掠** | 单点：Tr=0.70, k₁=1/12 |
| **判据** | $\|\mathbf{u}\|_{\max} < 0.05$ (lattice units) |

---

## 3. 单 Case 运行

### 3.1 命令行

```bash
uv run lbm-run \
    --case-name my_case \
    --app lbm_mrt/solver/mcmp_huang_256 \
    --config configs/huang_scmp.yaml \
    cs_T=0.90 \
    epsilon_huang=-0.48 \
    huang_R0=40.0 \
    huang_rho_g=0.04541 \
    huang_rho_l=0.24811
```

### 3.2 输出结构

```
results/my_case/
├── params.txt              # 展平后的所有参数
├── log.txt                 # 求解器 stdout/stderr
├── outputdata_scmp/        # VTK 输出
│   ├── flow0.vtk           # 初始场
│   ├── flow5000.vtk        # 每 5000 步
│   ├── flow10000.vtk
│   └── ...
└── ckpt/                   # 断点（默认关闭）
```

### 3.3 已验证成功的参数组合

#### Tr=0.90（低密度比 ~5:1，液滴稳定）

```bash
# ε sweep: 随意换 epsilon_huang ∈ {0, -0.24, -0.48, -0.72, -0.96}
uv run lbm-run \
    --case-name laplace_Tr0.90_eps-0.48_R40 \
    --app lbm_mrt/solver/mcmp_huang_256 \
    --config configs/huang_scmp.yaml \
    cs_T=0.90 epsilon_huang=-0.48 huang_R0=40.0 \
    huang_rho_g=0.04541 huang_rho_l=0.24811
```

#### Tr=0.70（高密度比 ~33:1，需要 ε=1.7 才稳定）

```bash
# 唯一稳定组合：ε=1.7
uv run lbm-run \
    --case-name laplace_Tr0.70_eps1.7_R40 \
    --app lbm_mrt/solver/mcmp_huang_256 \
    --config configs/huang_scmp.yaml \
    cs_T=0.70 epsilon_huang=1.7 huang_R0=40.0 \
    huang_rho_g=0.009170 huang_rho_l=0.305202
```

### 3.4 快速验证液滴是否稳定

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars
from lbm_mrt.validation.analytical import detect_interface_radius

v = latest_vtk('results/my_case/outputdata_scmp')
f, nx, ny = read_vtk_scalars(v)
r = f['rho']
c, R = detect_interface_radius(r, nx, ny)
stable = 0.001 < r.min() < 0.05 and R > 25
print(f'rho=[{r.min():.4f}, {r.max():.4f}], R={R:.1f}')
print('液滴稳定 ✅' if stable else '液滴异常 ❌ (可能溶解或膨胀)')
"
```

---

## 4. 参数选择指南

### 4.1 CS-EOS 参数（固定）

| 参数 | 值 | 说明 |
|------|-----|------|
| `cs_a` | 1.0 | Carnahan-Starling 引力参数 |
| `cs_b` | 4.0 | 硬球体积 |
| `cs_R` | 1.0 | 气体常数 |
| `cs_G` | −1.0 | 相互作用强度 |
| $T_c$ | 0.09433 | 临界温度 ($0.3773a/(bR)$) |

### 4.2 约化温度 `cs_T`

> **`cs_T` 填的是约化温度 $T_r = T/T_c$，不是绝对温度！** 求解器内部乘以 $T_c$。

| $T_r$ | 绝对 T | ρ_l (Maxwell) | ρ_g (Maxwell) | ρ_l/ρ_g | 液滴稳定性 |
|-------|--------|---------------|---------------|---------|:----------:|
| 0.70 | 0.0660 | 0.358 | 0.00929 | 38.5 | ❌ 需 ε=1.7 |
| 0.85 | 0.0802 | 0.279 | 0.0316 | 8.8 | ⚠️ 未充分测试 |
| **0.90** | **0.0849** | **0.248** | **0.0454** | **5.5** | **✅ 稳定** |
| 0.95 | 0.0896 | 0.189 | 0.0667 | 2.8 | ✅ 最稳定 |

### 4.3 ε 与表面张力

$$\varepsilon = -8(k_1 + k_2), \quad k_1 = -\varepsilon/8 \quad (\text{当 } k_2=0)$$

| ε | 实际 k₁ | 用途 |
|---|---------|------|
| **1.7** | −0.213 | **共存最优，Tr=0.70 液滴稳定** |
| 0.00 | 0.000 | 基准 MRT（无 Q_m 修正） |
| −0.24 | 0.030 | Laplace 扫掠 |
| −0.48 | 0.060 | Laplace 扫掠 |
| −0.72 | 0.090 | Laplace 扫掠 |
| −0.96 | 0.120 | Laplace 扫掠 |

> ⚠️ **关键**: `k1_huang` 在 params.txt 中只是遗留值，求解器从 ε 计算 k₁。设置 `epsilon_huang` 即可！

### 4.4 MRT 松弛参数（固定）

| 参数 | 值 | 说明 |
|------|-----|------|
| `tau_huang` | 1.5 | MRT 松弛时间 ($\nu \approx 0.5$) |
| `Lambda_huang` | 0.08333 (1/12) | 各向异性消除 magic parameter |
| `kd_huang` | −0.08333 (−1/12) | 压力张量常数 |
| `alpha_meq` | 1.0 | $\alpha$-modified $m_1^{eq}$ |

### 4.5 液滴初始化参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `huang_R0` | 30–60 | 液滴半径（R/NY ≤ 0.234 避免壁效应） |
| `huang_xc`, `huang_yc` | 128, 128 | 域中心 |
| `huang_W` | 3.0 | 界面 tanh 厚度 |
| `huang_rho_g`, `huang_rho_l` | Maxwell 或 LBM 实测值 | **必须显式注入！** |
| `huang_init_mode` | 1 | 液滴模式 |

### 4.6 共存密度：注入值 vs LBM 实测值

| $T_r$ | ε | 注入值 (Maxwell) | LBM 实测值 | 推荐 |
|-------|---|-----------------|-----------|:----:|
| 0.90 | −0.48 | ρ_g=0.0454, ρ_l=0.248 | ρ_g≈0.038, ρ_l≈0.245 | Maxwell |
| 0.70 | 1.7 | ρ_g=0.00929, ρ_l=0.358 | ρ_g=0.00917, ρ_l=0.305 | **LBM 实测** |

> 对于 ε=1.7, Tr=0.70：用 LBM 实测值（来自平界面 VTK）初始化，液滴更稳定。

---

## 5. ⚠️ 液滴不稳定的排查与解决

### 5.1 现象

| 时间步 | ρ 范围 | R | 状态 |
|--------|--------|----|:----:|
| t=0 | [0.009, 0.358] | 20 | ✅ 完美液滴 |
| t=5,000 | [0.000, 0.356] | 29 | ⚠️ 膨胀 + 气体密度降至 0 |
| t=50,000 | [0.000, 0.356] | 89 | ❌ 几乎填满全域 |
| t=100,000+ | [0.349, 0.349] | — | ❌ 完全均匀化 |

### 5.2 根因分析

```
Tr=0.70 + ε=−0.667 (k₁=+0.083)
  ├─ σ = 0.5×σ_max（表面张力弱）
  ├─ 气体相 ρ_g → 0（数值漂移）
  ├─ ψ → 0（伪势消失）
  └─ 相互作用力归零 → 液滴溶解 ❌
```

**解决**: ε=1.7 (k₁=−0.213, σ=2.3×σ_max) → 表面张力增强 4.5 倍 → 液滴稳定 ✅

### 5.3 排查 Checklist

1. **检查初始条件**: `flow0.vtk` 中 `rho.min()` 和 `rho.max()` 是否等于注入的 ρ_g, ρ_l
2. **检查 ε 符号**: 正值 ε 增强表面张力，负值减弱
3. **检查 cs_T**: Tr=0.90 天然比 Tr=0.70 更稳定
4. **检查共存密度注入**: 是否显式设置了 `huang_rho_g` 和 `huang_rho_l`
5. **检查时间演化**: 跑 5,000 步后查看 `flow5000.vtk`

```bash
# 快速诊断脚本
uv run python -c "
import sys; sys.path.insert(0, '.')
from lbm_mrt.io.vtk_reader import read_vtk_scalars
from lbm_mrt.validation.analytical import detect_interface_radius
import os

case = 'results/my_case'
vtk_dir = os.path.join(case, 'outputdata_scmp')
for vtk_name in ['flow0.vtk', 'flow5000.vtk', 'flow50000.vtk']:
    path = os.path.join(vtk_dir, vtk_name)
    if not os.path.exists(path): continue
    f, nx, ny = read_vtk_scalars(path)
    r = f['rho']
    c, R = detect_interface_radius(r, nx, ny)
    s = '✅' if 0.001 < r.min() < 0.05 and R > 25 else '❌'
    print(f'{vtk_name}: rho=[{r.min():.4f},{r.max():.4f}], R={R:.1f} {s}')
"
```

### 5.4 已知稳定组合

| Tr | ε | k₁ | 状态 | 验证来源 |
|----|---|-----|:----:|---------|
| 0.90 | 0.00 | 0.000 | ✅ | `results/validation/laplace_eps0.00_R*` |
| 0.90 | −0.48 | 0.060 | ✅ | `results/validation/laplace_eps-0.48_R*` |
| 0.90 | −0.96 | 0.120 | ✅ | `results/validation/laplace_eps-0.96_R*` |
| **0.70** | **1.7** | **−0.213** | **✅** | `results/laplace_Tr0.70_eps1.7_R40` |
| 0.70 | −0.667 | 0.083 | ❌ | 液滴溶解 |
| 0.70 | −0.48 | 0.060 | ❌ | 未测试（推定为不稳定） |

---

## 6. 后处理与分析

### 6.1 单 Case 快速检查

```bash
# 密度场
uv run python -c "
from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars
from lbm_mrt.validation.analytical import detect_interface_radius, extract_rho_l_g
v = latest_vtk('results/my_case/outputdata_scmp')
f, nx, ny = read_vtk_scalars(v)
print(f'ρ=[{f[\"rho\"].min():.4f}, {f[\"rho\"].max():.4f}]')
c, R = detect_interface_radius(f['rho'], nx, ny)
print(f'R={R:.1f}, center=({c[0]:.1f},{c[1]:.1f})')
"
```

### 6.2 Maxwell 共存对照

```bash
uv run python -c "
from lbm_mrt.validation.cs_eos import maxwell_coexistence, cs_critical_point
Tc = cs_critical_point(1.0, 4.0, 1.0)[0]
for Tr in [0.70, 0.85, 0.90]:
    rg, rl, peq = maxwell_coexistence(1.0, 4.0, 1.0, Tr*Tc)
    print(f'Tr={Tr:.2f}: ρ_g={rg:.6f}, ρ_l={rl:.6f}, p_eq={peq:.2e}')
"
```

### 6.3 批量分析

```bash
# 使用验证套件分析
uv run python scripts/06_run_huang_validation_suite.py --analyze results/validation

# 生成报告
uv run python scripts/06_run_huang_validation_suite.py --report results/validation
```

### 6.4 直接分析旧 Laplace 结果

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from lbm_mrt.io.vtk_reader import read_vtk_scalars, latest_vtk
from lbm_mrt.validation.analytical import detect_interface_radius, fit_pressure_inside_outside, laplace_sigma
import numpy as np, os, re

results_dir = 'results/validation'
pattern = re.compile(r'laplace_eps([-\d.]+)_R(\d+)')

groups = {}
for case_name in sorted(os.listdir(results_dir)):
    m = pattern.match(case_name)
    if not m: continue
    eps, R_nom = float(m.group(1)), float(m.group(2))
    v = latest_vtk(os.path.join(results_dir, case_name, 'outputdata_scmp'))
    if not v: continue
    f, nx, ny = read_vtk_scalars(v)
    c, R = detect_interface_radius(f['rho'], nx, ny)
    p_in, p_out = fit_pressure_inside_outside(f['rho'], f.get('pressure', f['rho']*0), c, R)
    groups.setdefault(eps, []).append((R, p_in-p_out))

for eps in sorted(groups.keys()):
    pts = groups[eps]
    if len(pts) < 3: continue
    R_arr = np.array([p[0] for p in pts])
    dP_arr = np.array([p[1] for p in pts])
    sigma, r2, _ = laplace_sigma(R_arr, dP_arr)
    print(f'ε={eps:+.2f}: σ={sigma:.4e}, R²={r2:.4f}, n={len(pts)}')
"
```

---

## 7. 已知 Bug 与修复历史

| # | Bug | 现象 | 修复 | 日期 |
|---|-----|------|------|------|
| 1 | ρ 从未从 $f_i$ 更新 | 所有验证 ρ 冻结于初始值 | 在 `evolution_scmp` 首行添加 `compute_rho_from_fin_gpu` | 2026-05-21 |
| 2 | Q_m 分母 `fabs(G)` 而非 `G` | 平板共存界面崩塌 | 改为 `G` 带符号，加 fmax 保护分母 | 2026-05-22 |
| 3 | 接触角 yc=128 居中 | 液滴"脱壁"（180°） | 改为 yc=1（贴壁） | 2026-05-24 |
| 4 | 共存密度提取误取界面区 | ρ_l 偏低 12-15% | 改用 97th/3rd 百分位法 | 2026-05-24 |

---

## 8. 参数流：YAML → CSV → params.txt → GPU

```
configs/huang_scmp.yaml
        │  load_config() → _flatten()
        ▼
  flat dict: {pp_mode: 1, cs_T: 0.70, epsilon_huang: -0.667, ...}
        │
        │  + CSV columns override (e.g. cs_T=0.90, epsilon_huang=1.7)
        ▼
  per-case flat dict
        │  write_params_txt()
        ▼
params.txt  ("key value" per line)
        │  load_params_txt()
        ▼
RuntimeParams struct
        │  k1_computed = -epsilon_huang / 8.0 - k2_huang  ← 关键！
        │  cudaMemcpyToSymbol → __constant__
        ▼
GPU kernels: get_epsilon_huang(), get_k1_huang(), ...
```

> ⚠️ **`k1_huang` 在 params.txt 中是无效的遗留值。** 求解器始终从 `epsilon_huang` 计算 k₁！

---

## 9. 文件地图

### 核心代码

| 路径 | 作用 |
|------|------|
| `lbm_mrt/solver/src/LBM.cu` | GPU 核函数：EOS、力、Q_m、碰撞、边界、初始化 |
| `lbm_mrt/solver/src/sim_utils.cu` | 参数读取、常量上传、SCMP 运行循环 |
| `lbm_mrt/solver/include/LBM.h` | 编译期常量 (NX=256, NY=256)、D2Q9 格子 |
| `lbm_mrt/solver/build.py` | `lbm-build --huang` 编译入口 |

### Python 层

| 路径 | 作用 |
|------|------|
| `lbm_mrt/core/config.py` | YAML → flat dict |
| `lbm_mrt/core/paths.py` | 路径常量 |
| `lbm_mrt/runners/single_run.py` | `run_one()`: 单 case 运行 |
| `lbm_mrt/runners/batch_run.py` | `run_batch()`: CSV 批量 |
| `lbm_mrt/io/params_writer.py` | `write_params_txt()` |
| `lbm_mrt/io/vtk_reader.py` | `read_vtk_scalars()`, `latest_vtk()` |

### 验证模块

| 路径 | 作用 |
|------|------|
| `lbm_mrt/validation/analytical.py` | `detect_interface_radius`, `fit_pressure_inside_outside`, `laplace_sigma`, `extract_rho_l_g` |
| `lbm_mrt/validation/cs_eos.py` | CS-EOS 压力/化学势、Maxwell 共存 |
| `lbm_mrt/validation/laplace_law.py` | Laplace 批量分析 |
| `lbm_mrt/validation/decoupling_sweep.py` | σ-解耦分析 |
| `lbm_mrt/validation/coexistence.py` | 共存曲线分析 |
| `lbm_mrt/validation/spurious_currents.py` | Spurious 分析 |

### 配置与数据

| 路径 | 作用 |
|------|------|
| `configs/huang_scmp.yaml` | SCMP 默认配置 |
| `scripts/06_run_huang_validation_suite.py` | 验证套件主入口 |
| `data/design_scmp_laplace.csv` | Laplace 设计 CSV |
| `data/design_scmp_decoupling.csv` | 解耦设计 CSV |
| `data/design_scmp_coexistence.csv` | 共存设计 CSV |
| `data/design_scmp_spurious.csv` | Spurious 设计 CSV |

### 参考文档

| 路径 | 内容 |
|------|------|
| `research/DEBUG_SUMMARY_20260524.md` | 完整 Debug 总结（Bug + 结果） |
| `research/HUANG_2016_VALIDATION_STRATEGY.md` | 论文复现策略 |
| `research/VALIDATION_CODE_GUIDE.md` | 代码结构指南 |
| `research/HUANG_MRT_REFACTOR.md` | 方程→代码映射 |
| `research/_archive/validation_plan.md` | 方法学说明 (plan-as-spec) |

---

## 10. 常见问题速查表

| 症状 | 原因 | 解决 |
|------|------|------|
| 液滴溶解（ρ 均匀化） | Tr 太低 + ε 太小 → 表面张力弱 | 提高 Tr（0.90）或增大 ε（1.7） |
| 气体相密度降到 0 | ψ→0 → 力消失 | 检查 cs_T 和 epsilon_huang 组合 |
| 所有 case σ 相同 | `epsilon_huang` 不在 CSV 中，用了 YAML 默认值 | 在 CSV 添加 `epsilon_huang` 列 |
| ρ 场冻结不演化 | Bug #1（缺 `compute_rho_from_fin_gpu`） | 确保使用最新求解器 |
| 平界面崩塌 | Bug #2（`fabs(G)` 符号错误） | 确保使用最新求解器 |
| `params.txt` 中有 k1_huang 但无效 | 求解器从 ε 计算 k₁ | 设置 `epsilon_huang` 即可，忽略 `k1_huang` |
| Laplace 压差异常 | VTK `pressure` 场不含压力张量梯度项 | 已知限制，待求解器扩展 |
| 共存 ρ_l 偏差 >10% | 提取函数取了界面区密度 | 用 `extract_rho_l_g()` 的百分位模式 |
| 绘图中文乱码 | Unicode 下标字符 | 用 LaTeX `$k_1$` 替代 `₁` |
