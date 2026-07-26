# Huang & Wu (2016) SCMP 验证 — 完整 Debug 总结

> 日期: 2026-05-21 至 2026-05-24
> 分支: `huang_mrt_2d`
> 产出目录: `results/20260522/`
> 二进制: `lbm_mrt/solver/mcmp_huang_{100,200,256,400}`

---

## §0 概述

在四天的集中调试中，完成了 Huang & Wu (2016) 单组份赝势 (SCMP) MRT-LBM 求解器的全部数值验证：Laplace 定律、σ-解耦 (k₂≠0)、共存曲线 (ε 优化)、Poiseuille 流、接触角 (Scheme IV + G_ads)、网格无关性。发现并修复了两个关键 Bug（ρ 冻结、Q_m 符号），确立了最优参数 ε=1.7，建立了 Scheme IV 接触角标定曲线。

---

## §1 代码架构快速参考

### 关键文件

| 文件 | 作用 |
|------|------|
| `lbm_mrt/solver/include/LBM.h` | 编译期常量 (NX, NY)、D2Q9 格子、MRT 矩阵、松弛率、设备全局变量声明 |
| `lbm_mrt/solver/src/LBM.cu` | **核心** — 所有 GPU 核函数：EOS、力、Q_m、碰撞、边界、初始化、演化 |
| `lbm_mrt/solver/src/sim_utils.cu` | 参数读取、常量上传 (`push_device_constants`)、运行循环 (`run_scmp_huang`) |
| `lbm_mrt/solver/src/main.cu` | 入口 + MCMP 逻辑 |
| `lbm_mrt/core/config.py` | Python 配置加载和展平 |
| `lbm_mrt/io/vtk_reader.py` | VTK 二进制读取 |
| `lbm_mrt/validation/analytical.py` | `detect_interface_radius`, `compute_sigma_from_rho`, `extract_rho_l_g` |
| `lbm_mrt/validation/contact_angle.py` | 圆拟合 + θ 测量 |
| `lbm_mrt/validation/cs_eos.py` | CS EOS 压力、化学势、Maxwell 共存 |
| `lbm_mrt/viz/viz_template.py` | Matplotlib 发表级 PDF 生成 (STIX 字体, journal 模式) |
| `configs/huang_scmp.yaml` | SCMP 默认配置 |

### 演化调用顺序 (`evolution_scmp`)

```
compute_rho_from_fin_gpu     ← ρ = Σ f_i
compute_p_psi_scmp_cs        ← p(ρ), ψ(ρ) from CS EOS
update_ghost_psi_bc          ← Scheme IV 接触角
compute_molecular_force_scmp ← F_mol = -Gψ∇ψ + G·ρ
compute_Q_huang_gpu          ← Q_m from F_mol
compute_adsorption_force_scmp + add_adsorption_to_force
compute_velocity_scmp        ← u from fin + F
compute_S_huang_gpu          ← Guo forcing term
mrt_collide_single_component_gpu
stream_single_component_gpu
boundary_scmp_gpu
zouhe_bottom_wall_gpu        ← mode=4 only
compute_pressure_tensor_scmp
```

### 核心参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `cs_a, cs_b, cs_R` | 1.0, 4.0, 1.0 | CS EOS 参数 |
| `cs_G` | −1.0 | 分子间作用力强度 |
| `cs_T` | 0.90 | Tr=T/Tc（多数验证用此值） |
| `epsilon_huang` | **1.7** (最优) | ε = −8(k₁+k₂)，调节力学稳定性 |
| `k2_huang` | 0.0 | k₂ 次要参数 |
| `tau_huang` | 1.5 | MRT 松弛时间 |
| `Lambda_huang` | 0.08333 (=1/12) | 各向异性消除 |

---

## §2 Bug 修复清单

### Bug 1: ρ 从未从分布函数更新 (2026-05-21)

**位置**: `evolution_scmp` → 缺少 `compute_rho_from_fin_gpu` 调用

**现象**: 所有验证中 ρ 场冻结于初始值。旧 MCMP 有 `update_fluid_A_rho_psi_pressure` (含 ρ 计算)，SCMP 遗漏。

**修复**: 在 `evolution_scmp` 首行加入：
```c
compute_rho_from_fin_gpu<<<grid_scmp, threads>>>(rho, fin, pointsflag);
```

**影响**: 接触角从 91.8°(不变) 恢复为可调节。其他验证因之前 ρ 冻结给出假结果，全部重跑。

### Bug 2: Q_m 分母使用 `fabs(G)` 而非 `G` (2026-05-22)

**位置**: `compute_Q_huang_gpu` (LBM.cu ~line 2365)

**现象**: 平板共存界面完全崩塌。`fabs(-1.0)=1.0` 使 Q_m 符号与压力张量不一致。

**修复** (用户发现):
```c
// 修复前
denom = fabs(cs_G_loc) * psi2 * c_gpu * c_gpu;

// 修复后
denom = cs_G_loc * psi2 * c_gpu * c_gpu;
denom = fmax(fabs(denom), 1e-12) * (denom < 0 ? -1.0 : 1.0);
```

**影响**: 平板共存从崩塌恢复，ε=1 时出现可识别的气液分离。σ-(1−6k₁) 线性关系保持 R²=0.9994。

### Bug 3: 接触角初始化 yc=128 (2026-05-24)

**位置**: CSV 设计 → `huang_yc: 128`

**现象**: ε=1.7 下液滴"脱壁"(全 180°)。实际是液滴初始化在域中央 (yc=128)，从未接触壁面。

**修复**: 改为 `huang_yc: 1` (贴壁)。Mode=4 的初始化公式与 mode=1 相同 (径向 tanh)，仅靠 yc 参数区分。

### Bug 4: 共存密度提取 `extract_rho_l_g` 使用梯度法误取界面区密度

**位置**: `lbm_mrt/validation/analytical.py`

**现象**: 平板共存 ρ_l 比 Maxwell 低 12-15%。

**修复**: 改用百分位法 (97th/3rd percentile) 提取 bulk 密度：
```python
rho_l = np.percentile(rho_flat, 97)  # 避免界面过渡区
rho_g = np.percentile(rho_flat, 3)
```
修复后 ρ_l 偏差降至 0.03%。

---

## §3 验证结果总览

### ε=1.7 最优参数下的完整结果

| 验证项 | Case 数 | 指标 | 状态 |
|--------|---------|------|------|
| Laplace 定律 | 20 (5ε×4R) | per-ε R²=0.99+ | ✅ |
| σ-解耦 (k₂≠0) | 7 | σ∝(1−6k₁), R²=0.9994 | ✅ |
| 平板共存 | 8 (Tr=0.60–0.95) | ρ_l 偏差 0.00-0.03% | ✅ ε=1.7 最优 |
| Poiseuille 流 | 2 | u_max 偏差 0.03% | ✅ |
| 接触角 (Scheme IV) | 25 (密集标定) | 30°–145° 可用 | ✅ 含反函数 |
| 接触角 (G_ads) | 11 | −0.05 到 +0.03 可用 | ⚠️ 范围窄，非线性 |
| 网格无关性 | 12 (4网格×3ε) | σ 随 Nx 收敛 | ✅ |

### ε 最优值确定

```
ε 扫描: 1.0(Δ=-0.40%) → 1.2(Δ=-0.27%) → 1.5(Δ=-0.09%) → 1.7(Δ=+0.02%) → 2.0(Δ=+0.18%)
最优: ε=1.7 (偏差 <0.03%, 所有 Tr 通过)
```

---

## §4 关键经验和教训

### 4.1 技术教训

1. **`fabs(G)` vs `G`** — 论文写 `|G|` 但代码压力张量用 `G` (带符号)。离散一致性要求 Q_m 分母也用 `G`，否则平板界面崩溃。这是用户的关键发现。

2. **周期性 BC 与平坦界面不兼容** — 单界面 tanh 的 ρ(0)≠ρ(NY−1) 违反周期性。必须用 slab (双界面) 且不等体积 (77% 液体) 来稳定取向。

3. **旧代码的"正确"可能是假象** — ρ 冻结时共存曲线显示 0% 偏差，但这是假结果。验证每一层的场更新是否正确计算。

4. **后处理 bug 比物理 bug 更难发现** — 共存密度偏移 15% 是因为提取函数取错了区域，而非 LBM 模型问题。Laplace 液滴法验证了模型正确性。

5. **ε 和 k₁ 独立但都有副作用** — ε>0 使 P_Q 为正 → 壁面排斥 → 接触角需独立标定。共存调参 (用 ε) 和表面张力 (用 k₁) 是论文的核心创新。

### 4.2 调试策略

- **对比工作/失败案例**: Laplace (工作) vs 平板共存 (失败) 的密度场对比直接锁定了提取函数问题
- **参数溯源**: 每次怀疑都回溯到 params.txt 确认实际运行值
- **小步验证**: ε 扫描时先跑 2 个 Tr 定位，再跑完整 8 个

---

## §5 命令行参考

### 编译

```bash
# 标准 256²
uv run lbm-build --huang --arch sm_120

# 自定义网格
uv run lbm-build --huang --grid 100 --arch sm_120
uv run lbm-build --huang --grid 400 --arch sm_120
```

### 批量运行

```bash
# 完整 batch
uv run lbm-batch --csv data/design_xxx.csv \
    --config configs/huang_scmp.yaml \
    --out-root results/20260522 \
    --app lbm_mrt/solver/mcmp_huang_256

# 筛选单 case
uv run lbm-batch --csv <(head -1 data/xxx.csv; grep case_name data/xxx.csv) \
    --config configs/huang_scmp.yaml \
    --out-root results/20260522 \
    --app lbm_mrt/solver/mcmp_huang_256
```

### 可视化

```python
from lbm_mrt.viz.viz_template import init_style, create_figure_ax, save_figure
init_style(mode='journal', base_fontsize=20, bold=True, axis_linewidth=2.0)
fig, ax = create_figure_ax(figsize=(7, 6))
# ... plot ...
save_figure(fig, 'results/20260522/plot.pdf', dpi=300)
```

### 分析

```bash
# 快速密度检查
python -c "
from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars
from lbm_mrt.validation.analytical import extract_rho_l_g
fields, nx, ny = read_vtk_scalars(vtk_path)
rho_l, rho_g = extract_rho_l_g(fields['rho'], ny)
"

# 接触角测量
python -c "
from lbm_mrt.validation.contact_angle import compute_theta_from_vtk
result = compute_theta_from_vtk('path/to/flow200000.vtk')
print(result['theta_deg'])
"

# Maxwell 共存对照
python -c "
from lbm_mrt.validation.cs_eos import maxwell_coexistence, cs_critical_point
Tc = cs_critical_point(1.0, 4.0, 1.0)[0]
mw = maxwell_coexistence(1.0, 4.0, 1.0, 0.9*Tc)
"
```

---

## §6 当前参数配置（ε=1.7 最优）

### `configs/huang_scmp.yaml` 关键字段

```yaml
cs_a: 1.0
cs_b: 4.0
cs_R: 1.0
cs_G: -1.0
cs_T: 0.90
epsilon_huang: 1.7          # ← 最优值，匹配 Maxwell 共存
k2_huang: 0.0
tau_huang: 1.5
Lambda_huang: 0.083333      # 1/12
huang_init_mode: 4           # 接触角用 mode=4
huang_R0: 30
huang_xc: 128
huang_yc: 1                  # ← 贴壁！非 128！
huang_W: 4.0
pp_mode: 1
```

### Scheme IV 接触角标定（ε=1.7, Tr=0.90）

```python
# 反向公式: θ_in → θ_meas
θ_in = -1.78e-4·θ_meas³ + 4.95e-2·θ_meas² - 3.02·θ_meas + 84.5

# 查表
| θ_meas  | θ_in  |
|---------|-------|
| 30°     | 25°   |
| 60°     | 43°   |
| 90°     | 84°   |
| 120°    | 127°  |
| 140°    | 148°  |
```

---

## §7 产出 PDF 清单

`results/20260522/` 下所有发表级可视化：

| PDF | 内容 |
|-----|------|
| `coexistence_curve.pdf` | 共存曲线 (ε=0,1,1.2,1.7,2.0) semilog, 论文 Fig.4 风格 |
| `laplace_fig5.pdf` | Laplace 定律 ΔP vs 1/R (5 ε 线) |
| `sigma_vs_1minus6k1.pdf` | σ vs (1−6k₁) 线性关系 |
| `poiseuille_velocity_profile.pdf` | Poiseuille 速度剖面 vs 解析解 |
| `contact_angle_calibration.pdf` | Scheme IV 标定 (密集点+正反拟合) |
| `contact_angle_Gads.pdf` | G_ads 扫参 |
| `mesh_convergence.pdf` | 网格无关性 σ vs Nx |

---

## §8 已知限制和后续工作

1. **低 Tr 气相不稳定** — Tr<0.80 时 ρ_g→0（晶格数值精度限制），对单相液态应用无影响
2. **G_ads 可用范围窄** — 仅 [−0.05,+0.03]，Scheme IV 为首选方案
3. **ε-k₁ 兼容性** — 接触角标定需针对特定 (ε,Tr,R₀) 组合，换参数需重新标定
4. **Q_m 预乘松弛率** — `C[1]=se*Qm1` 与论文 Eq.(5) 不一致（论文 Q_m 不进 Λ），但当前验证全部通过，影响待评估
5. **分子力中混入体力** — `Fx = -Gψ∇ψ + G·ρ`，对无体力验证无影响，Poiseuille 等有体力算例有潜在交叉污染
