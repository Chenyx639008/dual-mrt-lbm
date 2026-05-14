# Huang-MRT SCMP 验证套件 — 方法学说明文档（plan-as-spec）

> 本文件是一份 **方法学说明文档（method spec）**，不是 code change plan。
> 目的：把论文 Supporting Information 图 S1–S5 的四套 MCMP 验证（Laplace / 接触角 / Poiseuille / 网格收敛），在重构后的 **Huang & Wu (2016) 单组分双相（SCMP）** 求解器上重做，方法上明确每一项的几何、扫描设计、收敛判据、后处理算法与可视化要点。
> 全文路径、模块名、入口命令、CSV 列名都已经对齐到 **本仓库的真实状态**；读完后应当能直接落地实现，无需再回查原 .cu / .py。

---

## §0 当前现状（必读）

### 0.1 仓库布局（本仓库已就位的部分）

```
huang_mrt_2d/                             ← 即本工作树根目录
├── lbm_mrt/
│   ├── solver/                            ✓ mcmp_huang_256 已构建（256×256 SCMP）
│   │   ├── build.py                       ✓ uv run lbm-build --huang 可用
│   │   ├── mcmp_huang_256                 ✓ 二进制就位
│   │   ├── src/{LBM.cu, sim_utils.cu, …}  ✓ Phase 1–3 完成
│   │   └── include/{LBM.h, sim_utils.h}
│   ├── validation/                        ⚙️ Python 验证模块（部分就位）
│   │   ├── cs_eos.py                      ✓ Maxwell 共存（已修复，108 测试通过）
│   │   ├── coexistence.py                 ✓ 平界面剖面 + Maxwell 对比
│   │   ├── analytical.py                  ✓ radius / pressure_in_out / laplace_sigma
│   │   ├── laplace_law.py                 ☐ 待新增（Δp ~ 1/R 双 Tr 拟合 + 论文风格图）
│   │   ├── spurious_currents.py           ☐ 待新增（max|u| 扫描 + Huang vs Li 对比）
│   │   ├── decoupling_sweep.py            ☐ 待新增（σ ~ 1−6k₁ + ρ_l/ρ_g 漂移检测）
│   │   ├── contact_angle.py               ☐ 待新增（圆拟合 + θ-G_ads 线性回归）
│   │   ├── poiseuille_sp.py               ☐ 待新增（u(y) 解析对比）
│   │   └── mesh_convergence.py            ☐ 待新增（NY-ε 收敛 + Richardson）
│   ├── runners/{single_run.py, batch_run.py}   ✓ 通用扫掠驱动
│   ├── io/{vtk_reader.py, params_writer.py}    ✓
│   └── viz/viz_template.py                ✓ 论文级 matplotlib
├── configs/
│   ├── default.yaml                       ✓ legacy MCMP 默认
│   └── huang_scmp.yaml                    ✓ SCMP 默认（cs_a=0.75, cs_T=0.06）
├── scripts/
│   ├── 06_run_huang_validation_suite.py   ⚙️ 已支持 --sweep {laplace,decoupling,coexistence}
│   │                                          ☐ 待补 contact_angle / poiseuille / mesh_convergence / spurious
│   ├── visualize_cs_coexistence.py        ✓ Maxwell 曲线绘制
│   └── visualize_cs_coexistence_paper.py  ✓ 论文版双面板
├── data/
│   ├── design_scmp_laplace.csv            ✓ 7-R 单 Tr 设计（cs_T=0.06）
│   ├── design_scmp_laplace_mini.csv       ✓ 2-R smoke
│   └── validation_reference/huang_2016/cs_coexistence.json  ✓ 参考共存曲线
├── tests/test_huang_scmp.py               ✓ Maxwell 单元测试通过
├── validation/                            ← legacy Li-MCMP 验证（参考用，不直接复用）
│   ├── surface/{validation_r_eq1,eq2}/    ↪ §3 后处理脚本可移植
│   ├── contact_angle/{contact_eq1,eq2}/   ↪ §7 圆拟合算法移植
│   ├── twophase/{validation_v_eq1,eq2}/   ↪ §8 mid-line 速度提取移植
│   └── mesh_convergence/                  ↪ §9 Richardson 后处理移植
└── research/                              参考文档（HUANG_MRT_REFACTOR.md / Research.md / …）
```

### 0.2 求解器侧能力清单

| 能力 | 状态 | 备注 |
|---|---|---|
| SCMP CS-EOS + Q_m | ✓ | `LBM.cu::compute_p_psi_scmp_cs` / `compute_Q_huang_gpu` |
| 液滴 tanh 初始化（mode=1）| ✓ | `huang_init_mode=1`，半径 `huang_R0`、中心 `huang_xc/yc`、厚度 `huang_W` |
| 平界面初始化（mode=2）| ✓ | y 方向 tanh |
| 共存密度注入（host→device）| ✓ | `huang_rho_g` / `huang_rho_l` 字段；非零时覆盖 GPU 内启发式 |
| 周期边界 | ✓ | SCMP 全域 `pointsflag=1`，周期 4 边 |
| **bounce-back 壁** | ☐ | SCMP 路径**不支持** wall flag / ghost — §7、§8、§9 实现前需扩展 |
| **Yang/Li 润湿力（G_ads·ψ）** | ☐ | 求解器侧未接，§7 需补 |
| **均匀体力驱动（Gx/Gy）注入 SCMP** | ☐ | `compute_molecular_force_scmp` 目前只有分子力，§8/§9 需补 |
| **均匀液相初始化（mode=3）** | ☐ | §8/§9 需要全水（u=0、ρ=ρ_l），可借 mode=1 用大 R0 也可，但更干净是加 mode=3 |
| VTK 输出 | ✓ | 写 `<exp_dir>/outputdata_scmp/flow*.vtk`（注意目录名是 `outputdata_scmp`，不是 `outputdata_eq`）|
| `run_summary.txt` | ✓ | 全部 case 都有 |

### 0.3 已确认的方法学决策（来自 `change_plan.md` + 用户对话）

| 决策点 | 选择 |
|---|---|
| Eq1/Eq2 对照如何在单组分上复现 | **两个 Tr（默认 Tr=0.7 / 0.9）做两组对照**，分别给不同 ρ_l/ρ_g 比与 σ |
| 相对渗透率（双相）如何处理 | **改造为单相纯液 Poiseuille**，验证 u(y) 与解析解 |
| 网格无关性承载几何 | **Poiseuille ε(NY) + 静态液滴 spurious(NY) 双子图并列** |
| 接触角机制 | **保留 Yang/Li 风格 G_ads·ψ**，求解器内 SCMP 路径需新增此力项 |
| 输出形态 | **方法学说明文档**（plan-as-spec，函数/路径/CSV 列名/伪代码级精度） |
| σ-decoupling 是否做 | **做**，作为 Huang 方法相对 Li 的优势凸显 |

### 0.4 实现优先级（按依赖排序）

```
P0：纯 Python 后处理（求解器已就绪，立即可写） ────────────────────
  §3 Laplace 双 Tr        →  lbm_mrt/validation/laplace_law.py
  §4 σ-decoupling         →  lbm_mrt/validation/decoupling_sweep.py
  §5 共存曲线（已有 coexistence.py，补 driver+plot）
  §6 Spurious             →  lbm_mrt/validation/spurious_currents.py

P1：需先扩展 SCMP 求解器，再做 Python 验证 ──────────────────────
  §7 接触角  → 求解器: bounce-back wall + G_ads·ψ kernel + wall_mat 表
            → Python: lbm_mrt/validation/contact_angle.py
  §8 Poiseuille → 求解器: bounce-back wall + 均匀体力注入 + 均匀液初始化
                → Python: lbm_mrt/validation/poiseuille_sp.py
  §9 网格收敛 → 求解器: 同 §8（增加编译期 NX/NY 注入机制）
              → Python: lbm_mrt/validation/mesh_convergence.py
```

---

## §1 仓库使用约定（入口、扫掠、I/O）

### 1.1 编译

```bash
uv run lbm-build              # → lbm_mrt/solver/mcmp_sim         （legacy MCMP）
uv run lbm-build --hydrate    # → lbm_mrt/solver/mcmp_sim_hydrate （水合物）
uv run lbm-build --huang      # → lbm_mrt/solver/mcmp_huang_256   （Huang SCMP，本套件用）
```

### 1.2 跑一个 case

```bash
# 单 case（手写 params.txt）：
uv run lbm-run --case-name laplace_R40 \
    --app lbm_mrt/solver/mcmp_huang_256 \
    --config configs/huang_scmp.yaml \
    huang_R0=40.0 cs_T=0.06 k1_huang=0.0
```

### 1.3 批量扫掠（推荐入口）

所有 sweep 走 `data/design_scmp_*.csv` + `lbm_mrt.runners.batch_run.run_batch`：

```bash
# 生成 design CSV：
uv run python scripts/06_run_huang_validation_suite.py --sweep laplace
# 跑批：
uv run lbm-batch --csv data/design_scmp_laplace.csv \
                 --app lbm_mrt/solver/mcmp_huang_256
# 后处理：
uv run python scripts/06_run_huang_validation_suite.py \
    --analyze results/design_scmp_laplace_<TIMESTAMP>
```

CSV 列名 → `params.txt` 键名 → `RuntimeParams` 字段名是 **一一对应** 的；只需把列名写对，`batch_run.py` 自动接住。

### 1.4 输出约定

每个 case 输出到 `results/<batch_tag>/<case_name>/`：
```
results/<batch>/<case_name>/
├── params.txt
├── log.txt
├── run_summary.txt        ← 已有（含 step、ρ_l/ρ_g、Q、status）
├── outputdata_scmp/
│   ├── flow_5000.vtk
│   ├── flow_10000.vtk
│   └── flow_scmp_final.vtk
└── ckpt/                   （SCMP 默认不启用断点）
```

后处理找最新 VTK：`lbm_mrt.io.vtk_reader.latest_vtk(<case_dir>/outputdata_scmp)`。

---

## §2 关键参数语义与坐标系映射

### 2.1 cs_T 是物理 T，不是 Tr（重要）

| 名义 | 实际语义 | 算法 |
|---|---|---|
| YAML / params.txt 中 `cs_T` | **绝对（物理）温度**，单位与 `cs_a/cs_b/cs_R` 自洽 | — |
| 论文/方法学里的 `Tr` | T 相对于临界温度的比值 T/Tc | 主机端用 `cs_eos.cs_critical_point(a,b,R)` 算 Tc，再 `cs_T = Tr * Tc` |

CS-EOS 临界温度：`Tc = 0.3773·a/(b·R)`。当前 `huang_scmp.yaml` 取 `cs_a=0.75, cs_b=4, cs_R=1` → **Tc ≈ 0.07074**。

> **本套件统一用 cs_a=1.0, cs_b=4.0, cs_R=1.0（Tc ≈ 0.09433），与 Huang & Wu 2016 paper 第 5 节惯例一致**。`huang_scmp.yaml` 默认值需要调整为 `cs_a: 1.0`；`design_scmp_laplace*.csv` 已存在的 `cs_a=0.75` 行随之更新或在新生成时覆盖。

Tr ↔ cs_T 换算表（cs_a=1.0, cs_b=4.0, cs_R=1.0；Tc=0.09433）：

| Tr | cs_T | 预期 ρ_l | ρ_g | ρ_l/ρ_g |
|---|---|---|---|---|
| 0.60 | 0.0566 | ~0.30 | ~0.001 | ~300 |
| 0.70 | 0.0660 | ~0.27 | ~0.005 | ~50 |
| 0.85 | 0.0802 | ~0.18 | ~0.027 | ~6.5 |
| 0.90 | 0.0849 | ~0.15 | ~0.040 | ~3.8 |
| 0.95 | 0.0896 | ~0.097 | ~0.069 | ~1.4 |

数值由 `cs_eos.maxwell_coexistence(1.0, 4.0, 1.0, T)` 给出；以上为参考量级，实际取计算结果。

### 2.2 共存密度注入机制

求解器侧 `init_all_scmp_gpu` 在 `huang_rho_g > 1e-8 && huang_rho_l > huang_rho_g` 时**优先用主机端注入值**作为 tanh 剖面两侧的渐近密度；否则用 GPU 内启发式 (`0.1304/b * (1±2.4·ΔTr^0.325)`)。

**所有 P0 套件 case 都应在 design CSV 中显式注入 `huang_rho_g`、`huang_rho_l`**，由 Python 端调用 `cs_eos.maxwell_coexistence()` 在生成 CSV 时算好；这样避免低 Tr 下启发式估计偏离真实共存值。

### 2.3 σ-旋钮 k1_huang

`σ ∝ (1 − 6·k₁_huang)`，工作范围 k₁ ∈ [0, 1/6)。`k₁ = 0` ⇒ 最大 σ；`k₁ → 1/6` ⇒ σ → 0。本套件默认 `k₁ = 1/12 ≈ 0.08333`（对应 σ 折半）。

> 论文 Eq. 62 还含 k₂ 项；本套件保持 `k2_huang = 0`、`kd_huang = -1/12`、`alpha_meq = 1.0` 不动。

---

## §3 Laplace 定律验证（对应论文 Figure S1）—— P0

### 3.1 物理目标
2D 圆形静液滴，验证 **ΔP = σ/R**（过原点）。R² ≥ 0.99 即通过。

### 3.2 几何与初始化
- 域：**256 × 256**（求解器编译期固定），全周期边界
- 液滴中心 (NX/2, NY/2) = (128, 128)
- 界面 tanh，厚度 `huang_W = 3`（保留求解器默认）
- (ρ_l, ρ_g) 由主机端 Maxwell 算好并通过 `huang_rho_g/l` 注入
- `huang_init_mode = 1`

### 3.3 扫描设计

| 参数 | 取值 |
|---|---|
| R（lattice units）| `[20, 25, 30, 35, 40, 45, 50, 55, 60]` —— 9 个点 |
| Tr | `0.70, 0.90`（Eq1 / Eq2 对照）|
| k₁ | `1/12 ≈ 0.0833` 固定 |

**起点取 R=20**（避免 R≤15 在低 Tr 强密度比下界面相对厚度过大）；终点 R=60（避开域边界影响：R/NX ≤ 0.234）。

### 3.4 收敛判据

SCMP 求解器目前**没有 SteadyMonitor**（其只为 legacy MCMP 流量收敛设计），通过**固定 `NSTEPS`** 跑完。Phase 2 smoke 测试已确认 `NSTEPS = 200000` 步对 R=40 的液滴足够收敛（液滴形态稳定 + max|u| < 1e-3）。

> 如果发现某些 case 50k 步内已稳，可在求解器侧加 SCMP 版 SteadyMonitor（基于 `max|u|` 或 `Σρ` 的相对变化）；但 **P0 不依赖此项**。

### 3.5 后处理（`lbm_mrt/validation/laplace_law.py`，待新增）

复用 `lbm_mrt/validation/analytical.py` 现有函数：

```python
# 单 case 提取 (R, ΔP)
fields, nx, ny = read_vtk_scalars(latest_vtk(f"{case_dir}/outputdata_scmp"))
center, R_meas = detect_interface_radius(fields["rho"], nx, ny)
p_in, p_out    = fit_pressure_inside_outside(fields["rho"], fields["pressure"], center, R_meas)
dP             = p_in - p_out

# 跨 R 拟合 σ：按 Tr 分组
for Tr, group in df.groupby("Tr"):
    sigma, r2, intercept = laplace_sigma(group["R_meas"], group["dP"])
```

输出：
- `<results_dir>/laplace_summary.csv`：每个 case 一行 (Tr, R_nominal, R_meas, dP, invR)
- `<results_dir>/laplace_fit.csv`：每个 Tr 一行 (Tr, sigma, R², intercept, n)

### 3.6 可视化（替换论文 figures_si/fig1_laplace_eq{1,2}.pdf）

参考 legacy `validation/surface/validation_r_eq1/post_laplace.py:169-246` 的双面板风格（直接移植，把按 κ groupby 改为按 Tr groupby）：

- 双面板（a）Tr=0.7（b）Tr=0.9
- X = 1/R，Y = ΔP；空心散点 + 过原点拟合直线（**强约束 intercept=0**，调用 `np.polyfit(deg=1)` 或更稳的最小二乘解析式）
- 右下信息框：σ（4 sig fig）+ R²（3 dec）
- 字体 Times New Roman bold，axes.linewidth=2.0
- 输出 `<results_dir>/fig1_laplace_eq1.pdf`、`fig1_laplace_eq2.pdf`

### 3.7 与 legacy MCMP 差异（移植要点）

| 原 MCMP | Huang SCMP |
|---|---|
| ρ_A + ρ_B 两个场 | 单 ρ 场 |
| sweep 变量是 κ | sweep 变量是 Tr |
| `post_laplace.py` 按 κ groupby | 按 Tr groupby |
| `display_results` 内液滴 mask 取 B 相 | 直接 `rho > 0.5·(ρ_l+ρ_g)` |
| 不含 σ-decoupling | 见 §4 |

---

## §4 σ-decoupling 加测（Huang 方法独有，论文 Eq. 62 验证）—— P0

### 4.1 物理目标
固定 (Tr, R)，扫 k₁ ∈ {0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 1/6}：
- σ(k₁) / σ(0) ≈ (1 − 6·k₁)，跨 k₁ R² ≥ 0.99
- ρ_l, ρ_g 跨 k₁ 漂移 < 1%（核心改进）

### 4.2 扫描设计

| 参数 | 取值 |
|---|---|
| k₁ | `[0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.16667]` —— 8 个点（design CSV 已就位）|
| Tr | 默认 `0.70`（强密度比下放大 σ-leak 信号）|
| R | 固定 `40`（精度甜区） |

### 4.3 后处理（`lbm_mrt/validation/decoupling_sweep.py`，待新增）

```python
for k1, case_dir in cases:
    rho_l_obs, rho_g_obs = extract_plateau(fields["rho"])  # 复用 analytical.extract_rho_l_g
    sigma_k1 = (p_in - p_out) * R_meas                     # 单点估 σ（不做 1/R 扫，仅一个 R）
    rows.append((k1, sigma_k1, rho_l_obs, rho_g_obs))

# 验证两件事：
# (a) σ vs (1−6k₁) 过原点线性拟合，slope ≈ σ_max, R² ≥ 0.99
# (b) ρ_l std/mean < 0.01，ρ_g std/mean < 0.01
```

输出 `decoupling_summary.csv` (k1, sigma, rho_l, rho_g) + `fig_decoupling.pdf`（双面板：σ-k₁ 直线 / ρ_l ρ_g 漂移条形图）。

### 4.4 与套件其他模块的关系

- 该测试的 σ 估计是 **单 R 估计**（精度 ~5%），不替代 §3 的多 R 拟合
- 但需 §3 Laplace 拟合的 `sigma_max(Tr=0.7, k₁=0)` 作为归一基准，所以执行顺序 §3 → §4

---

## §5 共存曲线（Coexistence Curve）—— P0

### 5.1 物理目标
平界面初始化、Tr 扫描，提取数值 (ρ_l(Tr), ρ_g(Tr))，与 Maxwell 理论曲线比较。`|Δρ|/ρ < 2%` for Tr ∈ [0.6, 0.95]。

### 5.2 扫描设计

| 参数 | 取值 |
|---|---|
| Tr | `[0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]` —— 8 个点 |
| 域 | 256×256，`huang_init_mode = 2`（平界面）|
| 初始位置 | `huang_yc = 64`（界面位于 y=64），`huang_W = 5`（界面厚度稍大，便于平界面松弛） |
| k₁ | `1/12` 固定（共存与 k₁ 弱耦合，做基线即可）|

> 当前 06 脚本里 `generate_coexistence_design` 用绝对 T（[0.04, 0.045, …, 0.068]，假设 cs_a=0.75）；切换 cs_a=1.0 后需重算这些 T 值。建议改写为接受 Tr 列表，内部用 `cs_eos.cs_critical_point` 换算。

### 5.3 后处理（已在 `lbm_mrt/validation/coexistence.py`，需补 driver）

```python
# 已存在：extract_flat_interface_profile / identify_coexistence_point / compare_coexistence_curves
# 待补 driver 函数 run_coexistence_sweep + analyze_flat：
for case_dir in case_dirs:
    coord, profile, _ = extract_flat_interface_profile(latest_vtk(...), direction="y")
    rho_g_obs, rho_l_obs = identify_coexistence_point(profile)
    records.append((Tr, rho_g_obs, rho_l_obs))

# 与 Maxwell 比较：
maxwell = coexistence_curve(1.0, 4.0, 1.0, Tr_list * Tc)
plot_coexistence_compare(maxwell, records, out=fig3_coexistence.pdf)
```

### 5.4 可视化

- X = ρ（log scale），Y = Tr —— 标准的 phase envelope
- 实线 = Maxwell（cs_eos.coexistence_curve）
- 散点 = 数值
- 已有 `coexistence.py::compare_coexistence_curves` 可直接调用，仅需补一个 driver

输出：`fig3_coexistence.pdf`。

---

## §6 静态液滴 Spurious Currents 衰减 —— P0

### 6.1 物理目标
静态平衡液滴上 max|u| 应随 NY 升高而下降，并显著低于 Li (2013) 方案（Huang 2016 Fig. 4 的核心宣称）。

### 6.2 扫描设计

| 参数 | 取值 |
|---|---|
| NY（= NX，方域）| `[64, 96, 128, 192, 256]` —— 5 个点 |
| R / NY | 固定 `0.2`（NY=256 → R=51） |
| Tr | `0.70`（强密度比放大 spurious） |
| k₁ | `1/12` |

> **关键约束**：求解器编译期 `NX=NY=256` 是写死的（`LBM.h:20-27`），所以本扫描 **需要新增编译期注入机制**：
> - 在 `build.py` 中加 `--huang-grid NY` 参数 → 传 `-DNY_VAL=...` → `LBM.h` 中条件 `#ifndef NY_VAL ... #endif`
> - 或者：每个 NY 编译一个 `mcmp_huang_<NY>` 二进制（参考 legacy `validation/mesh_convergence/run_mesh_study.sh`）
> 建议后者，与 §9 网格收敛共用同一套机制。**这是 P0/P1 边界**：若该机制暂未实现，先仅做 NY=256 单点 spurious 提取（保留入口，标注为后续扩展）。

### 6.3 后处理（`lbm_mrt/validation/spurious_currents.py`，待新增）

```python
fields, nx, ny = read_vtk_scalars(latest_vtk(case_dir + "/outputdata_scmp"))
u_mag = np.sqrt(fields["U"][...,0]**2 + fields["U"][...,1]**2)  # VTK 写的是 U 矢量
max_u  = float(np.max(u_mag))
mean_u = float(np.mean(u_mag))
```

输出 `spurious_summary.csv` (NY, R, max_u, mean_u, rho_l_obs, rho_g_obs)。

### 6.4 对比基准（Huang vs Li）

可选项：用同样几何在 `lbm_mrt/solver/mcmp_sim`（legacy Li 方案）上跑一遍同样 NY 序列，画 Huang 蓝 / Li 红两条 log-log 曲线。预期 Huang 在所有 NY 下都低 2–5×。

> Li 方案需要 MCMP 双场跑单组分极限（A 占主导、B≈0、`GAB=GBA=0`、`sigmaA` 给值），不是完全干净的对比。**列为可选，先做 Huang 单条曲线**。

### 6.5 可视化

X = NY（log），Y = max|u|（log）；Huang 蓝实线（5 个点）、Li 红虚线（5 个点，可选）。输出 `fig_spurious.pdf`。

---

## §7 接触角验证（对应论文 Figure S2）—— P1（先扩展求解器）

> ⚠ **求解器侧前置工作**：当前 `mcmp_huang_256` 全域周期 BC、无 wall flag、无 Yang/Li 润湿力。本节前 3 项必须先做：

### 7.1 求解器扩展（Phase 5a）

1. **几何初始化扩展**：`init_all_scmp_gpu` 增加 wall flag 写入逻辑 —— 底壁 y=0,1 设 `pointsflag = −1`（wall ghost）、其余流体 `pointsflag = 1`；顶部和左右仍周期。复用 legacy `validation/contact_angle/contact_eq1/LBM.cu:906` 的模式。
2. **wall_mat 表与 G_ads 查表**：参考 `lbm_mrt/solver/src/sim_utils.cu::push_wettability_and_maps` 把 `GAw_by_mat_gpu[256]` 接到 SCMP；material id = 1 = quartz、`GAw_quartz = G_ads_input`（来自参数）。
3. **润湿力 kernel**：新增 `compute_adsorption_force_scmp`：`F_ads(x) = -G_ads · ψ(x) · Σ_k w_F[k] · s(x+e_k) · e_k`（s=1 if wall else 0）。加在 `compute_molecular_force_scmp` 之后、`compute_S_huang_gpu` 之前。
4. **域形变**：300 × 100 channel —— 同样需 §6 的 `NX/NY` 编译期注入机制。

### 7.2 几何与扫描设计（求解器扩展完成后）

| 参数 | 取值 |
|---|---|
| 域 | **300 × 100**，底壁 bounce-back，顶/左/右周期 |
| 液滴 | r=30，圆心 (150, 1) 紧贴底壁，`huang_init_mode=1`，`huang_xc=150`、`huang_yc=1`、`huang_R0=30`、`huang_W=4` |
| G_ads | `seq -0.15 0.01 0.15` —— 31 个 case（与 legacy `contact_eq1/run_cases.sh:5` 一致）|
| Tr | `0.70 / 0.90` 双对照 |
| k₁ | `1/12` |
| NSTEPS | Tr=0.7 → 50000；Tr=0.9 → 20000 |

### 7.3 接触角测量算法（Python 端，移植 legacy `contact_angels.py`）

```python
# 1. 读 VTK，取 rho 场，reshape (NY, NX)
rho = fields["rho"].reshape((ny, nx), order="C")

# 2. 阈值 thr = 0.5*(ρ_max + ρ_min)，提取 y 方向向下穿越点
pts = []
for j in range(ny - 1):
    for i in range(nx):
        if rho[j, i] >= thr and rho[j+1, i] < thr:
            pts.append([i, j + 0.5])

# 3. 过滤接近壁面（y < 5）的点（消除底部边缘伪界面）
# 4. 最小二乘拟合圆：x² + y² + A·x + B·y + C = 0
# 5. yc = -B/2，R = sqrt((A²+B²)/4 - C)
# 6. yc < 0  → θ = arccos(|yc|/R)         （锐角、亲液）
#    yc ≈ 0  → θ = 90°
#    yc > 0  → θ = 180° − arccos(yc/R)    （钝角、疏液）
#    务必 clip |yc|/R 到 [0, 1]
```

新模块：`lbm_mrt/validation/contact_angle.py`，输出 `contact_angle_vs_GAw.csv` (GAw, theta) + `fit_debug_<GAw>.png`（调试用）+ `fig2_contact_angle_eq{1,2}.pdf`。

### 7.4 局限性声明（SI 文本必须如实记录）

Huang & Wu (2016) paper **未给出** SCMP 接触角机制；本套件复用 Yang(2021) / Li(2019) 风格的伪势附加外场 G_ads·ψ。隐含假设：θ 由 (ρ_l, ρ_g, G_ads) 决定，与 k₁ 正交。**额外测试**：固定 G_ads 扫 k₁ ∈ {0, 0.05, 0.10, 0.15}，观察 θ 是否漂移 < 2°；若否，需在论文 SI 中说明此机制与 Q_m 的弱耦合。

---

## §8 单相 Poiseuille 速度剖面（对应论文 Figure S4）—— P1（先扩展求解器）

### 8.1 求解器扩展（Phase 5b）

1. **wall flag**：顶 (y=NY-1,NY-2) + 底 (y=0,1) bounce-back ghost；左右周期。`huang_init_mode = 3`（新增：均匀液相 `ρ = ρ_l`，u=0）。
2. **均匀体力注入**：`compute_molecular_force_scmp` 增加 `Fx += d_Gx * rho`、`Fy += d_Gy * rho`，或单独 kernel `add_body_force_scmp`。已有的 `d_Gx, d_Gy` 设备常量复用。
3. **流量监测**：SCMP 路径加最小版 SteadyMonitor —— 仅监 `Q_total = Σ rho·ux`，相对容差判稳。

### 8.2 几何与解析解

- 域 **300 × 200**（顶底 bounce-back，b = (NY−2)/2 = 99）
- F_b = `Gy = 5e-9`（与 legacy `validation/twophase/validation_v_eq1/LBM.h:89` 一致）
- 解析：`u(y) = F_b·(b² − y²) / (2·ρ_l·ν)`，ν = (1/3)·(τ_p − 0.5)，默认 τ_p=1 → ν=1/6

### 8.3 两阶段收敛（参考 legacy `mesh_convergence/main.cu:89-148`）

```
Stage 1 (equilibrate, drive_scale=0):  tol_rel=1e-6, need_consec=5, max_steps≥10000
Stage 2 (drive,       drive_scale=1):  tol_rel=5e-5, need_consec=5, monitor Q_total
```

> StageConfig 在 `sim_utils.h` 已存在；只需 SCMP 版 `run_stage_scmp` 调度，参考 `run_stage` 的结构。

### 8.4 后处理（`lbm_mrt/validation/poiseuille_sp.py`，待新增）

```python
fields, nx, ny = read_vtk_scalars(latest_vtk(...))
ux = fields["U"][..., 0].reshape((ny, nx), order="C")
u_profile = ux[:, nx // 2]              # x=NX/2 列
y         = np.arange(ny) - (ny - 1) / 2.0
b         = (ny - 2) / 2.0
u_an      = np.where(np.abs(y) <= b, Fb * (b**2 - y**2) / (2 * rho_l * nu), 0.0)
r2        = 1 - np.sum((u_profile - u_an)**2) / np.sum((u_profile - u_profile.mean())**2)
```

双 Tr 对照（Tr=0.7、Tr=0.9 → 不同 ρ_l → 不同 U_max）。

### 8.5 可视化

X = y/b（归一化）、Y = u/U_max（归一化）；实线 = 解析、散点 = LBM；右上 `R² = 0.99XX, Re = XX`。输出 `fig4_poiseuille_eq{1,2}.pdf`。

> SI 文本同步：S2.3 段从"two-phase Poiseuille / relative permeability"改写为"single-phase Poiseuille velocity-profile validation"；删除 k_rw/k_rnw 段落。

---

## §9 网格收敛（Poiseuille ε vs NY）—— P1

### 9.1 前置（与 §6、§8 共用）

需要 §6 / §8 中讨论的 **编译期 NX/NY 注入机制**。建议 `build.py` 增加：
```bash
uv run lbm-build --huang --grid 100   # → mcmp_huang_100
uv run lbm-build --huang --grid 200   # → mcmp_huang_200
uv run lbm-build --huang --grid 400   # → mcmp_huang_400
uv run lbm-build --huang --grid 800   # → mcmp_huang_800
```
内部传 `-DHUANG_NX=$NY -DHUANG_NY=$NY`，`LBM.h` 用 `#ifdef HUANG_NX` 重定义 NX/NY/NSTEPS/NOUTPUT。

### 9.2 扫描设计

| 参数 | 取值 |
|---|---|
| NY ∈ {NX} | `[100, 200, 400, 800]` |
| F_b 缩放 | `Fb(NY) = Fb_ref · (h_ref / h(NY))³`，Fb_ref=5e-9, h_ref=198（NY=200 基准） |
| NSTEPS | `[8e6, 12e6, 20e6, 35e6]`（粗快、细慢；总 GPU 时长约 4–5 小时） |
| Tr | `0.70` 单 case 即可（§9 关心数值 order，不关心 Tr 对照） |
| ν, τ | 固定 τ_p=1 → ν=1/6，跨 NY 不变 |

### 9.3 后处理（`lbm_mrt/validation/mesh_convergence.py`，待新增）

直接移植 legacy `validation/mesh_convergence/mesh_convergence.py`（839 行，含 Richardson / GCI）：
- 从 `<case_dir>/run_summary.txt` 读 `Q_LBM`, `Q_analytical`, `steady_step`, `status`
- ε(NY) = |Q − Q_an| / |Q_an|
- log-log 拟合斜率 ≈ −p_obs（理论 MRT p=2）
- 判据：NY=200 ε < 1% → SUFFICIENT；1–5% → MARGINAL；>5% → INSUFFICIENT

### 9.4 可视化（与 §6 合并成双子图）

```
fig5_mesh_convergence.pdf:
  ┌──────────────────────────┐  ┌──────────────────────────┐
  │ (a) NY vs ε(Poiseuille)  │  │ (b) NY vs max|u| spurious│
  │ log-log, 4 点 + 拟合斜率│  │ log-log, 5 点 (Huang)    │
  │  Eq1（Tr=0.7）           │  │ + 可选 Li 对比           │
  └──────────────────────────┘  └──────────────────────────┘
```

---

## §10 一键脚本扩展（`scripts/06_run_huang_validation_suite.py`）

### 10.1 当前已支持
```bash
uv run python scripts/06_run_huang_validation_suite.py \
    --sweep {laplace,decoupling,coexistence} [--run]
uv run python scripts/06_run_huang_validation_suite.py \
    --analyze results/<batch_dir>
```

### 10.2 P0 阶段扩展

```python
# 改造 _base_params(): 默认 cs_a=1.0 (替换当前 0.75)
# 改造 generate_laplace_design():
#   - 接受 Tr_list 参数；为每个 Tr 调 maxwell_coexistence 算 (ρ_l, ρ_g)，注入到行里
#   - case_name: "laplace_Tr{Tr:.2f}_R{R:.0f}"
#   - rows 在 9 R × 2 Tr = 18 个
# 新增 generate_spurious_design():
#   - 5 NY × 单 Tr = 5 个 case
#   - 等求解器支持多 NY 后才能跑；P0 阶段先生成"NY=256 单点"占位
# 完善 analyze_laplace():
#   - 按 Tr groupby，每个 Tr 单独拟合，写 laplace_fit.csv（Tr, sigma, R², n）
# 新增 analyze_decoupling(): σ vs (1−6k₁) + ρ_l/ρ_g 漂移检测
# 新增 analyze_spurious(): max|u|(NY) 提取
```

### 10.3 P1 阶段扩展（求解器扩展完成后）

```python
# generate_contact_angle_design()  → 2 Tr × 31 GAw = 62 case
# generate_poiseuille_design()     → 2 Tr × 1 = 2 case
# generate_mesh_design()           → 4 NY × 1 Tr = 4 case
# analyze_contact_angle() / analyze_poiseuille() / analyze_mesh()
```

### 10.4 一键运行（最终形态）

```bash
uv run python scripts/06_run_huang_validation_suite.py --all \
    --Tr-list 0.7,0.9 --k1 0.08333 \
    --out-root results/huang_validation_$(date +%Y%m%d_%H%M)
```

产生 `<out>/REPORT.md`，含通过判据表 + 嵌入图。

---

## §11 通过判据汇总

| 测试 | 通过判据 | 严格通过 |
|---|---|---|
| §3 Laplace（双 Tr）| R² ≥ 0.99 each | R² ≥ 0.999 |
| §4 σ-decoupling | σ-(1−6k₁) R² ≥ 0.99 AND ρ_l/ρ_g std/mean < 1% | drift < 0.5% |
| §5 共存曲线 | \|Δρ\|/ρ < 2% for Tr ∈ [0.6, 0.95] | < 1% |
| §6 Spurious | max\|u\|(Huang) ≤ 0.5·max\|u\|(Li)（若做对比）| ≤ 0.2× |
| §7 接触角 | θ-G_ads 线性 R² ≥ 0.95，θ 覆盖 30°–150° | 20°–160° |
| §8 Poiseuille | u(y) vs analytical R² ≥ 0.99 | R² ≥ 0.999 |
| §9 网格收敛 | ε(NY=200) < 1.5%，p_obs ∈ [1.5, 2.5] | ε < 1%，p_obs ≈ 2.0 |

---

## §12 关键文件路径速查表

### 12.1 已就位（直接使用）

| 文件 | 作用 |
|---|---|
| `lbm_mrt/solver/mcmp_huang_256` | Huang SCMP 二进制（P0 全部依赖）|
| `lbm_mrt/solver/build.py` | `--huang` 编译入口 |
| `lbm_mrt/validation/cs_eos.py` | Maxwell 共存（`cs_critical_point`、`maxwell_coexistence`、`coexistence_curve`）|
| `lbm_mrt/validation/analytical.py` | 后处理工具（`detect_interface_radius`、`fit_pressure_inside_outside`、`laplace_sigma`、`centerline_profile`、`extract_rho_l_g`）|
| `lbm_mrt/validation/coexistence.py` | 平界面剖面提取 + Maxwell 对比绘图 |
| `lbm_mrt/runners/batch_run.py` | `run_batch(design_csv, app, …)` 通用扫掠驱动 |
| `lbm_mrt/io/vtk_reader.py` | `latest_vtk()` / `read_vtk_scalars()` |
| `lbm_mrt/viz/viz_template.py` | 论文级 matplotlib 模板 |
| `scripts/06_run_huang_validation_suite.py` | 套件入口（部分子命令已实现）|
| `configs/huang_scmp.yaml` | SCMP 默认配置（**需把 cs_a 从 0.75 改为 1.0**）|
| `data/design_scmp_laplace.csv` | 单 Tr 7-R 设计（待扩展到 18 行：2 Tr × 9 R） |
| `data/validation_reference/huang_2016/cs_coexistence.json` | Maxwell 参考曲线 |
| `tests/test_huang_scmp.py` | Maxwell 单元测试（108 测试通过） |

### 12.2 待新增（P0）

| 文件 | 主要函数 |
|---|---|
| `lbm_mrt/validation/laplace_law.py` | `run_laplace_sweep`、`analyze_laplace`、`plot_laplace_double_panel` |
| `lbm_mrt/validation/decoupling_sweep.py` | `run_decoupling_sweep`、`analyze_decoupling`、`plot_decoupling` |
| `lbm_mrt/validation/spurious_currents.py` | `extract_max_u`、`analyze_spurious`、`plot_spurious` |

`scripts/06_run_huang_validation_suite.py` 扩展：
- 改造 `_base_params()`：cs_a=1.0
- 改造 `generate_laplace_design()`：接收 `Tr_list`，自动注入 `huang_rho_g/l`
- 新增 `generate_spurious_design()`
- 新增 `analyze_decoupling()`、`analyze_spurious()`、`make_report_md()`

### 12.3 待新增（P1，依赖求解器扩展）

| 文件 | 主要函数 |
|---|---|
| `lbm_mrt/validation/contact_angle.py` | `fit_circle_lstsq`、`compute_theta_from_vtk`、`run_contact_sweep`、`analyze_contact` |
| `lbm_mrt/validation/poiseuille_sp.py` | `extract_centerline`、`fit_parabola`、`analyze_poiseuille` |
| `lbm_mrt/validation/mesh_convergence.py` | 移植 legacy 同名文件（839 行），保留 Richardson / GCI |

求解器侧（CUDA）扩展：
- `build.py` 增加 `--grid N` → `-DHUANG_NX=N -DHUANG_NY=N`（编译多分辨率二进制）
- `LBM.cu` 增 `huang_init_mode = 3`（均匀液相），`compute_adsorption_force_scmp`、`add_body_force_scmp`
- `sim_utils.cu` 增 wall flag 写入、`run_stage_scmp` 两阶段调度

### 12.4 论文 figures_si/ 替换对照

| 原 MCMP | Huang SCMP |
|---|---|
| `fig1_laplace_eq{1,2}.pdf` | 重生成（Tr=0.7/0.9） |
| `fig2_contact_angle_eq{1,2}.pdf` | 重生成（Tr=0.7/0.9） |
| `fig3_coexistence.pdf`（若有） | 重生成 |
| `fig4_relperm_eq{1,2}.pdf` | **改名 `fig4_poiseuille_eq{1,2}.pdf`**（单相剖面，SI 文本 §S2.3 同步改） |
| `fig5_mesh_convergence.pdf` | 重生成（双子图：Poiseuille 收敛 + spurious 衰减） |

`Supporting_Information.tex` 同步修订要点：
- §S2.1：「T=0.85·T_c, surface tension was tuned via κ」→「T/T_c=0.7 (Eq1), 0.9 (Eq2); surface tension was set via k₁=1/12 in the Huang Q_m scheme」
- §S2.3：两相 Poiseuille / 相渗 → 单相 Poiseuille 速度剖面验证；删除 k_rw/k_rnw 段
- §S2.4：网格收敛末尾追加 spurious currents 子图说明

---

## §13 验证（Verification）

```bash
# 0. 求解器基线（应已绿）
uv run lbm-build                 # legacy mcmp_sim 不回归
uv run lbm-build --huang         # mcmp_huang_256 编译通过
uv run pytest tests/             # 现有测试不回归

# 1. CS-EOS Maxwell（已通过，sanity check）
uv run python scripts/visualize_cs_coexistence.py
#   预期：Tr=0.7 给出 ρ_l ≈ 0.27、ρ_g ≈ 0.005（cs_a=1.0）

# 2. P0：四个 sweep 全做
uv run python scripts/06_run_huang_validation_suite.py --sweep laplace      --run
uv run python scripts/06_run_huang_validation_suite.py --sweep decoupling   --run
uv run python scripts/06_run_huang_validation_suite.py --sweep coexistence  --run
# spurious 需求解器 multi-grid 支持，否则只跑 NY=256 单点

# 3. 单项 sanity 检查
ls results/design_scmp_laplace_*/laplace_fit.csv          # 双 Tr 行，R² > 0.99
ls results/design_scmp_decoupling_*/decoupling_summary.csv  # ρ_l/ρ_g 漂移 < 1%
ls results/design_scmp_coexistence_*/coexistence_summary.csv # 8 Tr 点
ls results/design_scmp_laplace_*/fig1_laplace_eq{1,2}.pdf

# 4. P1：求解器扩展完成后
# (扩展 build.py / LBM.cu / sim_utils.cu 见 §7.1 / §8.1)
uv run lbm-build --huang --grid 100
uv run lbm-build --huang --grid 200
uv run lbm-build --huang --grid 400
uv run lbm-build --huang --grid 800
uv run python scripts/06_run_huang_validation_suite.py --sweep contact_angle  --run
uv run python scripts/06_run_huang_validation_suite.py --sweep poiseuille     --run
uv run python scripts/06_run_huang_validation_suite.py --sweep mesh           --run

# 5. 替换论文图
cp results/huang_validation_*/fig*.pdf figures_si/   # 先备份原 MCMP 版
pdflatex Supporting_Information.tex
```

### 13.1 已知坑

1. **Tr=0.7 下 ρ_g ≈ 0.005 太稀**：小 R（< 20）下界面相对厚度（W=3）过大 → R_meas 误差。**对策**：Laplace 扫描 R 起点取 20。
2. **k₁ → 1/6 时 σ → 0**：液滴可能蒸发/变形剧烈。`decoupling_design.csv` 把 1/6 替换为 0.15 或 0.16 较安全。
3. **接触角极端 GAw（±0.15）**：液滴脱壁/铺平时圆拟合失败 → `compute_theta_from_vtk` 在界面点 < 10 时返回 `NaN`，绘图时跳过这些点。
4. **NY=800 单 case ~1 小时**：35M 步 × ~0.1 ms/step。显存 800² × 19 × 8 byte ≈ 100 MB，绰绰有余。
5. **替换 figures_si/ 前先备份**到 `figures_si_mcmp_backup/`，便于审稿时对比。
6. **cs_a 不一致风险**：当前 `huang_scmp.yaml` 用 cs_a=0.75（Tc=0.0707），但本套件统一 cs_a=1.0（Tc=0.0943）—— 切换时务必同步：
   - `configs/huang_scmp.yaml`：`cs_a: 1.0`
   - `scripts/06_run_huang_validation_suite.py::_base_params()`：`"cs_a": 1.0`
   - 所有现存 `data/design_scmp_*.csv`：要么重生成，要么手改 `cs_a` 列
   - `tests/test_huang_scmp.py`：用 `A=1.0, B=4.0, R=1.0` 跑 Maxwell，已就绪
7. **`huang_rho_g/l` 必须从主机端 Maxwell 注入**（§2.2）—— 否则 GPU 内启发式估计在低 Tr 偏离真实值 5–20%，导致初始密度场就在 spinodal 区域，前几千步剧烈漂移。

---

## §14 使用本文档时的注意事项

- **本文档只指定方法（What / Why）**，不指定代码改动行号；HOW 留给实现阶段（参考 `change_plan.md` Phase 5）
- **P0 / P1 分界严格**：P0 全部可在现有求解器上跑；P1 必须先扩展 SCMP 求解器（接 wall + 体力 + 润湿力 + 多分辨率编译）
- **σ-decoupling（§4）和 spurious 子图（§6）是 Huang 相对 Li 的优势凸显**，建议在论文 main text discussion 中也补一笔
- **接触角测试（§7.4）的局限性要在 SI 中如实写**：Huang 论文未给 SCMP 接触角机制，本套件复用 Yang/Li 风格附加外场 G_ads·ψ 并以测试验证；这是诚信问题，不能藏
- **路径全部以本仓库为绝对参照**，不再涉及"另一仓库"或外部脚本；所有 `validation/` 引用都指本仓库根的 legacy 参考目录
