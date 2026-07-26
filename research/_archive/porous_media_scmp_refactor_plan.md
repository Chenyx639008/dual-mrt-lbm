# Huang SCMP 多孔介质单相渗流改造计划 v3

> 日期: 2026-05-24 (修订版 v3)
> 基于: `v20260524-validation-complete` + 讨论反馈
> 目标: 读取 `.plt` 几何文件，用 Huang SCMP 求解器运行纯水单相渗流模拟并测量流量

---

## 〇、背景：两条路径对比

| 特性 | MCMP 路径 (`mcmp_sim`) | SCMP 路径 (`mcmp_huang_*`) |
|------|------------------------|---------------------------|
| 流体 | 双组分 (A=水, B=气) | 单组分 (纯物质气-液) |
| EOS | Peng-Robinson (水) + 理想气体 (气) | Carnahan-Starling |
| 润湿控制 | `GAw_by_mat_gpu[256]` 材质查表 (θ→GAw) | Scheme IV ψ-ghost BC (每 ghost 节点独立 θ) |
| 几何输入 | ✅ `.plt` 文件 + `build_circle_array` | ❌ 仅内置 mode 1-4 |
| 稳态检测 | ✅ `SteadyMonitor` + 两阶段 | ❌ 固定 NSTEPS 循环 |
| 断点续跑 | ✅ checkpoint 机制 | ❌ |
| 网格 | 339×212 (硬编码) | 支持 `-DHUANG_NX=...` 覆盖 |

---

## 一、改造目标

### 1.1 物理场景

| 要素 | 设置 |
|------|------|
| 流体 | **纯水单相**，占据全部孔隙空间 |
| 固体骨架 | 来自 `.plt` 文件（0=孔隙, 1=固体） |
| 水合物 | 来自 `.plt` 文件（若有 phase=0.5 标记为水合物固体），作为第二种固体 |
| 驱动力 | 体积力 $G_x$（压力梯度等效），x 方向周期性 |
| 壁面边界 | y=0 和 y=NY-1：无滑移壁面（ghost bounce-back） |
| 流动状态 | 稳态 Darcy 渗流 |
| 输出关注 | 速度场 $u_x, u_y$、密度 $\rho$、稳态流量 $Q$ |

### 1.2 单相实现方案：均匀密度 + 保持 cs_G = -1.0

**核心思路**：不需要修改 `cs_G`，不需要超临界退化。利用均匀密度初始化的自然性质。

| 参数 | 设置 | 物理效果 |
|------|------|----------|
| `cs_G` | **-1.0**（保持已验证值） | 分子力公式 $F_{mol} = -G\psi\nabla\psi$ 保持完整 |
| `huang_rho_l = huang_rho_g` | 均匀液态水密度 | 全域密度均匀 |
| `huang_init_mode` | 5 (porous media) | 孔隙区域初始化为液态水密度 |

**为什么均匀密度 + G=-1 就能实现单相流**：

1. 全域初始化为液态水共存密度 → 密度场均匀 → $\nabla\psi = 0$
2. $F_{mol} = -G\psi\nabla\psi \equiv 0$（梯度为零，与 $G$ 值无关）
3. 无密度梯度 → 无相分离驱动力 → 系统保持单相
4. 所有 kernel 使用已验证的 `cs_G = -1.0`，无需任何特殊分支

**与 cs_G=0 方案对比**：

| 方面 | cs_G = 0 | cs_G = -1.0（采用） |
|------|----------|---------------------|
| ψ 计算 | 除零，需保护分支 | 正常，已验证 |
| Q_m 项 | 除零，需保护分支 | 正常（但 $F_{mol}=0$ 故 $Q_m=0$） |
| 压力场 | $p=\rho c_s^2$（理想气体） | $p=\rho c_s^2 + \frac{1}{2}c^2 G\psi^2$（CS EOS 一致） |
| Scheme IV | ghost ψ ≈ 0 无法工作 | ghost ψ 有意义 → 接触角 BC 正常 |
| 代码改动 | 需多个 kernel 增加 G=0 分支 | **零改动** |

**结论**：保持 `cs_G = -1.0` 是最优方案——零额外代码、全 kernel 已有验证覆盖、Scheme IV 润湿性天然可用。

---

## 二、涉及文件清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `lbm_mrt/solver/include/LBM.h` | 修改 | `#ifdef` 守卫扩展 + per-material θ 数组 |
| `lbm_mrt/solver/src/main.cu` | 修改 | 透传 `geom_file` 到 SCMP 路径 |
| `lbm_mrt/solver/src/sim_utils.cu` | **重写** `run_scmp_huang` | 从验证循环升级为完整渗流管线 |
| `lbm_mrt/solver/src/LBM.cu` | 新增/修改 | mode 5 初始化 + 几何标记 kernel + per-material Scheme IV |
| `lbm_mrt/solver/src/steady_monitor.cu` | 新增 | SCMP 专用 `compute_Q_scmp` |
| `lbm_mrt/solver/include/steady_monitor.cuh` | 修改 | 声明 `compute_Q_scmp` |
| `lbm_mrt/solver/build.py` | 修改 | 从 .plt 自动检测 I×J，设置 `-DHUANG_NX=` `-DHUANG_NY=` |
| `configs/` | 新建 | `single_phase_water_scmp.yaml` |

> **与 v2 相比**：去掉了 `lbm_unit_conversion.py`（物性匹配不在 Phase 1 范围内），去掉了 cs_G=0 数值保护阶段。

---

## 三、详细改造方案

### Phase 1: 编译层面 — 从 .plt 自动检测网格尺寸

#### 3.1.1 设计思路

`.plt` 文件头部包含网格信息：
```
ZONE t = "solid" I = 300, J = 300 F = point
```

**方案**：在 `build.py` 中解析 .plt 头部的 `I=` 和 `J=`，自动设置 `-DHUANG_NX=<I>` `-DHUANG_NY=<J>`。这样：
- C++ 代码中 NX/NY 仍然是编译期常量（保持性能）
- 用户不需要手动指定网格尺寸
- `read_tecplot_to_flag` 中可增加一致性校验（数据行数 vs NX×NY）

#### 3.1.2 `LBM.h` 修改

在 `HUANG_256_BUILD` 块之后增加 `HUANG_POROUS_BUILD`：

```cpp
#elif defined(HUANG_POROUS_BUILD)
#ifndef HUANG_NX
#error "HUANG_POROUS_BUILD requires -DHUANG_NX=<N>"
#endif
#ifndef HUANG_NY
#error "HUANG_POROUS_BUILD requires -DHUANG_NY=<N>"
#endif
constexpr unsigned int SCALE   = 1;
constexpr unsigned int NX      = HUANG_NX;
constexpr unsigned int NY      = HUANG_NY;
constexpr unsigned int NSTEPS  = 2000000;
constexpr unsigned int NOUTPUT = 20000;
```

#### 3.1.3 `#ifdef` 守卫全局修改

所有 `#ifdef HUANG_256_BUILD` 改为 `#if defined(HUANG_256_BUILD) || defined(HUANG_POROUS_BUILD)`：

- `main.cu`（SCMP dispatch 入口，约第 53 行）
- `sim_utils.cu`（`run_scmp_huang` 定义，约第 1104 行）
- `LBM.cu`（所有 SCMP kernel：`init_all_scmp_gpu`, `evolution_scmp`, `boundary_scmp_gpu`, `compute_rho_from_fin_gpu`, `compute_p_psi_scmp_cs`, `compute_molecular_force_scmp`, `compute_Q_huang_gpu`, `compute_adsorption_force_scmp`, `update_ghost_psi_bc`, `compute_velocity_scmp`, `compute_S_huang_gpu`, `mrt_collide_single_component_gpu`, `stream_single_component_gpu`, `compute_pressure_tensor_scmp`, `outputvtk_scmp` 等，约 15 处）
- `LBM.h`（与 HYDRATE_ENABLE 互斥守卫：`#if (defined(HUANG_256_BUILD) || defined(HUANG_POROUS_BUILD)) && defined(HYDRATE_ENABLE)`）

#### 3.1.4 `build.py` 修改

```python
def detect_plt_dimensions(plt_path: str) -> tuple[int, int]:
    """Parse I= and J= from .plt ZONE header line."""
    import re
    with open(plt_path) as f:
        for line in f:
            m = re.search(r'I\s*=\s*(\d+).*J\s*=\s*(\d+)', line)
            if m:
                return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot detect I,J from {plt_path}")
```

编译命令：
```bash
uv run lbm-build --porous --plt data/geometry/for_lbm/geometry_case0000.plt
# → 自动检测 300×300，传入 nvcc -DHUANG_POROUS_BUILD -DHUANG_NX=300 -DHUANG_NY=300
```

---

### Phase 2: 几何管线 — .plt 读取 + 边界标记 + 材质映射

#### 3.2.1 标志位约定

| flag 值 | 含义 | 来源 |
|---------|------|------|
| 1 | 流体（孔隙水） | .plt phase=0 |
| 0 | 边界（bounce-back 节点） | `mark_geometry_scmp` 计算 |
| -1 | ghost（固体虚拟层） | `mark_geometry_scmp` 计算 |
| -2 | 固体内部（石英或水合物） | .plt phase=1 (石英) / phase=0.5 (水合物) |

#### 3.2.2 材质映射表

复用 MCMP 已有的 `d_wall_mat` 设备指针（在 `#ifdef LBM_DEFINE_GLOBALS` 中已声明，SCMP 编译时同样可见）：

```c
__device__ unsigned char* d_wall_mat = nullptr;  // 0=无, 1=石英, 2=水合物
```

> 该变量在 LBM.h 的 `#ifdef LBM_DEFINE_GLOBALS` 块中声明。`LBM.cu` define 了 `LBM_DEFINE_GLOBALS`，因此 SCMP 路径下 `d_wall_mat` 同样可用。但当前 SCMP 未分配它——需要在 `run_scmp_huang` 中增加 `cudaMalloc(&d_wall_mat, ...)`。

#### 3.2.3 `.plt` 解析增强

在 `read_tecplot_to_flag` 中增加材质输出参数：

```c
void read_tecplot_to_flag(const std::string& filename,
                          std::vector<int>& host_flag,
                          std::vector<unsigned char>& host_mat);  // 新增出参
```

解析逻辑：
- phase=1 → flag=-2, mat=1 (石英)
- phase=0.5 → flag=-2, mat=2 (水合物)
- phase=0 → flag=1, mat=0 (流体)

> 注意：当前 `for_lbm/` 下的文件只有 phase=0 和 1，无水合物 (0.5)。代码预留水合物支持。

#### 3.2.4 `mark_geometry_scmp` kernel

四阶段单 kernel 完成所有几何标记（等价于 MCMP 的 `mark_fluid_solid` + `mark_boundary` + `mark_ghost` + `init_wall_mat_from_flag`）：

```c
__global__ void mark_geometry_scmp(int* pointsflag, unsigned char* wall_mat) {
    // Phase 1: y=0/NY-1 → ghost(-1), mat=1
    // Phase 2: y=1/NY-2 流体 → boundary(0)
    // Phase 3: 流体邻固体(-2) → boundary(0)
    // Phase 4: boundary(0)邻固体(-2) → ghost(-1)，保留 wall_mat
}
```

| MCMP kernel | SCMP 等价 | 说明 |
|-------------|-----------|------|
| `mark_fluid_solid` | .plt 读取直接给出 flag | 不需要 |
| `mark_boundary` | `mark_geometry_scmp` Phase 2+3 | 流体邻固体→边界(0) |
| `mark_ghost` | `mark_geometry_scmp` Phase 4 | 固体邻边界→ghost(-1) |
| 域边界 y=0/NY-1 | `mark_geometry_scmp` Phase 1 | y=0/NY-1 强制 ghost |
| `init_wall_mat_from_flag` | .plt 解析时写入 host_mat | 材质已在读取时确定 |

#### 3.2.5 ghost 层的作用

ghost 层 (flag=-1) 在 LBM 边界处理中扮演关键角色：

- `boundary_scmp_gpu` 对每个 boundary 节点 (flag=0)：
  - 若邻居为 ghost(-1) → **反弹**：`fin[k] = fout[opp[k]]`（无滑移）
  - 若邻居为流体 → **正常流**：`fin[k] = fout_nb[k]`
- ghost 节点本身不参与 streaming（被 `stream_single_component_gpu` 跳过）
- ghost 节点的 ψ 值被 `update_ghost_psi_bc` 用于 Scheme IV 接触角控制

#### 3.2.6 `init_all_scmp_gpu` — mode 5

```c
} else if (mode == 5) {
    // Porous media: flag & mat already set by mark_geometry_scmp
    // Uniform liquid density (single-phase)
    // cs_G = -1.0 works fine here: uniform ρ → ∇ψ=0 → F_mol=0
    rho[idx] = fmax(get_huang_rho_l(), 1e-6);
}
```

mode 5 不修改 pointsflag，仅初始化均匀液态水密度。

---

### Phase 3: `run_scmp_huang()` 重写

#### 3.3.1 新流程

```
allocate Fluid_dev F
↓
cudaMalloc pointsflag_dev + wall_mat_dev  (分别 N*sizeof(int) 和 N*sizeof(unsigned char))
↓
read_tecplot_to_flag(geom_file) → host_flag[], host_mat[]
cudaMemcpy host_flag → pointsflag_dev
cudaMemcpy host_mat  → wall_mat_dev
↓
mark_geometry_scmp<<<>>>(pointsflag, wall_mat)
↓
init_device_variable()
push_device_constants()  ← 含 per-material θ 数组上传
↓
init_all_scmp (mode 5)              ← 均匀液态水密度（cs_G=-1 正常）
↓
SteadyMonitor 初始化 + prepare_domain
↓
┌─ driven flow (单阶段，无 equilibration) ──────────┐
│ for step = 0..flow_max_steps:                      │
│   evolution_scmp(F..., pointsflag);                 │
│   if step % OUTPUT_EVERY == 0: outputvtk_scmp(...) │
│   if step % SM.interval == 0:                     │
│     Q = SM.compute_Q_scmp(...)                    │
│     if SM.compare_and_update(Q, step): break;     │
│   if step % CP_EVERY == 0: save_checkpoint_scmp() │
└────────────────────────────────────────────────────┘
↓
写 run_summary.txt (Q, steady_step, elapsed, ...)
↓
free_all + cudaFree(pointsflag_dev) + cudaFree(wall_mat_dev)
```

#### 3.3.2 无 equilibration 阶段

单相均匀液态水初始化后无相分离过程，**不需要 equilibration stage**。直接进入 driven flow。

#### 3.3.3 SteadyMonitor 适配 — `compute_Q_scmp`

新增 SCMP 专用单相流量规约方法：

```c
// steady_monitor.cuh — 新增声明
void compute_Q_scmp(const double* ux, const double* rho,
                    const int* pointsflag,
                    double& Q, int nTot) const;
```

GPU kernel 规约公式：
$$Q = \frac{1}{NX} \sum_{i: \text{flag}_i > 0} u_x(i)$$

单相流无需 B 相和互斥占优掩码，实现比 MCMP 版更简单。

#### 3.3.4 断点续跑（SCMP 版）

保存/恢复内容（仅一个 Fluid_dev，约 15 个 double 数组）：
- ρ, ux, uy, psi, pressure, Fx_mol, Fy_mol, Fx_ads, Fy_ads, p_xx, p_yy, p_xy
- fin, fout, min, mout, S, C

与 MCMP checkpoint 共用 `CheckpointHeader` 结构，但只写/读一个 fluid snapshot。

---

### Phase 4: Per-material Scheme IV 润湿性

#### 4.1 设计动机

水合物沉积物包含两种固体（石英、水合物），具有不同表面润湿性。SCMP 的 Scheme IV 通过修改 ghost 节点 ψ 值控制接触角，天然支持 per-node 差异化——每个 ghost 节点根据 `wall_mat[idx]` 查表获取对应 θ。

#### 4.2 θ_in 反向公式

来自 `DEBUG_SUMMARY_20260524.md` §4.2：

$$\theta_{in} = -1.78\times10^{-4} \cdot \theta_{meas}^3 + 4.95\times10^{-2} \cdot \theta_{meas}^2 - 3.02 \cdot \theta_{meas} + 84.5$$

- $\theta_{meas}$：期望的测量接触角（°）
- $\theta_{in}$：输入到 Scheme IV 的 `theta_target_rad` 值

例如 $\theta_{meas}=30°$ → $\theta_{in} \approx 30.3°$（接近线性）。

#### 4.3 实现方案

**a) 新增 device constant** (`LBM.h`)：

```c
__constant__ double d_theta_by_mat_rad[256];  // mat_id → θ_in (radians)
```

**b) 在 `push_device_constants` 中填充**：

```c
// Python/C++ 侧预计算 θ_in
double h_theta[256] = {0};
h_theta[1] = theta_in_quartz_rad;   // mat=1: 石英
h_theta[2] = theta_in_hydrate_rad;  // mat=2: 水合物
// mat=0 (非壁面): 保持 0
cudaMemcpyToSymbol(d_theta_by_mat_rad, h_theta, sizeof(h_theta));
```

**c) 修改 `update_ghost_psi_bc` 签名和逻辑**：

```c
// 旧签名: (psi, pointsflag, theta_target_rad)
// 新签名: (psi, pointsflag, wall_mat)  ← theta 从 wall_mat 查表获取
__global__ void update_ghost_psi_bc(
    double* psi, const int* pointsflag,
    const unsigned char* wall_mat);
```

kernel 内部将 `theta_target_rad` 替换为 `d_theta_by_mat_rad[wall_mat[idx]]`。

**d) 同步修改 `evolution_scmp` 调用**：

```c
// 旧: update_ghost_psi_bc<<<>>>(psi, pointsflag, arg);
// 新: update_ghost_psi_bc<<<>>>(psi, pointsflag, wall_mat);
```

其中 `wall_mat` 需要作为额外的设备指针传入 `evolution_scmp`（新增参数）。

**e) 首版设置**：两种材质使用相同接触角（30°），后续可差异化。

#### 4.4 与 MCMP 润湿机制的对比

| 方面 | MCMP (GAw 查表) | SCMP (Scheme IV + wall_mat) |
|------|-----------------|---------------------------|
| 控制对象 | 吸附力系数 $G_{Aw}$ | ghost 节点伪势 $\psi_{ghost}$ |
| 影响机制 | $F_{ads} = -G_{Aw} \psi \nabla s$ | 通过 $\nabla\psi$ 影响 $F_{mol}$ |
| 材质区分 | `GAw_by_mat_gpu[mat_id]` | `d_theta_by_mat_rad[mat_id]` |
| 物理效果 | 修改壁面附近合力 | 修改壁面附近密度/ψ 梯度 |
| 已验证 | ✅ MCMP 多孔介质案例 | ✅ SCMP 接触角标定 (30°–145°) |

---

### Phase 5: 配置文件与输出

#### 5.1 `configs/single_phase_water_scmp.yaml`

```yaml
# 单相水渗流 — Huang SCMP (cs_G=-1.0, 均匀密度)
huang:
  pp_mode: 1
  huang_init_mode: 5         # porous media from .plt
  epsilon_huang: 1.7         # 已验证最优 ε
  k2_huang: 0.0
  tau_huang: 1.5             # ν = (1.5-0.5)/3 = 0.333
  Lambda_huang: 0.08333      # = 1/12
  alpha_meq: 1.0

  # CS-EOS（保持已验证参数）
  cs_a: 1.0; cs_b: 4.0; cs_R: 1.0
  cs_T: 0.9; cs_G: -1.0      # 保持已验证值

  # 均匀液态水密度
  huang_rho_l: 1.0
  huang_rho_g: 1.0

  # 数值护栏
  huang_u_max: 0.15
  huang_psi_cut: 1.0e-3

# ── 润湿性 (Scheme IV per-material) ──
wettability:
  theta_quartz_deg: 30.0     # 石英接触角（测量值）
  theta_hydrate_deg: 30.0    # 水合物接触角（初版相同）

# ── 几何 ──
geometry:
  geom_file: "data/geometry/for_lbm/geometry_case0000.plt"

# ── 驱动力 ──
driving:
  Gx: 1.0e-7; Gy: 0.0; drive_mode: 1

# ── 稳态判据 ──
convergence:
  flow:
    tol_rel: 1.0e-5; need_consec: 3; max_steps: 2000000

# ── I/O ──
output:
  OUTPUT_EVERY: 20000; ENABLE_CKPT: true; CP_EVERY: 50000
```

#### 5.2 输出目录结构

从 .plt 文件名提取 case 标识（`geometry_case0000.plt` → `case0000`）：

```
results/<case_id>/
├── params.txt                  # 本次运行的完整参数
├── run_summary.txt             # 稳态流量 + 运行统计
├── outputdata_flow/
│   ├── flow00000000.vtk
│   ├── flow00020000.vtk
│   └── ...
└── ckpt/
    ├── ckpt_flow_00050000.bin
    └── ckpt_latest_flow.bin
```

#### 5.3 `run_summary.txt` 格式

```
Q 1.234e-5
steady_step 450000
elapsed_s 123.4
MLUPS 5.67
Gx 1.0e-7
tau_huang 1.5
nu_lu 0.333
geom_file geometry_case0000.plt
NX 300
NY 300
```

#### 5.4 渗流率后处理

**不在求解器内计算**。Python 后处理脚本从 VTK 提取稳态速度场后计算：
$$K_{eff} = \frac{\nu \cdot \langle u_x \rangle}{G_x}$$

---

## 四、实施顺序

| 阶段 | 内容 | 文件 | 预计改动 | 依赖 |
|------|------|------|:------:|------|
| **P1** | `HUANG_POROUS_BUILD` + `#ifdef` 守卫 + build.py I×J 检测 | `LBM.h`, `build.py`, `main.cu`, `LBM.cu`, `sim_utils.cu` | ~40 行 | — |
| **P2** | `mark_geometry_scmp` + `read_tecplot_to_flag` 增强 + mode 5 + `d_wall_mat` 分配 | `LBM.cu`, `sim_utils.cu` | ~140 行 | P1 |
| **P3** | `run_scmp_huang` 重写（.plt 几何 + SteadyMonitor + 单阶段 + 断点） | `sim_utils.cu` | ~250 行 | P2 |
| **P4** | `compute_Q_scmp` + reduce kernel | `steady_monitor.cu/.cuh` | ~50 行 | P3 |
| **P5** | Per-material Scheme IV（`d_theta_by_mat_rad` + `wall_mat` 集成到 `evolution_scmp`） | `LBM.cu`, `LBM.h`, `sim_utils.cu` | ~80 行 | P2 |
| **P6** | 配置文件 + Smoke test | `configs/` + 编译运行 | ~50 行 | P5 |

**总预计改动量**：~610 行新增/修改，涉及 7-8 个文件。

> **与 v2 相比**：去掉了 cs_G=0 数值保护阶段（~30 行），实施阶段从 7 个减为 6 个。

---

## 五、关键决策确认

| # | 议题 | 决策 |
|---|------|------|
| 1 | 单相实现 | `cs_G=-1.0`（保持已验证）+ 均匀液态水密度 |
| 2 | ghost 层 | 保留，bounce-back 和 Scheme IV 需要 |
| 3 | 润湿性 | Per-material Scheme IV，初版均 30° |
| 4 | Equilibration | 不需要 |
| 5 | 断点续跑 | 需要 |
| 6 | 润湿机制 | Scheme IV ψ-ghost BC |
| 7 | 网格尺寸 | 从 .plt 头 I=/J= 自动检测 |
| 8 | 入口/出口 | Gx 体积力 + x 周期性 BC |
| 9 | SteadyMonitor | 新增 SCMP 专版 `compute_Q_scmp` |
| 10 | 输出目录 | `results/<case_id>/outputdata_flow/` |
| 11 | 渗透率 | 不在求解器中计算，仅输出 Q |
| 12 | 物性匹配 | 初版用简化格子参数，后续精确换算 |
