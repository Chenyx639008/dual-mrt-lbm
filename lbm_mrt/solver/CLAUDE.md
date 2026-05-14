# solver/ — CUDA MRT-LBM 核心代码目录

该目录是整个项目的计算核心，编译产出两个独立二进制文件。Python 层仅负责调用，不应修改此处的任何 `.cu` / `.h` 文件逻辑，除非有明确的物理或数值需求。

## 目录结构

```
solver/
├── include/                   # 头文件
│   ├── LBM.h                  # 格子常数、MRT 矩阵、结构体、设备全局变量声明
│   ├── sim_utils.h            # RuntimeParams、StageConfig、运行循环函数声明
│   ├── hydrate.h              # (HYDRATE_ENABLE) 热场/浓度/VOP 结构体与§1-§11声明
│   ├── steady_monitor.cuh     # SteadyMonitor：稳态判据 + limiter 监控
│   └── unified_cuda_error_check.cuh  # checkCudaErrors 宏
├── src/                       # CUDA 实现
│   ├── main.cu                # 程序入口：初始化 → eq → flow → 释放
│   ├── LBM.cu                 # 核心 GPU 核函数：碰撞、流、边界、几何、润湿
│   ├── sim_utils.cu           # 参数读取、常量上传、运行循环、VTK/断点 I/O
│   ├── steady_monitor.cu      # SteadyMonitor 实现（流量规约、锁死检测）
│   ├── hydrate.cu             # (HYDRATE_ENABLE) 热场 + 浓度场核函数
│   └── hydrate_vop.cu         # (HYDRATE_ENABLE) VOP 固相更新核函数
├── build.py                   # Python 编译入口（`uv run lbm-build` 走这里）
├── mcmp_sim                   # 编译产出：flow-only 二进制
└── mcmp_sim_hydrate           # 编译产出：hydrate-enabled 二进制
```

## 两变体编译模型

| 变体 | 预处理宏 | 额外源文件 | 产出二进制 |
|------|----------|-----------|------------|
| flow-only | 无 | 无 | `mcmp_sim` |
| hydrate | `-DHYDRATE_ENABLE` | `hydrate.cu` `hydrate_vop.cu` | `mcmp_sim_hydrate` |
| huang-scmp | `-DHUANG_256_BUILD` | 无（全部在 LBM.cu 内 #ifdef） | `mcmp_huang_256` |

`HUANG_256_BUILD` 与 `HYDRATE_ENABLE` 互斥（`LBM.h:8` 有 `#error` 守卫）。

所有水合物相关代码均被 `#ifdef HYDRATE_ENABLE` 保护，flow-only 编译完全不链接这些文件。

## 关键头文件速查

### `include/LBM.h` — 编译期常量与结构体

- **网格**：
  - flow-only / hydrate：`NX=339`, `NY=212`（`SCALE=1`），`NSTEPS=5000000`，`NOUTPUT=50000`
  - huang-scmp：`NX=256`, `NY=256`，`NSTEPS=200000`，`NOUTPUT=5000`
  - 修改网格尺寸必须重新编译，不能运行时改变
- **D2Q9 格子**：`e[9][2]`、`opp[9]`、`w[9]`、`w_F[9]`
- **MRT 矩阵**：`M[9][9]`、`Minv[9][9]`、松弛率 `tau_e/tau_t/tau_q/tau_p`
- **设备全局变量**：通过 `#define LBM_DEFINE_GLOBALS` 控制定义/声明分离
  - 仅 `LBM.cu` 在 include 前 `#define LBM_DEFINE_GLOBALS`，其他文件只得到 `extern` 声明
  - 关键 `__constant__`：`d_Gx/d_Gy`、`d_tau_p_a/b`、`d_kappa`、`A_a_gpu/A_b_gpu`、`GAw_by_mat_gpu`
- **主要结构体**：`Fluid_dev`（9个 double* 场 + fin/fout/min/mout/S/C）、`Mix_dev`、`Fluid_host`、`Mix_host`、`Obstacle`
- **PR-EOS 参数**：`eos::` 命名空间，水+甲烷 Peng-Robinson 常数

### `include/sim_utils.h` — 运行时参数与运行循环

- **`RuntimeParams`**：所有可调参数的 C++ 结构体（对应 `params.txt` 的键）
  - 几何：`morph`, `geom_file`, `r_obs`, `coat_thick`
  - 初始/润湿：`Sw`, `water_seed`, `thetaA_quartz_deg`, `thetaA_hydrate_deg`, `GAw_m/c`
  - 驱动：`Gx`, `Gy`, `drive_mode`, `rhoA_hi/lo`, `rhoB_hi/lo`
  - 物性：`tau_p_a/b`, `GAB`, `GBA`, `sigmaA`, `kappa`
  - 断点：`CP_EVERY`, `CP_KEEP`, `CP_RESUME`, `ENABLE_CKPT`
  - 目录：`ckpt_dir`, `file_dir`
  - 稳态判据：`eq_tol_rel/need_consec/max_steps`，`flow_tol_rel/need_consec/max_steps`
  - 水合物扩展（`HYDRATE_ENABLE`）：`hydrate_enable`, `T0_init`, `T0_inlet`, 热导率, `k0_rxn`, `Ea_rxn`, `Vh_init`…
- **`StageConfig`**：单阶段运行配置（`drive_scale`, `tol_rel`, `max_steps`, `tag`）
- **`RunResult`**：阶段运行结果（稳态步、流量 QA/QB、断点恢复信息、水合物诊断）
- **核心函数**：
  - `load_params_txt()` — 解析 `params.txt` 键值对到 `RuntimeParams`
  - `push_device_constants()` — 将 `RuntimeParams` 上传到 `__constant__` 内存
  - `push_wettability_and_maps()` — 构建润湿查表并上传到 `GAw_by_mat_gpu`
  - `build_and_upload_geometry_from_tecplot()` — 读取 `.plt` 几何文件，写 `pointsflag`
  - `run_stage()` — 通用单阶段时间推进循环（含稳态判据、断点、VTK 输出）
  - `run_stage_hydrate()` — 水合物耦合版推进循环
  - `run_equilibrate_then_flow()` — flow-only 的两阶段封装

### `include/hydrate.h` — 水合物相变扩展（`HYDRATE_ENABLE`）

- **D2Q5 格子**：`Q5=5`，`e5[5][2]`，MRT 变换矩阵 `M5[5][5]` / `Minv5[5][5]`
- **新结构体**：
  - `Therm_dev`：DDF 热 LBM 分布函数 `h_in/h_out` + 温度场 `T`
  - `Conc_dev`：MRT-CST 浓度 LBM 分布函数 `g_in/g_out` + 浓度场 `Cm`
  - `VOP_dev`：`Vh`（体积分数）、`diss_rate`（分解速率）、`S_latent`（潜热源）、`new_fluid_flag`（翻转标记）、`pore_origin`（诊断场）
- **设备常量**：通过 `#define HYDRATE_DEFINE_GLOBALS` 控制，仅 `hydrate.cu` 定义
- **物理函数**：
  - `init_device_variable_hydrate()` — 将物理→格子单位换算后上传常量
  - `step_thermal()` — 热场一步：DDF 碰撞→流→边界
  - `step_conc()` — 浓度场一步：MRT-CST + Kang 反应边界（写 `diss_rate`）
  - `step_vop()` — VOP 固相更新：`Vh` 减少 → 翻转检测 → 新流体重建
  - `step_hydrate_physics()` — 全耦合入口，依序调用以上三步

### `include/steady_monitor.cuh` — 稳态与锁死监控

- **`SteadyMonitor`**：
  - 体流量规约：`compute_Q_GPU()` → 通过 `d_sumA/B` 对 `ux*rho` 求和
  - 稳态判据：`compare_and_update(QA, QB, QT, step)` — 相对容差 + 连续命中次数
  - Limiter 监控：统计每步碰撞后 τ 限幅命中比率，可写入 `limiter_log.csv`
  - `prepare_domain()` — 预计算入口/出口截面掩码，初始化流体节点数 `nFluid`
  - `init_limiter_monitor()` — 保存 `pointsflag` 指针并初始化计数器缓冲



- **`LBM.h`** — grid constants (`NX=300`, `NY=300`, `SCALE`, `NSTEPS=5000000`, `NOUTPUT=50000`), D2Q9 lattice vectors, MRT matrices M / Minv, all `Fluid_dev` / `Mix_dev` struct definitions, and obstacle geometry
- **`sim_utils.h` / `sim_utils.cu`** — `RuntimeParams` struct (all tunable params), `load_params_txt()`, `push_device_constants()`, geometry loading from Tecplot `.plt` files
- **`LBM.cu`** — core GPU kernels: `mrt_collide_two_components_gpu`, `stream_two_components_gpu`, `boundary_gpu`, obstacle marking (`mark_fluid_solid`, `mark_ghost`), wettability upload
- **`main.cu`** — entry point; two-stage run: equilibration (`eq`) then driven flow (`flow`); reads `params.txt` via `argv[1]` or env var `LBM_FILE_DIR`; outputs VTK snapshots
- **`steady_monitor.cuh` / `steady_monitor.cu`** — `SteadyMonitor` struct: GPU-side flow-rate convergence detection (consecutive-hit criterion), per-fluid limiter hit counters (`d_ctA_any`, `d_ctB_any` etc.), and optional CSV logging to `data/file/limiter_log.csv`
- **`hydrate.h` / `hydrate.cu`** — thermal (D2Q5 DDF LBM), concentration (D2Q5 MRT-CST), and VOP struct definitions and §1–§12 physics; `init_vop(VP, pointsflag, Vh_init)` initializes the hydrate volume fraction field on GPU
- **`hydrate_vop.cu`** — VOP (Volume-Of-Pore) solid-phase update: `kernel_update_vop` (Vh reduction), `kernel_apply_vop_conversion` (flag flip -3→fluid, writes `pore_origin`), `kernel_reinit_new_fluid` (field reconstruction after conversion); `step_vop()` orchestrates these


## 节点 flag 约定

| 值 | 含义 |
|----|------|
| `> 0` | 流体内部 |
| `0` | Ghost/边界流体 |
| `-1` | 壁面 ghost（材质由 `d_wall_mat[]` 给出） |
| `-2` | 固体内部（石英，mat=1） |
| `-3` | 水合物内部（mat=2） |

VOP 核函数作用对象：`flag == -3` 或（`flag == -1` 且 `d_wall_mat == 2`），分解从水合物-流体界面向内传播。

## 关键数值参数（编译期）

```
NX = NY = 300          // 网格，修改后必须重编译
NSTEPS   = 5_000_000   // 最大时间步
NOUTPUT  = 50_000      // VTK 输出间隔
BLOCK_SIZE = 32        // CUDA block 宽度
线程配置: 32×1 per block，(NX/32)×NY grid
```

## 常见修改点索引

| 需求 | 位置 |
|------|------|
| 修改网格尺寸 | `include/LBM.h`：`NX` / `NY` / `SCALE` |
| 新增运行时参数 | `include/sim_utils.h`：`RuntimeParams` 字段 + `sim_utils.cu`：`load_params_txt()` + `push_device_constants()` |
| 修改 MRT 松弛率 | `include/LBM.h`：`tau_e/tau_t/tau_q` 和 `push_device_constants()` 组装的 `A_a/b_gpu` |
| 修改润湿性映射公式 | `sim_utils.cu`：`push_wettability_and_maps()` |
| 修改边界条件 | `src/LBM.cu`：`boundary_gpu` |
| 修改水合物反应动力学 | `src/hydrate.cu`：Kim-Bishnoi 核函数 + `RuntimeParams` 中 `k0_rxn/Ea_rxn` |
| 修改 VOP 翻转逻辑 | `src/hydrate_vop.cu`：`kernel_update_vop` / `kernel_apply_vop_conversion` |
| 新增 VTK 输出字段 | `sim_utils.cu`：`write_stage_output()` / `outputvtk_append_hydrate()` |

## 编译命令（从项目根目录）

```bash
uv run lbm-build                        # flow-only → mcmp_sim
uv run lbm-build --hydrate              # hydrate → mcmp_sim_hydrate
uv run lbm-build --arch sm_89           # 覆盖 GPU 架构（默认 sm_120）
uv run lbm-build --debug                # 加 -g -G
uv run lbm-build --hydrate --dry-run    # 仅打印 nvcc 命令
```
