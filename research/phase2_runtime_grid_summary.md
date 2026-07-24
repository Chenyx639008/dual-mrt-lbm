# Phase 2: CUDA 运行时网格 — 实施总结

> **实施日期**: 2026-07-17
> **状态**: ✅ 代码改造完成 · ✅ 编译验证通过 (unified + 256) · ✅ 非 unified 无回归 · ⏳ 运行时网格切换待验证
> **参考**: [`unified_framework_feasibility.md`](unified_framework_feasibility.md) §3

---

## 目标

将 Huang SCMP 的多个固定网格二进制（`mcmp_huang_256`、`mcmp_huang_400`、`mcmp_huang_800`）合并为**一个统一二进制 `mcmp_huang_unified`**，编译到最大网格（1024×1024），运行时通过 `params.txt` 指定实际网格尺寸。

---

## 修改清单

### 1. `lbm_mrt/solver/include/LBM.h`

| 变更 | 说明 |
|------|------|
| 新增 `#ifdef HUANG_UNIFIED_BUILD` 段 | 编译期最大网格 NX=1024, NY=1024 |
| 新增 `__constant__ int d_nx_active, d_ny_active` | 运行时活跃网格尺寸（仅 unified 构建） |
| 新增 `get_nx_active()` / `get_ny_active()` | 统一访问器：unified → d_nx/d_ny；非 unified → constexpr NX/NY |
| `LBM_DEFINE_GLOBALS` 块中新增 Group F | d_nx_active/d_ny_active 定义 + dbg sink 保护 |
| `#else` 块中新增 extern 声明 | 多编译单元可见性 |

```cpp
// 🔧 Phase 2 核心设计
#ifdef HUANG_UNIFIED_BUILD
__constant__ int d_nx_active, d_ny_active;
__device__ __forceinline__ int get_nx_active() { return d_nx_active; }
__device__ __forceinline__ int get_ny_active() { return d_ny_active; }
#else
__device__ __forceinline__ constexpr int get_nx_active() { return NX; }
__device__ __forceinline__ constexpr int get_ny_active() { return NY; }
#endif
```

**设计要点**：
- 非 unified 构建：`get_nx_active()` = `constexpr NX`，编译器优化为零开销
- Unified 构建：`d_nx_active` 存储在 `__constant__` 内存，延迟 ≈ 寄存器（constant cache 广播）
- 内存分配（`mem_size_scalar` 等）仍使用 `NX * NY`（最大网格），确保数组容量充足

### 2. `lbm_mrt/solver/include/sim_utils.h`

| 变更 | 说明 |
|------|------|
| `RuntimeParams` 新增 `nx_override`, `ny_override` | Section C（Huang SCMP），默认 0 = 使用编译期 NX/NY |

```cpp
// 🔧 Phase 2: Runtime grid override
int nx_override = 0;  // 0 = use NX_MAX; >0 = active grid width
int ny_override = 0;  // 0 = use NY_MAX; >0 = active grid height
```

### 3. `lbm_mrt/solver/src/sim_utils.cu`

| 变更 | 说明 |
|------|------|
| `load_params_txt()` 新增读取 `nx_override` / `ny_override` | 从 params.txt 读取运行时网格参数 |
| `push_device_constants()` 新增 d_nx_active/d_ny_active 上传 | 仅在 `HUANG_UNIFIED_BUILD` 下编译 |

```cpp
#ifdef HUANG_UNIFIED_BUILD
int active_nx = (p.nx_override > 0) ? p.nx_override : (int)NX;
int active_ny = (p.ny_override > 0) ? p.ny_override : (int)NY;
cudaMemcpyToSymbol(d_nx_active, &active_nx, sizeof(int));
cudaMemcpyToSymbol(d_ny_active, &active_ny, sizeof(int));
#endif
```

### 4. `lbm_mrt/solver/src/LBM.cu`

| 变更 | 说明 |
|------|------|
| 内核边界检查 `x >= NX \|\| y >= NY` → `x >= get_nx_active() \|\| y >= get_ny_active()` | **35 处**，覆盖所有内核（MCMP + SCMP） |
| `dbg_consts_once` 新增 sink 引用 | 防止设备链接器裁剪 `d_nx_active`/`d_ny_active` |
| `evolution_scmp` 函数头注释标记 `🔧 Phase 2` | 运行时网格就绪标识 |

**关键内核列表**（均已完成运行时网格适配）：

| 内核名 | 类型 | 网格边界 |
|--------|------|---------|
| `compute_molecular_force_scmp` | SCMP | ✅ get_nx/ny_active |
| `compute_adsorption_force_scmp` | SCMP | ✅ |
| `update_ghost_psi_bc` | SCMP 润湿 | ✅ |
| `add_adsorption_to_force` | SCMP | ✅ |
| `compute_velocity_scmp` | SCMP | ✅ |
| `compute_S_huang_gpu` | SCMP | ✅ |
| `compute_Q_huang_gpu` | SCMP | ✅ |
| `mrt_collide_single_component_gpu` | SCMP | ✅ |
| `stream_single_component_gpu` | SCMP | ✅ |
| `boundary_scmp_gpu` | SCMP | ✅ |
| `zouhe_bottom_wall_gpu` | SCMP | ✅ |
| `compute_pressure_tensor_scmp` | SCMP | ✅ |
| `init_all_scmp_gpu` | SCMP 初始化 | ✅ |
| `scmp_mark_boundary_gpu` | SCMP 几何 | ✅ |
| All MCMP kernels | MCMP | ✅ (兼容，非unified下等同NX/NY) |

### 5. `lbm_mrt/solver/build.py`（已有，无需修改）

| 编译选项 | 效果 |
|----------|------|
| `--huang-unified` | `-DHUANG_256_BUILD -DHUANG_UNIFIED_BUILD -DHUANG_NX=1024 -DHUANG_NY=1024` |
| 输出二进制 | `mcmp_huang_unified` |

---

## 使用方式

### 当前可用（已验证稳定）

```bash
# ── 固定网格 256×256（稳定、推荐日常使用）──
uv run lbm-build --huang                         # 编译 mcmp_huang_256
uv run lbm run scmp_cs_huang_256                  # 运行（质量守恒 ✅）

# ── 参数覆盖 ──
uv run lbm run scmp_cs_huang_256 cs_T=0.80 epsilon_huang=2.0 huang_R0=30

# ── 接触角案例 ──
uv run lbm run scmp_cs_huang_256_theta60          # θ=60° 亲水
uv run lbm run scmp_cs_huang_256_theta120         # θ=120° 疏水
```

### 🆕 统一二进制（已编译，运行时网格切换待验证）

```bash
# 编译统一二进制（一次性，需 CUDA 环境）
uv run lbm-build --huang-unified
# 输出: mcmp_huang_unified (最大 1024×1024)

# 运行任意网格（需先注册对应模型或在 params.txt 中指定 nx_override/ny_override）
uv run lbm run scmp_cs_huang_256    # 256×256（自动注入）
# 400×400 / 800×800 模型待注册
```

### 完整参数传递链路

```
ModelDefinition.to_params_dict()
  ├── EOS:   cs_a, cs_b, cs_R, cs_T (=T_reduced), cs_G
  ├── Force: epsilon_huang, k2_huang, kd_huang
  ├── Collision: tau_huang, Lambda_huang, alpha_meq
  ├── Wetting: G_ads, theta_contact_deg, huang_psi_l/g_ref
  ├── Initial: huang_init_mode, huang_R0/xc/yc/W
  ├── Coexistence: huang_rho_g, huang_rho_l (Python Maxwell 构造 🔧)
  ├── 🔧 Phase 2: nx_override, ny_override (unified 构建用)
  └── Guards: huang_u_max, huang_psi_cut, huang_tanh_factor, huang_rho_max_init
       ↓
  params.txt  →  load_params_txt()  →  RuntimeParams
       ↓
  push_device_constants()  →  __constant__ 设备内存
       ↓
  SCMP 内核: x >= get_nx_active() (35 处)

---

## 设计决策记录

### 为什么改了所有内核而非仅 SCMP？
`get_nx_active()` 在非 unified 构建中返回 `constexpr NX`，编译器优化为直接常量——零性能开销。统一改用访问器函数避免了条件编译分支，降低了维护成本。

### 为什么 mem_size 仍用 NX*NY？
内存分配发生在 host 端（`cudaMalloc`），需要在编译期知道最大尺寸。`NX=1024` 足够覆盖所有 Huang 变体（256/400/800）。显存增量可接受（1024² × Q × 8 bytes ≈ 75 MB per fluid）。

### 为什么选 `__constant__` 而非 `__device__`？
`__constant__` 内存通过 constant cache 广播，同一 warp 内所有线程读取相同值时延迟 ≈ 寄存器访问。网格边界检查在每个 kernel 开头执行一次，完全适合 constant memory。

---

## 待验证项

- [x] `mcmp_huang_unified` 编译通过 ✅
- [x] `mcmp_huang_256` 非 unified 构建无回归 ✅（质量守恒验证）
- [x] 参数稳定性验证 ✅（ε=1.7, ρ_g=0.00917, ρ_l=0.30520）
- [ ] `nx_override=256` 运行时网格精度与 `mcmp_huang_256` 对比
- [ ] 多网格尺寸切换：同一 `mcmp_huang_unified` 跑 256² / 400² / 800²
- [ ] `nx_override`/`ny_override` 自动注入 `to_params_dict()`
- [ ] 性能对比：unified vs 固定网格二进制

---

## 与 feasibility.md §3 的对照

| feasibility 建议 | 实施状态 |
|:---|:---|
| LBM.h 添加 `__constant__ int d_nx_active, d_ny_active` | ✅ |
| 内核循环边界改为 `d_nx_active`/`d_ny_active` | ✅ 35 处 |
| RuntimeParams 添加 `nx_override`/`ny_override` | ✅ |
| push_device_constants 上传运行时网格 | ✅ |
| 需要改为 `d_nx_active` 的内核约 15-20 个 | ✅ 实际覆盖全部 35 个（含 MCMP 兼容） |
| 改动量约 50 行 | ✅ 实际约 80 行（含头文件基础设施） |
