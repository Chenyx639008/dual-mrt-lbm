# 多孔介质 SCMP 达西流 — 开发与调试总结

> 日期: 2026-05-24
> 分支: `huang_mrt_2d`
> 产出: `results/steady_check/`, `results/hydrate_flag_fix/`
> 关联: [DEBUG_SUMMARY_20260524.md](DEBUG_SUMMARY_20260524.md) (前期 SCMP 验证)

---

## §0 概述

在已完成 Huang & Wu (2016) SCMP 求解器全面验证（ε=1.7 最优）的基础上，本日工作将求解器扩展为**多孔介质单相达西流模拟器**：从 `.plt` 几何文件读取孔隙/固体/水合物分布，按 MCMP 规范标记 ghost（Scheme IV 润湿性）和 boundary（半步反弹）节点，支持 per-material 接触角（石英 vs 水合物），并以体力密度 Gx 驱动稳态渗流。

核心产出：
- 一套从 `.plt` → flag 标记 → 稳态达西流 → Q 输出的完整流程
- 4 个关键 Bug 的定位与修复
- 科学可行性验证（稳态收敛、NaN 检查、反弹边界、连通性分析）

---

## §1 Bug 修复清单

### Bug P1: `d_wall_mat` 从未分配（致命）

**位置**: `main.cu` → SCMP dispatch 位于 `push_wettability_and_maps()` 之前

**现象**: `cudaMemcpyFromSymbol(&dm, d_wall_mat, ...)` 返回 NULL，导致 `scmp_mark_boundary_gpu` 写 `wall_mat[idx]=1` 时非法内存访问（compute-sanitizer: "Address 0x0 out of bounds"）。

**根因**: `main.cu` 中 SCMP 分支在 `push_wettability_and_maps()` 之前就 `return 0`，`alloc_wall_and_wettability_maps_host()` 从未被调用。

**修复** (`main.cu`):
```c
#if defined(HUANG_256_BUILD) || defined(HUANG_POROUS_BUILD)
    {
        const char* params_path = (argc >= 2) ? argv[1] : "params.txt";
#ifdef HUANG_POROUS_BUILD
        push_wettability_and_maps(P);   // ← 新增：分配 d_wall_mat
#endif
        run_scmp_huang(P, params_path);
    }
    return 0;
#endif
```

---

### Bug P2: 跨文件 MCMP kernel 调用失效

**位置**: `sim_utils.cu` → 通过 `extern __global__` 声明调用 `mark_boundary`/`mark_ghost`

**现象**: MCMP 的 `mark_boundary` 和 `mark_ghost` kernel 从 `sim_utils.cu` 调用时，`pointsflag` 未被修改（flag 只有 -2 和 1，无 0 和 -1）。

**根因**: `__device__` 指针（`d_GAw_map`/`d_GBw_map`）作用域在 `LBM_DEFINE_GLOBALS` 上下文中的可见性问题。MCMP kernel 依赖这些指针，跨文件调用时不可用。

**修复**: 在 `LBM.cu` 中编写 SCMP 原生几何 kernel（不使用 MCMP 的 `__device__` 指针）：

```c
// Kernel 1: 流体(1) 邻近固体(-2/-3) → 边界(0)
__global__ void scmp_mark_boundary_gpu(int* pointsflag, unsigned char* wall_mat);

// Kernel 2: 固体(-2/-3) 邻近流体/边界 → ghost(-1)
__global__ void scmp_mark_ghost_gpu(int* pointsflag, unsigned char* wall_mat);

// Host wrapper
__host__ void scmp_init_geometry(int* pointsflag, unsigned char* wall_mat);
```

**关键**: 必须在 `init_device_variable()` 之后调用（kernel 依赖 `e_gpu` 常量内存）。

---

### Bug P3: 水合物 flag=-3 未从 .plt 提取

**位置**: `LBM.cu` → `read_tecplot_to_flag`

**现象**: `.plt` 中 value=0.5（水合物，4714 节点）被映射为 `flag=-2`（与石英固体混淆），而非 `flag=-3`。

**根因**:
```c
// 修复前
} else if (std::abs(phase - 0.5f) < 1e-3f) {
    host_flag[idx] = -2; host_mat[idx] = 2; // 水合物 ← 错误！
}
```

**修复** (4 处联动):
| 位置 | 修改 |
|------|------|
| `read_tecplot_to_flag` | `host_flag[idx] = -3` (水合物) |
| `scmp_mark_boundary_gpu` | `fnb == -2` → `fnb == -2 \|\| fnb == -3` |
| `scmp_mark_ghost_gpu` | `!= -2` → `!= -2 && != -3` |
| `compute_p_psi_A/B_all` | 同样跳过 -3 |

**水合物表面节点**（ghost/boundary）通过 `wall_mat[idx]=2` 保留材料身份。

---

### Bug P4: 流动未达稳态 + 孤立孔隙

**位置**: 运行参数 → `NSTEPS` 和几何分析

**现象**: 57200 步时 Q 仍在缓慢下降（Q 变化 ~0.04%/50000步）。200000 步后收敛至 <0.01%/50000步。

**几何特征**（`.plt` 自带的物理特性，非 Bug）:
- 8-连通分量: 218 个，最大仅 12.2%
- 88% 孔隙空间为孤立孔隙
- 大量单格点宽喉道（每列 3~9 个 w=1 通道）
- 半步反弹在 1-lu 通道中产生极高流动阻力

**结论**: 这是几何的真实物理特征，窄喉道堵塞是预期的模拟结果。

---

## §2 代码结构

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `lbm_mrt/solver/src/main.cu` | SCMP 分支前加 `push_wettability_and_maps(P)` |
| `lbm_mrt/solver/src/LBM.cu` | 新增 `scmp_mark_boundary_gpu` + `scmp_mark_ghost_gpu` + `scmp_init_geometry`；修复 `read_tecplot_to_flag` 水合物 flag；修复 psi kernel 跳过 -3 |
| `lbm_mrt/solver/include/LBM.h` | 声明 `scmp_init_geometry`；`evolution_scmp` 签名改用 `wall_mat`；`d_theta_by_mat_rad[256]` 常量 |
| `lbm_mrt/solver/src/sim_utils.cu` | `run_scmp_huang` 中 .plt 加载 + scmp_init_geometry 调用流程 |

### 数据流

```
┌─────────────────────────────────────────────────────────────┐
│  .plt 文件 (x, y, phase)                                    │
│    phase=0.0 → 孔隙    phase=1.0 → 石英    phase=0.5 → 水合物 │
└──────────────────────┬──────────────────────────────────────┘
                       │ read_tecplot_to_flag()
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  host_flag[NY*NX]        host_mat[NY*NX]                    │
│    1 = 流体                 0 = 无                           │
│   -2 = 石英固体             1 = 石英                         │
│   -3 = 水合物               2 = 水合物                        │
└──────────────────────┬──────────────────────────────────────┘
                       │ cudaMemcpy + scmp_init_geometry()
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  pointsflag (GPU)          d_wall_mat (GPU)                 │
│    1 = 流体内部             按节点存材料 ID                   │
│    0 = 边界 (半步反弹)                                      │
│   -1 = ghost (Scheme IV ψ)                                  │
│   -2 = 石英内部                                             │
│   -3 = 水合物内部                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │ evolution_scmp(wall_mat, pointsflag)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Scheme IV 润湿性: update_ghost_psi_bc                       │
│    d_theta_by_mat_rad[wall_mat[idx]] → 对应材料的接触角       │
│  半步反弹: boundary_scmp_gpu (flag=0 节点)                   │
│  体力驱动: Fx = Gx·ρ (均匀施加于流体节点)                     │
└─────────────────────────────────────────────────────────────┘
```

### 演化调用顺序 (`evolution_scmp`)

```
compute_rho_from_fin_gpu         ← ρ = Σ f_i
compute_p_psi_scmp_cs            ← p(ρ), ψ(ρ) from CS EOS
update_ghost_psi_bc              ← Scheme IV per-material 接触角
compute_molecular_force_scmp     ← F_mol = -Gψ∇ψ + G·ρ
compute_Q_huang_gpu              ← Q_m (ψ=0 时退化为 ~0)
compute_adsorption_force_scmp    ← F_ads (G_ads=0 时退化为 0)
compute_velocity_scmp            ← u from fin + F
compute_S_huang_gpu              ← Guo forcing term
mrt_collide_single_component_gpu
stream_single_component_gpu
boundary_scmp_gpu                ← 半步反弹 (flag=0)
compute_pressure_tensor_scmp
```

---

## §3 命令行参考

### 编译

```bash
# 多孔介质模式（自动检测 .plt 的 I×J 维度）
uv run lbm-build --porous --plt data/geometry/for_lbm/geometry_case0000.plt

# 指定 GPU 架构
uv run lbm-build --porous --plt <path.plt> --arch sm_89

# 调试模式（-g -G）
uv run lbm-build --porous --plt <path.plt> --debug

# 仅打印 nvcc 命令（不编译）
uv run lbm-build --porous --plt <path.plt> --dry-run
```

**输出**: `lbm_mrt/solver/mcmp_huang_porous_{NX}x{NY}`

### 单次运行

```bash
# 准备 params.txt（Python 辅助）
python3 -c "
params = [
    'pp_mode 1', 'huang_init_mode 5', 'epsilon_huang 1.7',
    'tau_huang 1.5', 'Lambda_huang 0.08333',
    'cs_a 1.0', 'cs_b 4.0', 'cs_R 1.0', 'cs_T 0.9', 'cs_G -1.0',
    'huang_rho_l 1.0', 'huang_rho_g 1.0',
    'theta_contact_deg 25.0',
    'thetaA_quartz_deg 25.0',   # θ_in=25° → θ_meas≈30°
    'thetaA_hydrate_deg 25.0',  # 与石英一致
    'G_ads 0.0', 'Gx 1.0e-7', 'Gy 0.0', 'drive_mode 1',
    'OUTPUT_EVERY 50000', 'morph 0', 'init_eq 0',
    'geom_file data/path/to/geometry_case.plt',
    'file_dir results/my_run',
]
with open('params.txt', 'w') as f:
    for line in params: f.write(line + '\n')
"

# 运行
./lbm_mrt/solver/mcmp_huang_porous_300x300 params.txt
```

**输出**: `results/my_run/outputdata_scmp/flow{step}.vtk`

### VTK 包含的标量场

| 标量 | 含义 |
|------|------|
| `rho` | 密度 |
| `ux`, `uy` | 速度分量 |
| `pressure` | 压力 ($p=c_s^2\rho+\frac{1}{2}c^2G\psi^2$) |
| `p_xx`, `p_yy` | 压力张量分量 |
| `Fx`, `Fy` | 总力 (分子力 + 吸附力 + 体力) |
| `psi` | 赝势 (单相时 ≈0) |
| `flag` | 节点类型 (1/0/-1/-2/-3) |

---

## §4 批量模拟流程

### 4.1 参数空间设计

创建 CSV 设计文件，每行一个 case：

```csv
case_name,geom_file,Gx,tau_huang,thetaA_quartz_deg,thetaA_hydrate_deg,OUTPUT_EVERY,file_dir
case_01,data/geom/geo001.plt,1e-7,1.5,25,25,50000,results/batch_20260525
case_02,data/geom/geo001.plt,2e-7,1.5,25,25,50000,results/batch_20260525
case_03,data/geom/geo002.plt,1e-7,1.5,30,30,50000,results/batch_20260525
...
```

### 4.2 批量管理器（推荐）

**脚本**: `scripts/darcy_batch_manager.py` — 一键式跨项目批量管理

```bash
# Step 1: 扫描 hydrate_structure 案例，生成设计 CSV
uv run python scripts/darcy_batch_manager.py design --dry-run    # 预览
uv run python scripts/darcy_batch_manager.py design              # 正式生成

# 自定义扫描参数
uv run python scripts/darcy_batch_manager.py design \
    --gx 1e-8 5e-8 1e-7 5e-7 \
    --theta-quartz 25 30 45 \
    --theta-hydrate 25 60 80

# Step 2: 批量运行
uv run python scripts/darcy_batch_manager.py run --max-parallel 2
uv run python scripts/darcy_batch_manager.py run --force          # 强制重跑

# Step 3: 提取渗透率 → hydrate_structure/data/tables/flow.csv
uv run python scripts/darcy_batch_manager.py collect

# Step 4 (可选): 清理 VTK 释放空间
uv run python scripts/darcy_batch_manager.py clean --keep-last --dry-run
uv run python scripts/darcy_batch_manager.py clean --keep-last
```

**功能矩阵**:
| 子命令 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `design` | 扫描 hydrate_structure 案例 | `hydrate_structure/data/cases/` | `data/design_hydrate_batch.csv` |
| `run` | 并发批量 LBM 模拟 | design CSV | `results/hydrate_batch/*/` |
| `collect` | 提取 k 回写 | VTK 文件 | `hydrate_structure/data/tables/flow.csv` |
| `clean` | 清理 VTK 释放磁盘 | 结果目录 | 仅保留 params + summary
    # 跳过标题行
    [[ "$case_name" == "case_name" ]] && continue

    CASE_DIR="${file_dir}/${case_name}"
    mkdir -p "$CASE_DIR"

    cat > "${CASE_DIR}/params.txt" << EOF
pp_mode 1
huang_init_mode 5
epsilon_huang 1.7
k2_huang 0.0
tau_huang ${tau}
Lambda_huang 0.08333
alpha_meq 1.0
cs_a 1.0
cs_b 4.0
cs_R 1.0
cs_T 0.9
cs_G -1.0
huang_rho_l 1.0
huang_rho_g 1.0
huang_u_max 0.15
huang_psi_cut 1.0e-3
theta_contact_deg ${theta_qz}
thetaA_quartz_deg ${theta_qz}
thetaA_hydrate_deg ${theta_hy}
G_ads 0.0
Gx ${Gx}
Gy 0.0
drive_mode 1
OUTPUT_EVERY ${output_every}
flow_tol_rel 1.0e-5
flow_need_consec 3
morph 0
init_eq 0
geom_file ${geom_file}
file_dir ${CASE_DIR}
EOF

    echo "[run] ${case_name} Gx=${Gx} geom=${geom_file}"
    (cd "$BINARY_DIR" && ./$(basename "$BINARY") "${CASE_DIR}/params.txt") &

    # 控制 GPU 并发数（按显存调整）
    if (( $(jobs -r | wc -l) >= 2 )); then
        wait -n
    fi
done < "$CSV_FILE"
wait
echo "[done] all cases finished"
```

### 4.3 Python 批量管理器（推荐）

```python
# scripts/run_darcy_batch.py
import subprocess, csv, os
from pathlib import Path

BINARY = "lbm_mrt/solver/mcmp_huang_porous_300x300"

def run_case(row: dict, base_params: dict):
    case_dir = Path(row['file_dir']) / row['case_name']
    case_dir.mkdir(parents=True, exist_ok=True)

    params = {**base_params, **row}
    with open(case_dir / "params.txt", 'w') as f:
        for k, v in params.items():
            f.write(f"{k} {v}\n")

    result = subprocess.run(
        [BINARY, str(case_dir / "params.txt")],
        capture_output=True, text=True, timeout=600
    )

    # 提取最后一行的 Q 或 Done 信息
    for line in result.stdout.split('\n'):
        if 'Done.' in line:
            return {"case": row['case_name'], "status": "done", "info": line.strip()}
    return {"case": row['case_name'], "status": "timeout", "info": ""}

# 使用示例
base_params = {
    'pp_mode': 1, 'huang_init_mode': 5, 'epsilon_huang': 1.7,
    'tau_huang': 1.5, 'Lambda_huang': 0.08333,
    # ... 其他固定参数
}

with open('data/design_darcy.csv') as f:
    for row in csv.DictReader(f):
        result = run_case(row, base_params)
        print(result)
```

### 4.4 后处理：提取渗透率

```python
from lbm_mrt.io.vtk_reader import read_vtk_scalars
import numpy as np

def compute_permeability(vtk_path, Gx):
    """从 VTK 提取 Darcy 渗透率 k = <ux>·ν / Gx"""
    fields, nx, ny = read_vtk_scalars(vtk_path)
    flag, ux, rho = fields['flag'], fields['ux'], fields['rho']

    fluid = (flag == 1)
    Q_total = (ux[fluid] * rho[fluid]).sum()
    u_darcy = Q_total / fluid.sum()

    nu = 1.5 / 3.0  # τ=1.5 → ν = (τ-0.5)/3
    k = u_darcy * nu / Gx
    return k

# 批量处理
import glob
for vtk in sorted(glob.glob('results/batch_*/outputdata_scmp/flow200000.vtk')):
    k = compute_permeability(vtk, Gx=1e-7)
    print(f"{vtk}: k = {k:.6f}")
```

---

## §5 当前状态与验证结果

### 默认参数（单相达西流）

| 参数 | 值 | 含义 |
|------|-----|------|
| `epsilon_huang` | 1.7 | 已验证最优 |
| `cs_G` | -1.0 | 单相时 ψ=0 退化为纯体力 |
| `tau_huang` | 1.5 | ν = (τ-0.5)/3 ≈ 0.33 |
| `huang_init_mode` | 5 | 多孔介质（保留 flag） |
| `thetaA_quartz_deg` | 25 | θ_in → θ_meas≈30° |
| `thetaA_hydrate_deg` | 25 | 与石英一致 |
| `Gx` | 1.0e-7 | 体力密度 |
| `NSTEPS` | 200000 | 稳态收敛 |

### 验证清单

| 项目 | 状态 | 备注 |
|------|------|------|
| flag=-3 水合物识别 | ✅ | 2208 节点 |
| ghost 节点 (Scheme IV ψ) | ✅ | 22935 节点, ux=0 |
| boundary 节点 (半步反弹) | ✅ | 22224 节点, ux≈流体×2.4 |
| NaN/Inf | ✅ | 全部干净 |
| 稳态收敛 | ✅ | Q 变化 <0.01%/50000步 |
| 质量守恒 | ✅ | ρ ∈ [0.99998, 1.00002] |
| 力平衡 | ✅ | Fx=Gx, Fy=0 |
| 窄喉道物理 | ✅ | 1-lu 通道高阻力 |
| 孤立孔隙 | ⚠️ | 几何特征 (88%)，非求解器问题 |

---

## §6 输出文件位置

| 路径 | 内容 |
|------|------|
| `results/hydrate_flag_fix/` | flag=-3 修复验证运行 (θ_qz=θ_hy=25°) |
| `results/steady_check/` | 200000 步稳态验证 (Q 收敛检查) |
| `results/steady_check/outputdata_scmp/flow200000.vtk` | 最终稳态场 |
| `results/steady_check/params.txt` | 运行参数 |

---

## §7 后续工作

1. **批量 Darcy 渗透率扫描**: Gx 线性验证 (k 应不依赖 Gx)
2. **多几何测试**: 不同 .plt 几何的渗透率对比
3. **网格无关性**: 300² → 400² → 600² 验证 k 收敛
4. **两相流切换**: cs_G=-1 保留，ψ≠0 激活赝势力
5. **润湿性影响**: θ 扫描对渗透率的影响（需要两相流）
6. **Q_m 监测**: 在输出中打印 Q 值以自动化稳态判断
