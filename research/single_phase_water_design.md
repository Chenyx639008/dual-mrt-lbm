# 单相水流动批量模拟设计方案（初步设想）

> **定位**：基于已验证的 Huang & Wu (2016) SCMP-MRT 求解器，构建单相水在复杂水合物沉积物多孔介质中的批量流动模拟管线。输入 `.plt` 几何文件，自动生成参数组合并批量运行，输出结构化结果供后续生成与评价研究使用。
>
> **撰写日期**：2026-05-14
>
> **状态**：初步设想 / 待评审

---

## 一、核心需求

### 1.1 物理场景

| 要素 | 说明 |
|------|------|
| 流体 | **纯水（单相，无气体）** |
| 固体骨架 | 石英颗粒（来自 `.plt` 几何文件） |
| 水合物相 | 静态固体（赋存形态由几何文件定义，不参与相变） |
| 驱动力 | 体积力 $G_x$（压力梯度等效） |
| 流动状态 | 稳态 Darcy 渗流 |
| 输出关注 | 速度场、渗透率、迂曲度、流动路径可视化 |

### 1.2 设计目标

1. **单相**：域内仅有液态水一种流体，无气体相、无相界面
2. **批量自动化**：输入几何文件 → 自动生成参数组合 CSV → 批量调用 `lbm-batch` → 结构化结果目录
3. **几何驱动**：从 Tecplot `.plt` 文件读取 `pointsflag`（固体/流体/水合物标志），不依赖 `build_circle_array` 参数化生成
4. **基于已验证求解器**：使用 Huang SCMP 路径（`mcmp_huang_256`），因为该路径已通过 CS-EOS、Maxwell 共存、Laplace 定律验证，代码质量有保证
5. **可复现**：每个 case 目录包含完整 `params.txt` + `geometry_case.plt` + VTK 输出

---

## 二、为什么用 SCMP 跑单相？

### 2.1 直觉冲突

Huang SCMP 本质是**单组分两相**（纯物质气-液共存），按设计不适合单相流。

### 2.2 可行性论证

SCMP 可以通过以下参数组合**退化到单相**：

| 参数 | 设置 | 效果 |
|------|------|------|
| `cs_T` | $T \gg T_c$（如 `cs_T=0.15`，$T_r \approx 1.6$） | CS-EOS 进入超临界区，无气-液分离 |
| `huang_init_mode` | `2`（flat interface）或新增 `3`（uniform density） | 全域均匀密度初始化 |
| `huang_rho_g = huang_rho_l` | 均设为超临界均匀密度 $\rho_0$ | 无密度梯度 → 无伪势力 → 无相分离 |
| `cs_G` | `0` 或极小值 | 关闭分子间作用力 |
| `k1_huang` | 任意值（不起作用） | 表面张力项 $Q_m \propto |F_{mol}|^2 = 0$ |

在这种退化模式下：
- 伪势 $\psi \approx \text{const}$ → 分子力 $F_{mol} \approx 0$
- $Q_m$ 项归零
- MRT 碰撞退化为单相牛顿流体 LBM
- 黏度由 $\tau$ 控制：$\nu = c_s^2(\tau - 0.5)$

**优点**：
- 复用已验证的 MRT 碰撞内核和 Huang 代码路径
- 无需维护两套求解器
- 未来需要多相时，仅需改参数即可切换

**缺点**：
- 256×256 网格（`HUANG_256_BUILD`）与几何文件原始尺寸（339×212）不匹配
- 需要解决网格尺寸适配问题

---

## 三、架构设计

### 3.1 整体管线

```
┌──────────────────────────────────────────────────────────────────┐
│                    单相水批量模拟管线                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ ① 几何输入                                                       │
│   data/geometry/case0003_pf/geometry_case0003.plt                │
│   │                                                              │
│   ▼                                                              │
│ ② 几何预处理（Python）                                           │
│   · 解析 .plt → pointsflag[NX][NY]                               │
│   · 重采样/裁剪到 256×256（如需要）                               │
│   · 写入标准化 .plt 供求解器读取                                  │
│   │                                                              │
│   ▼                                                              │
│ ③ 设计矩阵生成（Python → CSV）                                    │
│   · 扫参维度：Gx, τ, 几何文件, 水合物饱和度                      │
│   · 每行 = 一个 case，自动生成 case_name                         │
│   │                                                              │
│   ▼                                                              │
│ ④ 批量运行（lbm-batch）                                          │
│   · 调用 mcmp_huang_256（需适配 339×212 或统一 256×256）         │
│   · 每个 case 独立目录，含 params.txt + geometry_case.plt        │
│   │                                                              │
│   ▼                                                              │
│ ⑤ 后处理 & 汇总（Python）                                        │
│   · 提取稳态速度场 → 计算 K_eff, 迂曲度                          │
│   · 生成汇总 CSV + 对比图                                        │
│   · 输出到 results/single_phase_water_<ts>/                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 网格尺寸问题（关键决策）

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A: 改 SCMP 到 339×212** | 新增 `HUANG_339_212_BUILD` 宏 | 与几何文件原生匹配 | 需重新验证 Huang 参数；编译变体增多 |
| **B: 几何下采样到 256×256** | Python 预处理将 339×212 裁剪/填充/插值到 256² | SCMP 代码零改动 | 丢失几何精度；填充区域引入边界 artifact |
| **C: 统一 MCMP 跑单相** | 用 legacy `mcmp_sim`（已支持 339×212），Sw=1 + 关闭界面力 | 网格原生匹配、零新代码 | 非 Huang 路径；需单独验证单相正确性 |

**推荐优先级**：**C > A > B**

理由：
- Legacy `mcmp_sim` 已经过 benchmark 验证，原生 339×212
- Sw=1 + kappa=0 + GAB=GBA=0 即可退化到单相
- 无需修改 C++ 代码，只需配置参数和初始条件
- 如果将来需要两相（气-水共存），改 Sw < 1 即可立即切换

如果必须使用 Huang 路径（例如需要其 MRT 参数化或发表一致性），则选方案 A。

---

## 四、参数设计

### 4.1 单相水基准参数（方案 C：legacy MCMP）

```yaml
# configs/single_phase_water.yaml
project:
  name: single_phase_water
  notes: "单相水在复杂多孔介质中的 Darcy 渗流"

geometry:
  morph: 0              # 从 .plt 文件读取
  geom_file: "geometry_case.plt"

fluids:
  Sw: 1.0               # 全域水 → 单相
  water_seed: 1234567
  rhoA_ini_h: 1.0       # 均匀密度（单相无密度差）
  rhoA_ini_l: 1.0
  rhoB_ini_h: 0.0001    # B 组分极低密度（等效不存在）
  rhoB_ini_l: 0.0001

wettability:
  thetaA_quartz_deg: 30
  thetaA_hydrate_deg: 80
  GBw_quartz: 0.0
  GBw_hydrate: 0.0

scenario:
  init_eq: 0            # 跳过相分离阶段
  drive: 2

driving:
  Gx: 1.0e-7            # 扫参维度 ①
  Gy: 0.0
  drive_mode: 1

relaxation:
  tau_p_a: 1.0          # 扫参维度 ②（控制黏度/Re）
  tau_p_b: 0.775
  kappa: 0.0             # 关闭界面力
  GAB: 0.0
  GBA: 0.0
  sigmaA: 0.0

convergence:
  eq:
    max_steps: 0         # 跳过 eq 阶段
  flow:
    tol_rel: 1.0e-6
    need_consec: 3
    max_steps: 500000
```

### 4.2 扫参矩阵（设计 CSV 列）

| 列名 | 含义 | 典型取值 |
|------|------|----------|
| `geometry_src` | 几何文件路径 | `data/geometry/case0003_pf/geometry_case0003.plt` |
| `Gx` | 驱动力（体积加速度） | `[1e-8, 5e-8, 1e-7, 5e-7, 1e-6]` |
| `tau_p_a` | 松弛时间（黏度） | `[0.8, 1.0, 1.5, 2.0]` |
| `Sw` | 水饱和度 | `1.0`（单相固定） |
| `case_name` | 自动生成 | `geom0003_Gx1e-7_tau1.0` |

### 4.3 扫参策略（按研究阶段）

| 阶段 | 内容 | case 数 | 预估机时 |
|------|------|:------:|:--------:|
| **Smoke** | 1 几何 × 1 Gx × 1 τ | 1 | ~5 min |
| **Gx sweep** | 1 几何 × 5 Gx × 1 τ | 5 | ~25 min |
| **τ sweep** | 1 几何 × 1 Gx × 4 τ | 4 | ~20 min |
| **Geometry sweep** | 9 几何 × 1 Gx × 1 τ | 9 | ~45 min |
| **Full factorial** | 9 几何 × 5 Gx × 4 τ | 180 | ~15 h |

---

## 五、实现计划

### 5.1 Phase 1：Smoke test（0.5 天）

1. 创建 `configs/single_phase_water.yaml`
2. 手写一个 params.txt 验证单相退化正确性：
   - 均匀初始密度 → 无相分离
   - 施加 Gx → 产生合理的速度场（抛物线型或 Darcy 型）
   - 检查 max|u| 在合理范围
3. 确认 VTK 输出中 `rho` 场均匀、`ux` 场符合预期

### 5.2 Phase 2：几何适配（0.5 天）

1. 确认 `.plt` 文件格式与 `load_geometry_from_plt()` 兼容
2. 测试几何文件读取 → `pointsflag` 正确识别固体/流体边界
3. 检查边界处速度是否为 0（无滑移在固体表面）

### 5.3 Phase 3：批量管线（1 天）

1. 编写 `scripts/08_generate_single_phase_design.py`：
   - 读入几何文件列表
   - 笛卡尔积生成参数组合
   - 输出 `data/design_single_phase_water.csv`
2. 测试 `lbm-batch` 端到端：
   ```bash
   uv run lbm-batch --csv data/design_single_phase_water.csv --config configs/single_phase_water.yaml
   ```

### 5.4 Phase 4：后处理（1 天）

1. 编写 `scripts/09_analyze_single_phase.py`：
   - 从 VTK 提取稳态速度场
   - 计算有效渗透率 $K_{eff} = \frac{\nu \cdot \langle u_x \rangle}{G_x}$
   - 计算迂曲度 $\tau_{tor} = \langle |u| \rangle / \langle u_x \rangle$
   - 生成汇总 CSV + 流线图

### 5.5 Phase 5：扩展到 Huang 路径（可选，1-2 天）

如果后续需要用 Huang SCMP 路径（例如与两相结果对比），则：
1. 新增 `HUANG_339_212_BUILD` 或统一几何到 256×256
2. 验证超临界单相退化的正确性
3. 与 MCMP 单相结果交叉验证

---

## 六、输出结构

```
results/
└── single_phase_water_20260515_120000/
    ├── design.csv                          # 本次运行的设计矩阵
    ├── summary.csv                         # 汇总结果（K_eff, 迂曲度, max|u|, ...）
    ├── geom0003_pf/
    │   ├── Gx1e-7_tau1.0/
    │   │   ├── params.txt
    │   │   ├── geometry_case.plt
    │   │   ├── outputdata_eq/
    │   │   └── run_summary.txt
    │   ├── Gx5e-7_tau1.0/
    │   └── ...
    ├── geom0004_gc/
    └── ...
```

---

## 七、关键待确认问题

| # | 问题 | 影响 |
|---|------|------|
| 1 | `.plt` 文件的 flag 编码规范：固体=? 流体=? 水合物=? | 决定初始化和边界条件 |
| 2 | 几何文件原始分辨率是否确为 339×212？ | 决定是否可复用 `mcmp_sim` |
| 3 | 单相时水合物节点应设为什么 flag？静态固体（无滑移）？ | 决定 `pointsflag` 中水合物节点的处理 |
| 4 | 是否需要渗透率各向异性（$K_{xx}$ vs $K_{yy}$）？ | 决定是否需要 Gx + Gy 双方向扫参 |
| 5 | 是否需要在 SCMP 中新加 `huang_init_mode=3`（uniform density）？ | 决定初始化代码改动量 |
| 6 | 水合物沉积物"生成"与"评价"的具体指标？ | 决定后处理需要提取哪些量 |

---

## 八、与现有工作关系

| 现有组件 | 复用方式 |
|----------|----------|
| `mcmp_sim` 二进制 | 直接使用（方案 C） |
| `lbm_mrt/runners/batch_run.py` | 直接使用 |
| `lbm_mrt/runners/single_run.py` | 直接使用 |
| `lbm_mrt/core/config.py` | 新增 `single_phase_water.yaml` |
| `lbm_mrt/io/vtk_reader.py` | 直接使用 |
| `lbm_mrt/io/params_writer.py` | 直接使用 |
| `scripts/01_make_design.py` | 参考模式 |
| `scripts/05_run_porous_hydrate_case.py` | 参考模式 |

**不依赖**：
- ❌ `hydrate.cu` / `hydrate_vop.cu`（无水合物相变物理）
- ❌ `-DHYDRATE_ENABLE`（不需要水合物编译宏）
- ❌ Huang SCMP 验证套件（除非选方案 A）

---

## 九、后续扩展路径

```
单相水 Darcy 流（本方案）
    │
    ├─→ ① 多几何统计：9+ 几何 → K_eff 分布 → 等效 REV 参数
    │
    ├─→ ② 非牛顿流体：τ 随剪切率变化 → 幂律/Cross 模型
    │
    ├─→ ③ 示踪剂输运：D2Q5 浓度场 → 突破曲线 → 弥散系数
    │
    ├─→ ④ 两相驱替：Sw < 1 → 气-水两相 → 相对渗透率曲线
    │
    └─→ ⑤ 热-流耦合：加热边界 → 自然对流 → Nu-Ra 关系
```
