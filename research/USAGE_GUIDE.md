# lbm_mrt 使用手册

> 本文档面向日常使用者。所有操作均在项目根目录 `lbm_mrt/` 下执行。

---

## 1. 新架构一览

```
lbm_mrt/                       ← 项目根目录
├── configs/
│   ├── default.yaml            ← 所有物理参数的唯一真实来源（取代 params.txt）
│   └── hydrate.yaml            ← 水合物分解专用参数覆盖层
│
├── data/
│   ├── geometry/               ← Tecplot .plt 几何文件（原 geometry_cases/）
│   │   ├── case0000_clean/
│   │   ├── case0001_pf/
│   │   └── ...
│   ├── benchmarks/             ← 基准测试参考数据
│   └── design_from_geometry.csv← 算例设计矩阵（由脚本 02 生成）
│
├── results/                    ← 运行时输出（VTK、log.txt、检查点）
├── logs/                       ← 运行日志
│
├── lbm_mrt/                    ← 核心 Python 包
│   ├── core/
│   │   ├── config.py           ← 加载 YAML → 扁平 params 字典
│   │   └── paths.py            ← 集中路径常量（PROJ_ROOT、RESULTS_DIR 等）
│   ├── solver/
│   │   ├── build.py            ← CUDA 编译脚本（取代 compile.sh）
│   │   ├── src/*.cu            ← CUDA 源文件
│   │   └── include/*.h/.cuh    ← CUDA 头文件
│   ├── io/
│   │   ├── params_writer.py    ← 将 params 字典写为 params.txt
│   │   └── vtk_reader.py       ← 读取 VTK 输出文件
│   ├── runners/
│   │   ├── single_run.py       ← 单次模拟（lbm-run 入口）
│   │   └── batch_run.py        ← 批量模拟（lbm-batch 入口）
│   └── utils/
│       └── eos.py              ← Peng-Robinson EOS & 单位换算
│
├── scripts/                    ← 独立执行脚本（按编号顺序运行）
│   ├── 01_make_design.py       ← 生成参数扫描设计矩阵
│   ├── 02_make_design_porous.py← 从几何目录生成算例设计矩阵
│   └── 03_validate_benchmarks.py← 验证基准测试结果
│
└── vendor/
    └── compile_legacy.sh       ← 原始 compile.sh（仅作存档参考）
```

**核心原则：**
- 修改参数 → 只改 `configs/default.yaml`，无需触碰 C++ 源码或手写 `params.txt`
- 构建二进制 → `uv run lbm-build`
- 运行模拟 → `uv run lbm-run` 或 `uv run lbm-batch`
- 测试代码 → `uv run pytest`

---

## 2. 编译 CUDA 求解器

### 2.1 编译流场版（标准双组分两相流）

```bash
uv run lbm-build
# 默认 GPU 架构：sm_120（RTX 5090 / H100）
# 输出：lbm_mrt/solver/mcmp_sim
```

### 2.2 编译水合物版（含热场 + 浓度场 + 相变）

```bash
uv run lbm-build --hydrate
# 输出：lbm_mrt/solver/mcmp_sim_hydrate
```

### 2.3 其他编译选项

```bash
# 指定 GPU 架构（A100=sm_80，RTX 4090=sm_89，H100=sm_90）
uv run lbm-build --arch sm_89

# 开启设备调试（-g -G，性能会大幅下降，仅用于 cuda-gdb）
uv run lbm-build --debug

# 仅打印 nvcc 命令，不实际执行
uv run lbm-build --dry-run

# 同时指定水合物版本 + 特定架构
uv run lbm-build --hydrate --arch sm_80
```

---

## 3. 运行模拟

### 3.1 单次模拟（`lbm-run`）

```bash
# 最简用法：使用 default.yaml 默认参数 + 指定几何文件
uv run lbm-run \
    --case-name my_case_01 \
    --geom data/geometry/case0001_pf/geometry_case0001.plt

# 输出目录：results/my_case_01/
#   ├── params.txt          ← 本次实际使用的参数（可复查）
#   ├── log.txt             ← 求解器控制台输出
#   ├── outputdata_eq*.vtk  ← 平衡阶段 VTK 快照
#   ├── outputdata_flow*.vtk← 流动阶段 VTK 快照
#   └── ckpt/               ← 检查点文件
```

#### 在命令行覆盖参数

```bash
# 修改水饱和度、接触角、驱动力（KEY=VALUE 追加到命令末尾）
uv run lbm-run \
    --case-name test_Sw07 \
    --geom data/geometry/case0003_pf/geometry_case0003.plt \
    Sw=0.7 thetaA_quartz_deg=45 Gx=2e-8
```

#### 从头重跑（忽略现有检查点）

```bash
uv run lbm-run --case-name my_case_01 --resume 0 \
    --geom data/geometry/case0001_pf/geometry_case0001.plt
```

#### 使用水合物二进制

```bash
uv run lbm-run \
    --case-name hydrate_test \
    --geom data/geometry/case0001_pf/geometry_case0001.plt \
    --app lbm_mrt/solver/mcmp_sim_hydrate \
    --config configs/hydrate.yaml
```

### 3.2 批量模拟（`lbm-batch`）

```bash
# 读取 data/design_from_geometry.csv，逐行运行
uv run lbm-batch

# 指定自定义设计表
uv run lbm-batch --csv data/my_custom_design.csv

# 几何文件不存在时直接跳过该算例（而非报错）
uv run lbm-batch --strict-geometry

# 从头重跑所有算例
uv run lbm-batch --resume 0
```

批量运行结束后，`results/` 下会出现以 `case_name` 命名的子目录：

```
results/
├── geom0001_pf_Sh0.45_Sw0.30/
│   ├── params.txt
│   ├── log.txt
│   ├── outputdata_flow*.vtk
│   └── ckpt/
├── geom0002_pf_Sh0.38_Sw0.50/
└── ...
```

---

## 4. 脚本执行流（`scripts/`）

脚本按编号顺序执行，彼此独立，可按需选用。

### `01_make_design.py` — 参数扫描矩阵

扫描形态参数（`r_mid` / `coat_thick`），生成不含几何文件引用的设计矩阵，适合**参数化几何**（`morph=1/2/3`）批量扫描。

```bash
uv run python scripts/01_make_design.py
# 输出：data/design_geom_scan.csv
```

### `02_make_design_porous.py` — 几何驱动矩阵

扫描 `data/geometry/` 目录下所有 `caseXXXX_*` 几何案例，对每个案例展开 `Sw = 0.0, 0.1, …, 1.0` 共 11 个水饱和度，生成用于 `lbm-batch` 的标准设计矩阵。

```bash
uv run python scripts/02_make_design_porous.py
# 输出：data/design_from_geometry.csv（供 lbm-batch 直接使用）
```

### `03_validate_benchmarks.py` — 基准验证

读取 VTK 输出，对比解析解，验证五类基准测试是否通过（需用 `mcmp_sim_hydrate` 运行才有温度/浓度场）。

```bash
# 验证单个基准
uv run python scripts/03_validate_benchmarks.py --bm BM1 --dir results/my_hydrate_case

# 一次验证全部（BM1~BM5）
uv run python scripts/03_validate_benchmarks.py --all --dir results/my_hydrate_case
```

| 基准 | 含义 | 判断标准 |
|------|------|----------|
| BM1 | 纯热扩散稳态 | 温度剖面线性残差 < 0.1% |
| BM2 | 共轭热传递 | dT 梯度比接近 λ_s/λ_f（误差 < 10%）|
| BM3 | 反应-扩散 | 特征长度误差 < 5% |
| BM4 | VOP 质量守恒 | 质量漂移 < 0.5% |
| BM5 | 全耦合趋势 | Vh 单调递减、T/Cm 变化符合物理预期 |

---

## 5. 参数修改指南

**所有参数只需修改 `configs/default.yaml`，不需要手动写 `params.txt` 或改 C++ 源码。**

### 5.1 YAML 文件结构

```yaml
# configs/default.yaml（节选）

geometry:
  morph: 1          # 几何形态：1=孔隙填充, 2=涂层, 3=混合
  geom_file: ""     # 为空时使用参数化几何；非空时从 .plt 读取

fluids:
  Sw: 0.3           # 初始水饱和度 [0, 1]

driving:
  Gx: 1.5e-8        # 体力加速度（格子单位）

relaxation:
  tau_p_a: 1.0      # A 相弛豫时间，必须 > 0.5
  tau_p_b: 0.775    # B 相弛豫时间，必须 > 0.5

checkpoint:
  enable: true
  every: 50000      # 每多少步保存一次检查点
  keep: 3           # 保留最近 N 份

convergence:
  eq:
    max_steps: 200000   # 平衡阶段最大步数
  flow:
    max_steps: 5000000  # 流动阶段最大步数
```

### 5.2 典型修改场景

**换接触角（润湿性）：**
```yaml
# configs/default.yaml
wettability:
  thetaA_quartz_deg: 60    # 改这里，不需要改任何 C++ 文件
  thetaA_hydrate_deg: 100
```

**换驱动模式（边界压差 → 体力）：**
```yaml
driving:
  drive_mode: 1   # 1=体力(Gx,Gy); 2=边界压差/速度
  Gx: 2.0e-8
```

**临时覆盖单次运行的参数（不改 YAML）：**
```bash
uv run lbm-run --case-name quick_test --geom data/geometry/case0001_pf/geometry_case0001.plt \
    Sw=0.9 thetaA_quartz_deg=60 Gx=3e-8
```

**为某批算例使用水合物专用配置：**
```bash
uv run lbm-batch \
    --csv data/design_from_geometry.csv \
    --config configs/hydrate.yaml \
    --app lbm_mrt/solver/mcmp_sim_hydrate
```

### 5.3 参数与 C++ 字段对应关系

YAML 中的参数键名与 C++ `sim_utils.cu::load_params_txt()` 读取的键名一一对应。
少数特殊映射：

| YAML / params.txt 键 | C++ `RuntimeParams` 字段 |
|----------------------|--------------------------|
| `rhoA_ini_h` | `rhoA_hi` |
| `rhoA_ini_l` | `rhoA_lo` |
| `rhoB_ini_h` | `rhoB_hi` |
| `rhoB_ini_l` | `rhoB_lo` |
| `ENABLE_CKPT` | `ENABLE_CKPT` (bool) |

`GAB`、`GBA`、`sigmaA` 在旧 `params.txt` 中缺失（使用 C++ 默认值），现已收录至 `default.yaml` 中可直接修改。

---

## 6. 运行测试

```bash
# 运行全部单元测试
uv run pytest

# 仅运行配置相关测试
uv run pytest tests/test_config.py -v

# 仅运行物理工具测试
uv run pytest tests/test_eos.py -v
```

---

## 7. 快速索引

| 我想做… | 命令 / 文件 |
|---------|------------|
| 改模拟参数 | 编辑 `configs/default.yaml` |
| 编译求解器 | `uv run lbm-build` |
| 编译水合物版 | `uv run lbm-build --hydrate` |
| 单次运行 | `uv run lbm-run --case-name X --geom Y` |
| 批量运行 | `uv run lbm-batch --csv data/design_from_geometry.csv` |
| 生成算例设计矩阵 | `uv run python scripts/02_make_design_porous.py` |
| 验证基准测试 | `uv run python scripts/03_validate_benchmarks.py --all --dir results/X` |
| 运行单元测试 | `uv run pytest` |
| 查看求解器输出 | `cat results/<case_name>/log.txt` |
| 查看参数实际值 | `cat results/<case_name>/params.txt` |
