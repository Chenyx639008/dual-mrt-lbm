# huang_mrt_2d — AI 入口

> **当前状态**: 双轨架构 (CUDA 生产 ↔ JAX 镜像) Phase 1-5 完成 · 生产就绪 (ε=1.7) · MCMP 已纳入 | 2026-07-22
> **一键入口**: `uv run lbm run scmp_cs_huang_256`
> **框架手册**: [`research/FRAMEWORK_GUIDE.md`](research/FRAMEWORK_GUIDE.md) | **可行性评估**: [`research/unified_framework_feasibility.md`](research/unified_framework_feasibility.md) | **研究索引**: [`research/INDEX.md`](research/INDEX.md) | **CUDA 内核**: [`lbm_mrt/solver/CLAUDE.md`](lbm_mrt/solver/CLAUDE.md)

双轨 MRT-LBM 两相流 + 甲烷水合物分解 + D2Q9 SCMP — C++17/CUDA 生产轨 + JAX 镜像验证轨 + Python 统一调度。

---

## 快速命令

```bash
uv sync --all-groups                          # 安装全部依赖
uv run lbm-build --huang --huang-unified      # 编译所有 SCMP 二进制
uv run lbm models                             # 列出预注册模型
uv run lbm run scmp_cs_huang_256 --steps 50000  # 运行
uv run pytest && uv run ruff check . && uv run ruff format .  # 测试 + lint
```

---

## 核心约定（AI 必须知道）


- **配置流**: YAML → `load_config()` → flat dict → `params.txt` → `load_params_txt()` → `__constant__` 设备内存。水合物运行时叠加 `hydrate.yaml`。SCMP 统一路径：`ModelDefinition.to_params_dict()` → `params.txt`。
- **网格编译期常量**: `NX`/`NY` 是 `LBM.h` 中 `constexpr`，改网格需重编译。统一 Huang 二进制用 `d_nx_active`/`d_ny_active` 运行时网格（Phase 2 进行中）。
- **params.txt 是唯一边界**: Python → C++ 通信仅通过 `params.txt`。key 名必须精确匹配 `RuntimeParams` 字段名，不匹配的被静默忽略。
- **VTK Legacy Binary**: 输出为 big-endian 二进制 VTK（`results/<case>/outputdata_eq/flow*.vtk`），读入需用 `lbm_mrt/io/vtk_reader.py`。
- **JAX 独立于 CUDA**: `jax_lbm/` 是独立 D2Q9 BGK 实现，不依赖 CUDA 编译，用于 autograd 敏感度分析。
- **Python**: uv 管理（pyproject.toml + uv.lock），GPU 目标 sm_120（RTX 5090 / H100）。

---

## 目录导航

| 目录 | 说明 | 详细索引 |
|------|------|---------|
| `lbm_mrt/solver/` | CUDA C++17 核心（内核、EOS、边界、水合物） | [`CLAUDE.md`](lbm_mrt/solver/CLAUDE.md) |
| `lbm_mrt/unified/` | 🆕 统一框架（models / CLI / runner） | — |
| `jax_lbm/` | 🆕 JAX D2Q9 BGK + Shan-Chen 验证工具 | — |
| `configs/` | YAML 配置（MCMP 默认 + 水合物叠加 + SCMP） | [`INDEX.md`](configs/INDEX.md) |
| `scripts/` | 工作流入口（01-06）+ 批量脚本 | [`README.md`](scripts/README.md) |
| `research/` | 科学文档、水合物指南、统一框架规划 | [`INDEX.md`](research/INDEX.md) |
| `validation/` | contact_angle / mesh_convergence / surface | — |
| `data/` | 几何 (.plt)、benchmark、design CSV | — |
| `results/` | 模拟输出（gitignored） | — |

---

## 数据流

| 流 | 路径 |
|----|------|
| **SCMP 统一** | `ModelDefinition` → `to_params_dict()` → `params.txt` → `mcmp_huang*` → VTK |
| **MCMP 两相** | `configs/default.yaml` → `load_config()` → `params.txt` → `mcmp_sim` → VTK |
| **MCMP 水合物** | `default.yaml` + `hydrate*.yaml`（叠加）→ `params.txt` → `mcmp_sim_hydrate` → VTK |
| **JAX 验证** | `d2q9_bgk.py`（独立运行，不经过 params.txt）→ autograd 梯度 |

---

## 常见陷阱（Top 5）

1. **ε 默认值**: 论文 ε=−2/3 (k₁=1/12) 在此代码中不稳定。必须用经验值 **ε=1.7**。参数覆盖时用 `epsilon_huang` 而非 `k1_huang`（后者不被 params.txt 识别）。
2. **网格改完没重编译**: `NX`/`NY` 在 `LBM.h` 中是 `constexpr`，改数值后必须 `lbm-build` 重新编译。
3. **params.txt key 命名**: 必须精确匹配 C++ `RuntimeParams` 字段名（如 `tau_p_a` 不是 `tauA`）。不匹配的 key 被 `load_params_txt()` 静默忽略，不报错。
4. **水合物 `#ifdef` 隔离**: hydrate 相关代码全部在 `#ifdef HYDRATE_ENABLE` 内。用 `mcmp_sim` 跑 hydrate 参数会静默忽略。
5. **VTK 字节序**: 输出为 big-endian legacy binary。用 `vtk_reader.py` 读取，不要直接 parse。

---

## 相关 AI 工具

- **记忆管理**: 使用 `cyx_memory` skill → 更新 CLAUDE.md + 子索引 + 交叉引用校验
- **代码分析**: 使用 `/understand` → 生成交互式知识图谱分析 CUDA/Python 代码结构
- **论文写作**: 使用 `nature-polishing` / `cyx_*_review` skills → 学术写作润色与审稿
- **版本发布**: 使用 `cyx_git_push` skill → 自动提交 + tag + push
